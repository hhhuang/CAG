import torch
import pandas as pd
import argparse
import os
import json
from time import time
from sentence_transformers import SentenceTransformer, util
from transformers import BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM
from transformers.cache_utils import DynamicCache
import random
import logging 


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def get_env():
    env_dict = {}
    with open(file=".env" if os.path.exists(".env") else "env", mode="r") as f:
        for line in f:
            key, value = line.strip().split("=")
            env_dict[key] = value.strip('"')
    return env_dict

def validate_env_variables():
    required_keys = ["HF_TOKEN"]
    env = get_env()
    for key in required_keys:
        if key not in env or not env[key]:
            raise ValueError(f"Missing required environment variable: {key}")
    return env

"""Hugging Face Llama model"""
env = validate_env_variables()
HF_TOKEN = env["HF_TOKEN"]

global model_name, model, tokenizer
global rand_seed


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")


"""KV Cache test"""
# Allowlist the DynamicCache class
torch.serialization.add_safe_globals([DynamicCache])
torch.serialization.add_safe_globals([set])


def generate(
    model,
    input_ids: torch.Tensor,
    past_key_values,
    max_new_tokens: int = 300
) -> torch.Tensor:
    """
    Generate text with greedy decoding.

    Args:
        model: HuggingFace model with automatic device mapping
        input_ids: Input token ids
        past_key_values: KV Cache for knowledge
        max_new_tokens: Maximum new tokens to generate
    """

    embed_device = model.model.embed_tokens.weight.device

    origin_ids = input_ids
    input_ids = input_ids.to(embed_device)

    output_ids = input_ids.clone()
    next_token = input_ids

    with torch.no_grad():
        for _ in range(max_new_tokens):
            outputs = model(
                input_ids=next_token, 
                past_key_values=past_key_values,
                use_cache=True
            )
            next_token_logits = outputs.logits[:, -1, :]
            next_token = next_token_logits.argmax(dim=-1).unsqueeze(-1)
            next_token = next_token.to(embed_device)

            past_key_values = outputs.past_key_values

            output_ids = torch.cat([output_ids, next_token], dim=1)

            if next_token.item() in model.config.eos_token_id:
                break
    return output_ids[:, origin_ids.shape[-1]:]





def preprocess_knowledge(
    model,
    tokenizer,
    prompt: str,
) -> DynamicCache:
    """
    Prepare knowledge kv cache for CAG.
    Args:
        model: HuggingFace model with automatic device mapping
        tokenizer: HuggingFace tokenizer
        prompt: The knowledge to preprocess, which is basically a prompt

    Returns:
        DynamicCache: KV Cache
    """
    embed_device = model.model.embed_tokens.weight.device
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(embed_device)
    past_key_values = DynamicCache()
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            use_cache=True,
            output_attentions=False,
            output_hidden_states=False
        )
    return outputs.past_key_values


def write_kv_cache(kv: DynamicCache, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    """
    Write the KV Cache to a file.
    """
    torch.save(kv, path)


def clean_up(kv: DynamicCache, origin_len: int):
    """
    Truncate the KV Cache to the original length.
    """
    for i in range(len(kv.key_cache)):
        kv.key_cache[i] = kv.key_cache[i][:, :, :origin_len, :]
        kv.value_cache[i] = kv.value_cache[i][:, :, :origin_len, :]


def read_kv_cache(path: str) -> DynamicCache:
    """
    Read the KV Cache from a file. If the cache file is invalid or empty, return None.
    """
    if os.path.exists(path) and os.path.getsize(path) > 0:
        kv = torch.load(path, weights_only=True)
        return kv
    else:
        # Regenerate cache if it doesn't exist or is too small
        return None


"""Sentence-BERT for evaluate semantic similarity"""
bert_model = SentenceTransformer('all-MiniLM-L6-v2')  # Use a lightweight sentence-transformer

def get_bert_similarity(response, ground_truth):
    # Encode the query and text
    query_embedding = bert_model.encode(response, convert_to_tensor=True)
    text_embedding = bert_model.encode(ground_truth, convert_to_tensor=True)

    # Compute the cosine similarity between the query and text
    cosine_score = util.pytorch_cos_sim(query_embedding, text_embedding)

    return cosine_score.item()


def prepare_kvcache(documents, filepath: str, answer_instruction: str = None, batch_size: int = 1000):
    # Prepare the knowledges kvcache

    if answer_instruction is None:
        answer_instruction = "Answer the question with a super short answer."
    knowledges = f"""
    <|begin_of_text|>
    <|start_header_id|>system<|end_header_id|>
    You are an assistant for giving short answers based on given context.<|eot_id|>
    <|start_header_id|>user<|end_header_id|>
    Context information is bellow.
    ------------------------------------------------
    {documents}
    ------------------------------------------------
    {answer_instruction}
    Question:
    """
    # Get the knowledge cache
    t1 = time()
    # Process large documents in batches
    doc_chunks = [documents[i:i + batch_size] for i in range(0, len(documents), batch_size)]
    kv_caches = []
    
    for chunk in doc_chunks:
        chunk_kv = preprocess_knowledge(model, tokenizer, chunk)
        kv_caches.append(chunk_kv)
    
    # Merge KV caches
    final_kv = merge_kv_caches(kv_caches)
    write_kv_cache(final_kv, filepath)
    t2 = time()
    logger.info(f"KV cache prepared in {t2 - t1:.2f} seconds.")
    return final_kv, t2 - t1


def get_kis_dataset(filepath: str):
    df = pd.read_csv(filepath)
    dataset = zip(df['sample_question'], df['sample_ground_truth'])
    text_list = df["ki_text"].to_list()

    return text_list, dataset


def parse_squad_data(raw):
    dataset = {"ki_text": [], "qas": []}

    for k_id, data in enumerate(raw['data']):
        article = []
        for p_id, para in enumerate(data['paragraphs']):
            article.append(para['context'])
            for qa in para['qas']:
                ques = qa['question']
                answers = [ans['text'] for ans in qa['answers']]
                dataset['qas'].append({"title": data['title'], "paragraph_index": tuple((k_id, p_id)), "question": ques, "answers": answers})
        dataset['ki_text'].append({"id": k_id, "title": data['title'], "paragraphs": article})

    return dataset


def get_squad_dataset(filepath: str, max_knowledge: int = None,
                      max_paragraph: int = None, max_questions: int = None):
    # Open and read the JSON file
    with open(filepath, 'r') as file:
        data = json.load(file)
    # Parse the SQuAD data
    parsed_data = parse_squad_data(data)

    print("max_knowledge", max_knowledge, "max_paragraph", max_paragraph, "max_questions", max_questions)

    # Set the limit Maximum Articles, use all Articles if max_knowledge is None or greater than the number of Articles
    max_knowledge = max_knowledge if max_knowledge is not None and max_knowledge < len(parsed_data['ki_text']) else len(parsed_data['ki_text'])

    # Shuffle the Articles and Questions
    if rand_seed is not None:
        random.seed(rand_seed)
        random.shuffle(parsed_data["ki_text"])
        random.shuffle(parsed_data["qas"])
        k_ids = [i['id'] for i in parsed_data["ki_text"][:max_knowledge]]

    text_list = []
    # Get the knowledge Articles for at most max_knowledge, or all Articles if max_knowledge is None
    for article in parsed_data['ki_text'][:max_knowledge]:
        max_para = max_paragraph if max_paragraph is not None and max_paragraph < len(article['paragraphs']) else len(article['paragraphs'])
        text_list.append(article['title'])
        text_list.append('\n'.join(article['paragraphs'][0:max_para]))

    # Check if the knowledge id of qas is less than the max_knowledge
    questions = [qa['question'] for qa in parsed_data['qas'] if qa['paragraph_index'][0] in k_ids and (max_paragraph is None or qa['paragraph_index'][1] < max_paragraph)]
    answers = [qa['answers'][0] for qa in parsed_data['qas'] if qa['paragraph_index'][0] in k_ids and (max_paragraph is None or qa['paragraph_index'][1] < max_paragraph)]

    dataset = zip(questions, answers)

    return text_list, dataset


def get_hotpotqa_dataset(filepath: str, max_knowledge: int = None):
    # Open and read the JSON
    with open(filepath, "r") as file:
        data = json.load(file)

    if rand_seed is not None:
        random.seed(rand_seed)
        random.shuffle(data)

    questions = [qa['question'] for qa in data]
    answers = [qa['answer'] for qa in data]
    dataset = zip(questions, answers)

    if max_knowledge is None:
        max_knowledge = len(data)
    else:
        max_knowledge = min(max_knowledge, len(data))

    text_list = []
    for i, qa in enumerate(data[:max_knowledge]):
        context = qa['context']
        context = [c[0] + ": \n" + "".join(c[1]) for c in context]
        article = "\n\n".join(context)

        text_list.append(article)

    return text_list, dataset


def kvcache_test(args: argparse.Namespace):
    answer_instruction = None
    if args.dataset == "kis_sample":
        datapath = "./datasets/rag_sample_qas_from_kis.csv"
        text_list, dataset = get_kis_dataset(datapath)
    if args.dataset == "kis":
        datapath = "./datasets/synthetic_knowledge_items.csv"
        text_list, dataset = get_kis_dataset(datapath)
    if args.dataset == "squad-dev":
        datapath = "./datasets/squad/dev-v1.1.json"
        text_list, dataset = get_squad_dataset(datapath, max_knowledge=args.maxKnowledge, max_paragraph=args.maxParagraph, max_questions=args.maxQuestion)
    if args.dataset == "squad-train":
        datapath = "./datasets/squad/train-v1.1.json"
        text_list, dataset = get_squad_dataset(datapath, max_knowledge=args.maxKnowledge, max_paragraph=args.maxParagraph, max_questions=args.maxQuestion)
        answer_instruction = "Answer the question with a super short answer."
    if args.dataset == "hotpotqa-dev":
        datapath = "./datasets/hotpotqa/hotpot_dev_fullwiki_v1.json"
        text_list, dataset = get_hotpotqa_dataset(datapath, args.maxKnowledge)
        answer_instruction = "Answer the question with a super short answer."
    if args.dataset == "hotpotqa-test":
        datapath = "./datasets/hotpotqa/hotpot_test_fullwiki_v1.json"
        text_list, dataset = get_hotpotqa_dataset(datapath, args.maxKnowledge)
        answer_instruction = "Answer the question with a super short answer."
    if args.dataset == "hotpotqa-train":
        datapath = "./datasets/hotpotqa/hotpot_train_v1.1.json"
        text_list, dataset = get_hotpotqa_dataset(datapath, args.maxKnowledge)
        answer_instruction = "Answer the question with a super short answer."

    kvcache_path = "./data_cache/cache_knowledges.pt"

    knowledges = '\n\n\n\n\n\n'.join(text_list)
    knowledge_cache, prepare_time = prepare_kvcache(knowledges, filepath=kvcache_path, answer_instruction=answer_instruction)
    kv_len = knowledge_cache.key_cache[0].shape[-2]
    print(f"KVcache prepared in {prepare_time} seconds")
    with open(args.output, "a") as f:
        f.write(f"KVcache prepared in {prepare_time} seconds\n")

    results = {
        "cache_time": [],
        "generate_time": [],
        "similarity": [],
        "prompts": [],
        "responses": []
    }

    dataset = list(dataset)  # Convert the dataset to a list

    max_questions = min(len(dataset), args.maxQuestion) if args.maxQuestion is not None else len(dataset)
    # Retrieve the knowledge from the vector database
    for id, (question, ground_truth) in enumerate(dataset[:max_questions]):
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

        # Read the knowledge cache from the cache file
        cache_t1 = time()
        # if args.kvcache == "file":
        #     knowledge_cache = read_kv_cache(kvcache_path)

        # Not a good idea to use this method, as it will consume a lot of memory
        # if args.kvcache == "variable":
        #     knowledge_cache = documents_cache
        cache_t2 = time()

        # Generate Response for the question
        knowledges = '\n\n\n'.join(text_list)

        if args.usePrompt:
            prompt = f"""
    <|begin_of_text|>
    <|start_header_id|>system<|end_header_id|>
    You are an assistant for giving short answers based on given context.<|eot_id|>
    <|start_header_id|>user<|end_header_id|>
    Context information is bellow.
    ------------------------------------------------
    {knowledges}
    ------------------------------------------------
    {answer_instruction}
    Question:
    {question}<|eot_id|>
    <|start_header_id|>assistant<|end_header_id|>
    """
            generate_t1 = time()
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
            output = generate(model, input_ids, DynamicCache()) 
            generated_text = tokenizer.decode(output[0], skip_special_tokens=True, temperature=None)
            generate_t2 = time()
        else:
            prompt = f"""
    {question}<|eot_id|>
    <|start_header_id|>assistant<|end_header_id|>
    """
            generate_t1 = time()
            clean_up(knowledge_cache, kv_len)
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
            output = generate(model, input_ids, knowledge_cache)
            generated_text = tokenizer.decode(output[0], skip_special_tokens=True, temperature=None)
            generate_t2 = time()

        # print("D: ", knowledges)
        print("Q: ", question)
        print("A: ", generated_text)
 
        # Evaluate bert-score similarity
        similarity = get_bert_similarity(generated_text, ground_truth)

        print(f"[{id}]: Semantic Similarity: {round(similarity, 5)},",
              f"cache time: {cache_t2 - cache_t1},",
              f"generate time: {generate_t2 - generate_t1}")
        with open(args.output, "a") as f:
            f.write(f"[{id}]: Semantic Similarity: {round(similarity, 5)},\t cache time: {cache_t2 - cache_t1},\t generate time: {generate_t2 - generate_t1}\n")

        results["prompts"].append(question)
        results["responses"].append(generated_text)
        results["cache_time"].append(cache_t2 - cache_t1)
        results["generate_time"].append(generate_t2 - generate_t1)
        results["similarity"].append(similarity)

        with open(args.output, "a") as f:
            f.write(f"[{id}]: [Cumulative]: "
                    + f"Semantic Similarity: {round(sum(results['similarity']) / (len(results['similarity'])) , 5)},"
                    + f"\t cache time: {sum(results['cache_time']) / (len(results['cache_time'])) },"
                    + f"\t generate time: {sum(results['generate_time']) / (len(results['generate_time'])) }\n")

    avg_similarity = sum(results["similarity"]) / len(results["similarity"])
    avg_cache_time = sum(results["cache_time"]) / len(results["cache_time"])
    avg_generate_time = sum(results["generate_time"]) / len(results["generate_time"])
    print()
    print(f"Prepare time: {prepare_time}")
    print(f"Average Semantic Similarity: {avg_similarity}")
    print(f"cache time: {avg_cache_time},\t generate time: {avg_generate_time}")
    print()
    with open(args.output, "a") as f:
        f.write("\n")
        f.write(f"Result for {args.output}\n")
        f.write(f"Prepare time: {prepare_time}\n")
        f.write(f"Average Semantic Similarity: {avg_similarity}\n")
        f.write(f"cache time: {avg_cache_time},\t generate time: {avg_generate_time}\n")


# Define quantization configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,              # Load model in 4-bit precision
    bnb_4bit_quant_type="nf4",      # Normalize float 4 quantization
    bnb_4bit_compute_dtype=torch.float16,  # Compute dtype for 4-bit base matrices
    bnb_4bit_use_double_quant=True  # Use nested quantization
)


def load_quantized_model(model_name, hf_token=None):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=hf_token
    )

    # Load model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",          # Automatically choose best device
        trust_remote_code=True,     # Required for some models
        token=hf_token
    )

    return tokenizer, model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAG test with specified parameters.")
    # parser.add_argument('--method', choices=['rag', 'kvcache'], required=True, help='Method to use (rag or kvcache)')
    # parser.add_argument('--kvcache', choices=['file', 'variable'], required=True, help='Method to use (from_file or from_var)')
    parser.add_argument('--modelname', required=False, default="meta-llama/Llama-3.2-1B-Instruct", type=str, help='Model name to use')
    parser.add_argument('--quantized', required=False, default=False, type=bool, help='Quantized model')
    parser.add_argument('--kvcache', choices=['file'], required=True, help='Method to use (from_file or from_var)')
    parser.add_argument('--similarity', choices=['bertscore'], required=True, help='Similarity metric to use (bertscore)')
    parser.add_argument('--output', required=True, type=str, help='Output file to save the results')
    parser.add_argument('--maxQuestion', required=False, default=None, type=int, help='Maximum number of questions to test')
    parser.add_argument('--maxKnowledge', required=False, default=None, type=int, help='Maximum number of knowledge items to use')
    parser.add_argument('--maxParagraph', required=False, default=None, type=int, help='Maximum number of paragraph to use')
    parser.add_argument('--usePrompt', default=False, action="store_true", help='Do not use cache')
    parser.add_argument('--dataset', required=True, help='Dataset to use (kis, kis_sample, squad-dev, squad-train)',
                        choices=['kis', 'kis_sample',
                                 'squad-dev', 'squad-train',
                                 'hotpotqa-dev',  'hotpotqa-train', 'hotpotqa-test'])
    parser.add_argument('--randomSeed', required=False, default=None, type=int, help='Random seed to use')
    # 48 Articles, each article average 40~50 paragraph, each average 5~10 questions

    args = parser.parse_args()

    print("maxKnowledge", args.maxKnowledge, "maxParagraph", args.maxParagraph, "maxQuestion", args.maxQuestion, "randomeSeed", args.randomSeed)

    model_name = args.modelname
    rand_seed = args.randomSeed if args.randomSeed is not None else None

    if args.quantized:
        tokenizer, model = load_quantized_model(model_name=model_name, hf_token=HF_TOKEN)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            token=HF_TOKEN
        )

    def unique_path(path, i=0):
        if os.path.exists(path):
            # path = path.split("_")[:-1] if i != 0 else path
            return unique_path(path + "_" + str(i), i + 1)
        return path

    if os.path.exists(args.output):
        args.output = unique_path(args.output)

    kvcache_test(args)
