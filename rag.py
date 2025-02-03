import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from llama_index.core import VectorStoreIndex, Document
from transformers.cache_utils import DynamicCache
import cag.dataset as cagds
import cag.similarity as cagsim
import argparse
import os
from transformers import BitsAndBytesConfig
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

from dotenv import load_dotenv
load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN not found")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
JINA_API_KEY = os.getenv("JINA_API_KEY")


"""Hugging Face Llama model"""

global model_name, model, tokenizer
global rand_seed

# Allowlist the DynamicCache class
torch.serialization.add_safe_globals([DynamicCache])
torch.serialization.add_safe_globals([set])


from time import time
from llama_index.core import Settings

def getOpenAIRetriever(documents: list[Document], similarity_top_k: int = 1):
    """OpenAI RAG model"""
    import openai
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not found")
    openai.api_key = OPENAI_API_KEY
    # from llama_index.llms.openai import OpenAI
    # Settings.llm = OpenAI(model="gpt-3.5-turbo")
    
    from llama_index.embeddings.openai import OpenAIEmbedding
    # Set the embed_model in llama_index
    Settings.embed_model = OpenAIEmbedding(model_name="text-embedding-3-small", api_key=OPENAI_API_KEY, title="openai-embedding")
    # model_name: "text-embedding-3-small", "text-embedding-3-large"
    
    # Create the OpenAI retriever
    t1 = time()
    index = VectorStoreIndex.from_documents(documents)
    OpenAI_retriever = index.as_retriever(similarity_top_k=similarity_top_k)
    t2 = time()
    logger.info(f"OpenAI retriever prepared in {t2 - t1:.2f} seconds.")
    return OpenAI_retriever, t2 - t1
    

def getGeminiRetriever(documents: list[Document], similarity_top_k: int = 1):
    """Gemini Embedding RAG model"""
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY not found")
    from llama_index.embeddings.gemini import GeminiEmbedding
    model_name = "models/embedding-001"
    # Set the embed_model in llama_index
    Settings.embed_model = GeminiEmbedding( model_name=model_name, api_key=GOOGLE_API_KEY, title="gemini-embedding")
    
    # Create the Gemini retriever
    t1 = time()
    index = VectorStoreIndex.from_documents(documents)
    Gemini_retriever = index.as_retriever(similarity_top_k=similarity_top_k)
    t2 = time()
    logger.info(f"Gemini retriever prepared in {t2 - t1:.2f} seconds.")
    return Gemini_retriever, t2 - t1
    
def getBM25Retriever(documents: list[Document], similarity_top_k: int = 1):
    from llama_index.core.node_parser import SentenceSplitter  
    from llama_index.retrievers.bm25 import BM25Retriever
    import Stemmer

    splitter = SentenceSplitter(chunk_size=512)
    
    t1 = time()
    nodes = splitter.get_nodes_from_documents(documents)
    # We can pass in the index, docstore, or list of nodes to create the retriever
    bm25_retriever = BM25Retriever.from_defaults(
        nodes=nodes,
        similarity_top_k=similarity_top_k,
        stemmer=Stemmer.Stemmer("english"),
        language="english",
    )
    t2 = time()
    bm25_retriever.persist("./bm25_retriever")

    return bm25_retriever, t2 - t1

def getJinaRetriever(documents: list[Document], similarity_top_k: int = 1):
    """Jina Embedding model"""
    if not JINA_API_KEY:
        raise ValueError("JINA_API_KEY not found")
    try:
        from llama_index.embeddings.jinaai import JinaEmbedding
        model_name = "jina-embeddings-v3"
        Settings.embed_model = JinaEmbedding(
            api_key=JINA_API_KEY,
            model=model_name,
            task="retrieval.passage",
        )

        # Create the Jina retriever
        t1 = time()
        index = VectorStoreIndex.from_documents(documents)
        Jina_retriever = index.as_retriever(similarity_top_k=similarity_top_k)
        t2 = time()
        logger.info(f"Jina retriever prepared in {t2 - t1:.2f} seconds.")
        return Jina_retriever, t2 - t1
    except ImportError:
        logger.error("Failed to import JinaEmbedding. Please install jinaai package.")
        raise
    except Exception as e:
        logger.error(f"Error creating Jina retriever: {str(e)}")
        raise

    
def rag_test(args: argparse.Namespace):
    answer_instruction = "Answer the question with a super short answer."
    text_list, dataset = cagds.get(args.dataset, max_knowledge=args.maxKnowledge, max_paragraph=args.maxParagraph, max_questions=args.maxQuestion)
        
    if answer_instruction != None:
        answer_instruction = "Answer the question with a super short answer."
    
    # document indexing for the rag retriever
    documents = [Document(text=t) for t in text_list]
    
    retriever = None
    prepare_time = 0.0
    if args.index == "gemini":
        retriever, prepare_time = getGeminiRetriever(documents, similarity_top_k=args.topk)
    if args.index == "openai":
        retriever, prepare_time = getOpenAIRetriever(documents, similarity_top_k=args.topk)
        logger.info(f"Testing {args.index.upper()} retriever with {len(documents)} documents.")
    if args.index == "bm25":
        retriever, prepare_time = getBM25Retriever(documents, similarity_top_k=args.topk)
    if args.index == "jina":
        retriever, prepare_time = getJinaRetriever(documents, similarity_top_k=args.topk)
        logger.info(f"Testing {args.index.upper()} retriever with {len(documents)} documents.")

    if retriever is None:
        raise ValueError("No retriever, `--index` not set")
        
    print(f"Retriever {args.index.upper()} prepared in {prepare_time} seconds")
    with open(args.output, "a") as f:
        f.write(f"Retriever {args.index.upper()} prepared in {prepare_time} seconds\n")
    
    results = {
        "retrieve_time": [],
        "generate_time": [],
        "similarity": [],
        "prompts": [],
        "responses": []
    }
    
    dataset = list(dataset) # Convert the dataset to a list
    
    max_questions = min(len(dataset), args.maxQuestion) if args.maxQuestion != None else len(dataset)
    
    for id, (question, ground_truth) in enumerate(dataset[:max_questions]):    # Retrieve the knowledge from the vector database
        retrieve_t1 = time()
        nodes = retriever.retrieve(question)
        retrieve_t2 = time()
        
        knowledge = "\n---------------------\n".join([node.text for node in nodes])
        # short_knowledge = knowledge[:knowledge.find("**Step 4")]
        
        prompt = f"""
    <|begin_of_text|>
    <|start_header_id|>system<|end_header_id|>
    You are an assistant for giving short answers based on given context.<|eot_id|>
    <|start_header_id|>user<|end_header_id|>
    Context information is below.
    ------------------------------------------------
    {knowledge}
    ------------------------------------------------
    {answer_instruction}
    Question:
    {question}
    <|eot_id|>
    <|start_header_id|>assistant<|end_header_id|>
    """

        # Generate Response for the question
        generate_t1 = time() 
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
        output = model.generate(
            input_ids,
            max_new_tokens=300,  # Set the maximum length of the generated text
            do_sample=False,  # Ensures greedy decoding,
            temperature=None
        )
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        generate_t2 = time() 

        generated_text = generated_text[generated_text.find(question) + len(question):]
        generated_text = generated_text[generated_text.find('assistant') + len('assistant'):].lstrip()
        
        # print("R: ", knowledge)
        print("Q: ", question)
        print("A: ", generated_text)
        
        # Evaluate bert-score similarity
        similarity = cagsim.bert(generated_text, ground_truth)
        
        print(f"[{id}]: Semantic Similarity: {round(similarity, 5)},\t",
            f"retrieve time: {retrieve_t2 - retrieve_t1},\t",
            f"generate time: {generate_t2 - generate_t1}"
            )
        with open(args.output, "a") as f:
            f.write(f"[{id}]: Semantic Similarity: {round(similarity, 5)},\t retrieve time: {retrieve_t2 - retrieve_t1},\t generate time: {generate_t2 - generate_t1}\n")
            
        results["prompts"].append(prompt)
        results["responses"].append(generated_text)
        results["retrieve_time"].append(retrieve_t2 - retrieve_t1)
        results["generate_time"].append(generate_t2 - generate_t1)
        results["similarity"].append(similarity)
        
        with open(args.output, "a") as f:
            f.write(f"[{id}]: [Cumulative]: " 
                    + f"Semantic Similarity: {round(sum(results['similarity']) / (len(results['similarity'])) , 5)}," 
                    + f"\t retrieve time: {sum(results['retrieve_time']) / (len(results['retrieve_time'])) },"
                    + f"\t generate time: {sum(results['generate_time']) / (len(results['generate_time'])) }\n")
        
        
    avg_similarity = sum(results["similarity"]) / len(results["similarity"])
    avg_retrieve_time = sum(results["retrieve_time"]) / len(results["retrieve_time"])
    avg_generate_time = sum(results["generate_time"]) / len(results["generate_time"])
    print()
    print(f"Prepare time: {prepare_time}")
    print(f"Average Semantic Similarity: {avg_similarity}")
    print(f"retrieve time: {avg_retrieve_time},\t generate time: {avg_generate_time}")
    print()
    with open(args.output, "a") as f:
        f.write("\n")
        f.write(f"Result for {args.output}\n")
        f.write(f"Prepare time: {prepare_time}\n")
        f.write(f"Average Semantic Similarity: {avg_similarity}\n")
        f.write(f"retrieve time: {avg_retrieve_time},\t generate time: {avg_generate_time}\n")


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
    parser.add_argument('--modelname', required=False, default="meta-llama/Llama-3.2-1B-Instruct", type=str, help='Model name to use')
    parser.add_argument('--quantized', required=False, default=False, type=bool, help='Quantized model')
    parser.add_argument('--index', choices=['gemini', 'openai', 'bm25', 'jina'], required=True, help='Index to use (gemini, openai, bm25, jina)')
    parser.add_argument('--similarity', choices=['bertscore'], required=True, help='Similarity metric to use (bertscore)')
    parser.add_argument('--output', required=True, type=str, help='Output file to save the results')
    parser.add_argument('--maxQuestion', required=False, default=None ,type=int, help='Maximum number of questions to test')
    parser.add_argument('--maxKnowledge', required=False, default=None ,type=int, help='Maximum number of knowledge items to use')
    parser.add_argument('--maxParagraph', required=False, default=None ,type=int, help='Maximum number of paragraph to use')
    parser.add_argument('--topk', required=False, default=1, type=int, help='Top K retrievals to use')
    parser.add_argument('--dataset', required=True, help='Dataset to use (kis, squad or hotpotqa)', 
                        choices=['kis', 'kis_sample', 
                                'squad-dev', 'squad-train', 
                                'hotpotqa-dev',  'hotpotqa-train', 'hotpotqa-test'])
    parser.add_argument('--randomSeed', required=False, default=None, type=int, help='Random seed to use')
    
    # 48 Articles, each article average 40~50 paragraph, each average 5~10 questions
    
    args = parser.parse_args()
    
    print("maxKnowledge", args.maxKnowledge, "maxParagraph", args.maxParagraph, "maxQuestion", args.maxQuestion, "randomSeed", args.randomSeed)
    
    model_name = args.modelname
    rand_seed = args.randomSeed if args.randomSeed != None else None
    
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
            return unique_path(path + "_" + str(i), i + 1)
        return path
    
    if os.path.exists(args.output):
        args.output = unique_path(args.output)
        
    rag_test(args)
