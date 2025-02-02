from sentence_transformers import SentenceTransformer, util

# Use a lightweight sentence-transformer
bert_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def bert(response, ground_truth):
    """
    @param response: the response from LLM
    @param ground_truth: the ground truth of the question
    @return: the cosine similarity
    """
    query_embedding = bert_model.encode(response, convert_to_tensor=True)
    text_embedding = bert_model.encode(ground_truth, convert_to_tensor=True)

    # Compute the cosine similarity between the query and text
    cosine_score = util.pytorch_cos_sim(query_embedding, text_embedding)

    return cosine_score.item()
