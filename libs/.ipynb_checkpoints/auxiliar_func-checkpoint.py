import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from pinecone import Pinecone, ServerlessSpec
import os
import numpy as np
import pandas as pd
import tqdm

#define functions for product search using Tf-IDF
def calculate_tfidf(dataframe):
    """
    Calculate the TF-IDF for combined product name and description.

    Parameters:
    dataframe (pd.DataFrame): DataFrame with product_id, and other product information.

    Returns:
    TfidfVectorizer, csr_matrix: TF-IDF vectorizer and TF-IDF matrix.
    """
    # Combine product name and description to vectorize
    # NOTE: Please feel free to use any combination of columns available, some columns may contain NULL values
    combined_text = dataframe['product_name'] + ' ' + dataframe['product_description']
    vectorizer = TfidfVectorizer()
    # convert combined_text to list of unicode strings
    tfidf_matrix = vectorizer.fit_transform(combined_text.values.astype('U'))
    return vectorizer, tfidf_matrix

def get_top_products(vectorizer, tfidf_matrix, query, top_n=10):
    """
    Get top N products for a given query based on TF-IDF similarity.

    Parameters:
    vectorizer (TfidfVectorizer): Trained TF-IDF vectorizer.
    tfidf_matrix (csr_matrix): TF-IDF matrix for the products.
    query (str): Search query.
    top_n (int): Number of top products to return.

    Returns:
    list: List of top N product IDs.
    """
    query_vector = vectorizer.transform([query])
    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    top_product_indices = cosine_similarities.argsort()[-top_n:][::-1]
    return top_product_indices

#define functions for evaluating retrieval performance
def map_at_k(true_ids, predicted_ids, k=10):
    """
    Calculate the Mean Average Precision at K (MAP@K).

    Parameters:
    true_ids (list): List of relevant product IDs.
    predicted_ids (list): List of predicted product IDs.
    k (int): Number of top elements to consider.
             NOTE: IF you wish to change top k, please provide a justification for choosing the new value

    Returns:
    float: MAP@K score.
    """
    #if either list is empty, return 0
    if not len(true_ids) or not len(predicted_ids):
        return 0.0

    score = 0.0
    num_hits = 0.0

    for i, p_id in enumerate(predicted_ids[:k]):
        if p_id in true_ids and p_id not in predicted_ids[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    return score / min(len(true_ids), k)

def create_product_embeddings(df, model, batch_size=256):
    """Encodes product descriptions into BERT embeddings (batched for speed)."""

    df["combined_text"] = (
        df["product_name"].fillna("") + " " +
        df["product_description"].fillna("") + " " +
        df["category hierarchy"].fillna("")
    )

    embeddings = []
    for i in tqdm(range(0, len(df), batch_size)):
        batch = df["combined_text"].iloc[i:i+batch_size].tolist()
        batch_embeddings = model.encode(batch, convert_to_numpy=True)
        embeddings.append(batch_embeddings)

    return np.vstack(embeddings)

def get_top_10(query, model,  product_df, top_n=10):
    """
    Get top N products for a given query based on embedding similarity.

    Parameters:
    query (str): Search query.
    product_df (DataFrame): DataFrame containing product names and embeddings.
    model (SentenceTransformer): Embedding model to convert query to vector.
    top_n (int): Number of top products to return.

    Returns:
    list: List of top N product names.
    """
    # Convert query into an embedding
    query_vector = model.encode([query])
    query_vector = np.array(query_vector).reshape(1, -1)


    product_embeddings = np.array(product_df["embeddings"].tolist())
    cosine_similarities = cosine_similarity(query_vector, product_embeddings).flatten()

    # Get top N product indices
    top_product_indices = cosine_similarities.argsort()[-top_n:][::-1]

    # Get top product names
    top_products = product_df.iloc[top_product_indices]["product_name"].tolist()

    return top_product_indices

#implementing a function to retrieve exact match product IDs for a query_id
def get_exact_matches_for_query(query_id,grouped_label_df):
    query_group = grouped_label_df.get_group(query_id)
    exact_matches = query_group.loc[query_group['label'] == 'Exact'] ['product_id'].values
    return exact_matches
