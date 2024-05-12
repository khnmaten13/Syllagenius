import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def bow_for_comparing(texts):
    #######################
    # Converts a list of texts to a Bag of Words model
    vectorizer = CountVectorizer()
    bow_matrix = vectorizer.fit_transform(texts)
    return bow_matrix

def calculate_cosine_similarity(bow_matrix):
    """Calculates and returns the cosine similarity matrix from a BoW matrix."""
    similarity_matrix = cosine_similarity(bow_matrix)
    return similarity_matrix

def create_bow_from_text(texts):
    """Generate Bag of Words model from text, including bi-grams."""
    # Initialize TfidfVectorizer to include uni-grams and bi-grams
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')
    # Ensure that each element in texts is a string
    if any(isinstance(text, list) for text in texts):
        # Join each list of strings into a single string if necessary
        texts = [' '.join(text) if isinstance(text, list) else text for text in texts]
    # Transform text into TF-IDF matrix
    X = vectorizer.fit_transform([texts])

    # Extract feature names (words and bi-grams)
    feature_names = vectorizer.get_feature_names_out()

    # Sum tf-idf scores for each feature to find the most frequent terms
    summed_tfidf = X.sum(axis=0).A1

    # Sort features by summed TF-IDF scores
    sorted_indices = np.argsort(summed_tfidf)[::-1]  # Sort descending

    # Prioritize bi-grams by boosting their index sort value if they contain a space (indicative of a bi-gram)
    bi_gram_boosted_indices = sorted(sorted_indices, key=lambda idx: (' ' in feature_names[idx], summed_tfidf[idx]), reverse=True)

    # Get top features with highest summed TF-IDF scores
    top_features = [feature_names[i] for i in bi_gram_boosted_indices[:10]]  # Example: Get top 10 features

    # Print top features for verification
    print("Top Features (uni-grams or bi-grams) based on summed TF-IDF scores:", top_features)
    
    return vectorizer, X, top_features