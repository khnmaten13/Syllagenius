import fitz  # PyMuPDF for PDF processing
import nltk
import re
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, MWETokenizer
import random
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Import categories and templates from a separate file (assuming they are defined there)
from education_data import categories, learning_outcomes_templates, exclude_sections

def create_bow_from_chapters(chapter_texts):
    """Generate Bag of Words model from chapter texts, including bi-grams."""
    # Combine all chapter texts into a list
    texts = list(chapter_texts.values())
    
    # Initialize TfidfVectorizer to include uni-grams and bi-grams
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    
    # Transform texts into TF-IDF matrix
    X = vectorizer.fit_transform(texts)
    
    # Extract feature names (words and bi-grams)
    feature_names = vectorizer.get_feature_names_out()
    
    # Sum tf-idf scores for each feature across all documents to find the most frequent terms
    summed_tfidf = X.sum(axis=0)
    
    # Get indices of top 3 features with highest summed TF-IDF scores
    top_indices = np.argsort(summed_tfidf).A1[-4:]  # Use .A1 to flatten the matrix to 1D
    
    # Extract the names of the top 3 features (bi-grams and/or uni-grams)
    top_features = [feature_names[i] for i in top_indices]
    
    # Print top features for verification
    print("Top 3 Features (uni-grams or bi-grams) based on summed TF-IDF scores:", top_features)
    
    return vectorizer, X, top_features

def extract_chapter_text(pdf_path, excluded_sections):
    # Open the PDF document
    doc = fitz.open(pdf_path)
    toc = doc.get_toc(simple=True)

    # Determine top-most level with excluded sections present
    top_most_level = None
    for entry in toc:
        if any(ex.lower() in entry[1].lower() for ex in excluded_sections):
            top_most_level = entry[0]
            break

    # If no excluded sections found at any level, start from the general top level
    if top_most_level is None:
        top_most_level = min(entry[0] for entry in toc)

    # Filter TOC to include only entries at the determined top-most level, excluding sections
    filtered_toc = [entry for entry in toc if entry[0] == top_most_level and not any(
        ex.lower() in entry[1].lower() for ex in excluded_sections)]

    chapter_texts = {}
    for entry in filtered_toc:
        level, title, page_start = entry
        try:
            # Find the next page start to determine the current page end
            page_end = [e[2] for e in filtered_toc if e[2] > page_start][0] - 1
        except IndexError:
            # If no next page, assume end of document
            page_end = doc.page_count - 1

        # Extract text from the determined page range
        text = ""
        for page_num in range(page_start - 1, page_end):
            page = doc.load_page(page_num)
            text += page.get_text()

        # Clean and store the chapter text
        chapter_texts[title] = clean_text(text)

    doc.close()
    return chapter_texts

def clean_text(text):
    text = text.lower()
    tokens = word_tokenize(text)
    words = [word for word in tokens if word.isalpha()]
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)

def generate_course_outcomes(chapter_texts, vectorizer):
    outcomes = []
    for title, text in chapter_texts.items():
        # Get BoW representation for the cleaned text
        bow = vectorizer.transform([text])
        feature_array = vectorizer.get_feature_names_out()
        # Sort terms by frequency and select top keywords
        tfidf_sorting = bow.toarray().flatten().argsort()[::-1]
        top_keywords = [feature_array[i] for i in tfidf_sorting[:5]]
        # Generate three random outcomes using these terms
        ri = random.randint(2, 3)
        for _ in range(ri):
            template = random.choice(learning_outcomes_templates)
            category_name = random.choice(list(categories.keys()))
            action = random.choice(categories[category_name])
            keyword = random.choice(top_keywords)
            outcome = template.format(action, keyword)
            outcomes.append((title, outcome))
    return outcomes
# Path to your PDF
pdf_path = 'Data Science Algorithms in a Week.pdf'
# Extract texts from each chapter
chapter_texts = extract_chapter_text(pdf_path, exclude_sections)
# Create BoW from the chapter texts and identify top bi-grams
vectorizer, X, top_features = create_bow_from_chapters(chapter_texts)
# Generate course outcomes based on the BoW of each chapter
course_outcomes = generate_course_outcomes(chapter_texts, vectorizer)
# Print each outcome
for index, (chapter_title, outcome) in enumerate(course_outcomes, start = 1):
    print(f"{index} | {outcome}")