import random
from transformers import pipeline
import fitz  # PyMuPDF
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from pdf_extractor import extract_text_from_introduction_onwards

def extract_text_from_pdf(filepath):
    """Extract text from each page of the PDF and return it as a single string."""
    doc = fitz.open(filepath)
    text = " ".join(page.get_text() for page in doc)
    doc.close()
    return text

def create_bow_from_text(text):
    """Generate Bag of Words model from text, including bi-grams."""
    # Initialize TfidfVectorizer to include uni-grams and bi-grams
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')

    # Transform text into TF-IDF matrix
    X = vectorizer.fit_transform([text])

    # Extract feature names (words and bi-grams)
    feature_names = vectorizer.get_feature_names_out()

    # Sum tf-idf scores for each feature to find the most frequent terms
    summed_tfidf = X.sum(axis=0)

    # Get indices of top features with highest summed TF-IDF scores
    top_indices = np.argsort(summed_tfidf).A1[-5:]  # Example: Get top 10 features

    # Extract the names of the top features
    top_features = [feature_names[i] for i in top_indices]

    # Print top features for verification
    print("Top Features (uni-grams or bi-grams) based on summed TF-IDF scores:", top_features)
    
    return vectorizer, X, top_features


# Bag of words categorized by parts of speech or usage
adjectives = ["comprehensive", "advanced", "introductory", "detailed", "practical"]
nouns = ["basics", "concepts", "applications", "principles", "techniques"]
verbs = ["covers", "introduces", "explores", "discusses", "explains"]

user_title = "Data analytics"

# Sentence templates using regex-like syntax
templates = [
    "This {adjective} course {verb} the {noun} of {title}.",
    "Students will learn {noun} and {noun} in {topic}.",
    "The course {verb} key {noun} essential for {topic}.",
    "Explore {noun} through {adjective} {topic} lectures and hands-on projects.",
    "Gain {noun} in {topic} through {adjective} study and real-world {noun}."
]

def generate_description(templates, word_dict):
    description = []
    for template in templates:
        sentence = template.format(
            title=user_title,
            adjective=random.choice(word_dict['adjectives']),
            noun=random.choice(word_dict['nouns']),
            verb=random.choice(word_dict['verbs']),
            topic=random.choice(word_dict['topics'])
        )
        description.append(sentence)
    return " ".join(description)

def find_introduction_page(pdf_path, titles=["summary", "1", "chapter 1"]):
    doc = fitz.open(pdf_path)
    toc = doc.get_toc()
    glossary_index_page_num = None

    for item in toc:
        title = item[1].lower()
        if any(title_word in title for title_word in titles):
            # Try to match the glossary/index based on title; adjust if necessary
            glossary_index_page_num = item[2] - 1  # Considering PyMuPDF page index starts at 0
            break
    
    if glossary_index_page_num is not None:
        # Extra check to ensure the page number is within the document's range
        glossary_index_page_num = min(glossary_index_page_num, len(doc) - 1)
    
    return glossary_index_page_num



# Initialize the paraphrasing pipeline
paraphraser = pipeline("text2text-generation", model="stanford-oval/paraphraser-bart-large")

# Paraphrase each sentence
def paraphrase_description(description):
    sentences = description.split('.')
    paraphrased_sentences = []
    for sentence in sentences:
        if sentence.strip():  # check if the sentence is not just spaces
            paraphrase = paraphraser(sentence.strip() + '.', max_length=60)
            paraphrased_sentences.append(paraphrase[0]['generated_text'])
    return " ".join(paraphrased_sentences)

# Path to your PDF file
pdf_path = 'Anil Maheshwari - Data analytics-McGraw-Hill Education (2017).pdf'

start_page = find_introduction_page(pdf_path)

# Extract text from PDF
pdf_text = extract_text_from_introduction_onwards(pdf_path, start_page)

# Create BoW with n-grams from extracted text
_, _, topics = create_bow_from_text(pdf_text)

word_dict = {
    'adjectives': adjectives,
    'nouns': nouns,
    'verbs': verbs,
    'topics': topics
}

# Generate course description
generated_description = generate_description(templates, word_dict)
# Get paraphrased description
paraphrased_description = paraphrase_description(generated_description)
print("Course Description:", paraphrased_description)
