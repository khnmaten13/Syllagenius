import fitz  # PyMuPDF
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk

# Ensure necessary NLTK resources are downloaded
nltk.download('stopwords')

def extract_text_from_pdf(pdf_path):
    """Extracts and returns text from a given PDF file using fitz."""
    document = fitz.open(pdf_path)
    text = ""
    for page in document:
        text += page.get_text()
    return text

def clean_and_tokenize(text):
    """Cleans and tokenizes text."""
    # Convert text to lowercase
    text = text.lower()
    # Remove non-alphanumeric characters
    text = re.sub(r'\W+', ' ', text)
    # Tokenization
    tokens = text.split()
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    # Stemming
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    return " ".join(tokens)

def compare_documents(text1, text2):
    """Compares two documents using cosine similarity."""
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

# Paths to the PDF files
pdf_path1 = 'Anil Maheshwari - Data analytics-McGraw-Hill Education (2017).pdf'
pdf_path2 = 'Data Science Algorithms in a Week.pdf'

# Extract and process text from PDFs
text1 = clean_and_tokenize(extract_text_from_pdf(pdf_path1))
text2 = clean_and_tokenize(extract_text_from_pdf(pdf_path2))

# Compare the documents and determine similarity
similarity_score = compare_documents(text1, text2)
threshold = 0.5  # Define a threshold for similarity
if similarity_score > threshold:
    print(f"The documents are similar with a similarity score of: {100 * similarity_score:.2f}%")
else:
    print(f"The documents are not similar with a similarity score of: {100 * similarity_score:.2f}%")
