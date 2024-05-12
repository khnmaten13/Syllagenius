import fitz  # PyMuPDF
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def extract_text_from_content(pdf_path):
    doc = fitz.open(pdf_path)
    text = []
    for page in doc:
        if '1' in page.get_text():
            text.append(page.get_text())
    return "\n".join(text)

def extract_text_from_pdf(pdf_path, start_page, end_page):
    document = fitz.open(pdf_path)
    text = ""
    for page_num in range(start_page, end_page):  # Specify end page for extraction
        page = document.load_page(page_num)
        text += page.get_text()
    return text

# Function to preprocess text
def preprocess_text(text):
    nltk.download('punkt')
    nltk.download('stopwords')
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize

    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text.lower())
    filtered_text = [w for w in word_tokens if not w in stop_words and w.isalnum()]
    return " ".join(filtered_text)

def create_bow(texts):
    """Converts a list of texts to a Bag of Words model."""
    vectorizer = CountVectorizer()
    bow_matrix = vectorizer.fit_transform(texts)
    return bow_matrix

def calculate_cosine_similarity(bow_matrix):
    """Calculates and returns the cosine similarity matrix from a BoW matrix."""
    similarity_matrix = cosine_similarity(bow_matrix)
    return similarity_matrix

#print chapters
def print_chapters(pdf_path, book_num):
    doc = fitz.open(pdf_path)

    # Retrieve the table of contents (TOC)
    toc = doc.get_toc(simple=True)  # simple=True provides (level, title, page)

    if not toc:
        print("No Table of Contents found.")
        return

    # Define a set of sections to exclude
    exclude_sections = {'title', 'contents','section', 'introduction', 'preface', 'glossary', 'index', 'appendix'}

    # Print the chapter information, excluding specified sections
    print(f"Chapters for book {book_num}")
    for entry in toc:
        level, title, page = entry
        # Normalize the title to lowercase to make the check case-insensitive
        if not any(excluded.lower() in title.lower() for excluded in exclude_sections) and level == 1:
            indent = '  ' * (level - 1)  # Indent chapters based on their level
            print(f"{book_num} - {indent}{title} - starts on page {page}")

# Main function to execute the process
def compare_pdf_texts(main_path, supporting_paths):
    # Check if the main document path is provided
    if not main_path:
        print("No main PDF provided.")
        return False
    # Extract and preprocess text from the main document
    main_text = extract_text_from_content(main_path)
    preprocess_main_text = preprocess_text(main_text)
    
    if not supporting_paths:
        print("No supporting PDFs provided. Displaying chapters from the main document only.")
        print_chapters(main_path)
        return True

    # Initialize lists for storing processed texts
    texts = [preprocess_main_text]


    # Process each supporting document
    for path in supporting_paths:
        supporting_text = extract_text_from_content(path)
        preprocess_supporting_text = preprocess_text(supporting_text)
        texts.append(preprocess_supporting_text)

    # Create BoW for all texts including the main document
    bow_matrix = create_bow(texts)
    similarity_matrix = calculate_cosine_similarity(bow_matrix)

    # Handle different numbers of supporting documents
    if len(supporting_paths) == 1:
        similarity_percentage = similarity_matrix[0, 1] * 100
        print(f"Comparison of Main Document with Supporting Document: {similarity_percentage}% similar.")
        input("Please press Enter to continue")
    elif len(supporting_paths) == 2:
        similarity_percentage1 = similarity_matrix[0, 1] * 100
        similarity_percentage2 = similarity_matrix[0, 2] * 100
        print(f"Comparison of Main Document with Supporting Document 1: {similarity_percentage1}% similar.")
        print(f"Comparison of Main Document with Supporting Document 2: {similarity_percentage2}% similar.")

        # Check if either document is significantly dissimilar
        if similarity_percentage1 < 50.0 or similarity_percentage2 < 50.0:
            response = input("At least one supporting document is not significantly similar. Do you still want to continue? Y/N ")
            if response.lower() == 'y':
                print("Continuing...")
            else:
                print("Operation cancelled.")
                return

    # Print chapters for the main document and each supporting document
    print_chapters(main_path, 'Main Document')
    for idx, path in enumerate(supporting_paths, start=1):
        print_chapters(path, f'Supporting Document {idx}')

main_pdf = 'Anil Maheshwari - Data analytics-McGraw-Hill Education (2017).pdf'
sup_pdf = ['Data Science Algorithms in a Week.pdf']
# Main process
compare_pdf_texts(main_pdf, sup_pdf)

