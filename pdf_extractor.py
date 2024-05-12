import fitz  # PyMuPDF
import re

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

def extract_text_from_content(pdf_path):
    #######################################################
    doc = fitz.open(pdf_path)
    text = []
    for page in doc:
        if '1' in page.get_text():
            text.append(page.get_text())
    return "\n".join(text)

def extract_text_from_introduction_onwards(pdf_path, start_page):
    doc = fitz.open(pdf_path)
    text = ""

    for page_num in range(start_page, len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text()
        cleaned_text = re.sub(r'\([^)]*\)', '', text)
    return cleaned_text

def extract_chapters(pdf_path):
    """
    Extracts chapters from a PDF eBook using the table of contents.

    Parameters:
    - pdf_path: Path to the PDF file.

    Returns:
    A list of chapters with their titles and starting page numbers.
    """
    # Open the PDF file
    doc = fitz.open(pdf_path)

    # Retrieve the table of contents
    toc = doc.get_toc()

    # Initialize a list to hold chapter details
    chapters = []

    # Iterate through the ToC entries
    for entry in toc:
        level, title, page_number = entry
        # Assuming you want all levels; adjust if you only want top-level chapters
        chapters.append({'title': title, 'page': page_number})

    # Close the document
    doc.close()

    return chapters