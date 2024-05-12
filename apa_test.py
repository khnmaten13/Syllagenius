import fitz  # PyMuPDF
import re

def extract_pdf_metadata(pdf_path):
    with fitz.open(pdf_path) as doc:
        metadata = doc.metadata
    return metadata

def extract_text_from_first_pages(pdf_path, num_pages=3):
    """
    Extracts text from the first few pages of a PDF.
    
    Parameters:
    pdf_path (str): Path to the PDF file.
    num_pages (int): Number of pages to extract text from, starting from the beginning.
    
    Returns:
    str: Concatenated text from the specified number of pages.
    """
    text = ""
    with fitz.open(pdf_path) as doc:
        for page_num in range(min(num_pages, len(doc))):
            page = doc.load_page(page_num)
            text += page.get_text()
    return text

def find_publisher_in_text(text):
    """
    Attempts to find publisher information in the extracted text using simple keyword search.
    
    Parameters:
    text (str): Text extracted from a PDF.
    
    Returns:
    str: Publisher name if found, otherwise "Unknown Publisher".
    """
    # Common patterns or lines that might indicate publisher information
    patterns = [
        r"Published by (?P<publisher>[\w\s,]+)",
        r"Publisher: (?P<publisher>[\w\s,]+)"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group("publisher").strip()
    
    return "Unknown Publisher"

def parse_authors(authors_str):
    """
    Parse author names from the provided string and format them according to APA standards.
    
    Parameters:
    authors_str (str): A string containing author names, possibly including credentials.
    
    Returns:
    str: Formatted author names in "Last, F. M." format, joined by "&" for APA citation.
    """
    # Split authors by common separators (e.g., ",", "and")
    authors = re.split(',|and', authors_str)
    formatted_authors = []
    
    for author in authors:
        # Remove titles or credentials (e.g., "PhD")
        author = re.sub(r'PhD|Dr\.?', '', author).strip()
        # Split name into parts
        name_parts = author.split()
        # Format name parts: Last, F. M.
        if len(name_parts) > 1:
            last_name = name_parts[-1]
            initials = ' '.join([name[0] + '.' for name in name_parts[:-1]])
            formatted_name = f"{last_name}, {initials}"
        else:
            formatted_name = author  # Fallback for single-name authors
        formatted_authors.append(formatted_name)
    
    # Join formatted author names with "&"
    return ' & '.join(formatted_authors)

def format_apa_reference(metadata):
    author_str = metadata.get('author', 'Unknown Author')
    # Call parse_authors to format author names
    authors_formatted = parse_authors(author_str)
    
    title = metadata.get('title', 'Unknown Title').capitalize()
    year = metadata.get('creationDate', 'Unknown Year')
    
    # Attempt to extract the year
    year_match = re.search(r'D:(\d{4})', year)
    year = year_match.group(1) if year_match else "Unknown Year"
    
    # Assuming the publisher is not available in PDF metadata
    text = extract_text_from_first_pages(pdf_path, num_pages=5)
    publisher = find_publisher_in_text(text)
    
    apa_reference = f"{authors_formatted} ({year}). {title}. {publisher}."
    return apa_reference

# Example usage
pdf_path = 'Biology for Dummies.pdf'
metadata = extract_pdf_metadata(pdf_path)
apa_reference = format_apa_reference(metadata)
print(apa_reference)
