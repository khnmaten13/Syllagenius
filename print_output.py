import fitz
import random
import numpy as np
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize the BART pipeline
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

def print_chapters(pdf_path, book_num):
    ########################################################
    doc = fitz.open(pdf_path)

    # Retrieve the table of contents (TOC)
    toc = doc.get_toc(simple=True)  # simple=True provides (level, title, page)

    if not toc:
        print("No Table of Contents found.")
        return

    # Define a set of sections to exclude
    exclude_sections = {'title', 'contents','section', 'introduction', 'preface', 'glossary', 'index', 'appendix'}

    for entry in toc:
        level, title, page = entry
        # Normalize the title to lowercase to make the check case-insensitive
        if not any(excluded.lower() in title.lower() for excluded in exclude_sections) and level == 1:
            indent = '  ' * (level - 1)  # Indent chapters based on their level
            print(f"{book_num} - {indent}{title} - starts on page {page}")

def generate_description(templates, word_dict, user_title):
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