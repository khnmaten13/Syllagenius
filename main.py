from pdf_extractor import find_introduction_page, extract_text_from_introduction_onwards
from preprocessing import preprocess_text
from print_output import print_chapters, generate_description, paraphrase_description
from algorithms import bow_for_comparing, calculate_cosine_similarity, create_bow_from_text
from education_data import adjectives, nouns, verbs, templates

def main(title, main_path, supporting_paths):
    # Check if the main document path is provided
    if not main_path:
        print("No main PDF provided.")
        return False
    
    start_page = find_introduction_page(main_path)

    # Extract and preprocess text from the main document
    main_text = extract_text_from_introduction_onwards(main_path, start_page)
    preprocess_main_text = preprocess_text(main_text)
    
    if not supporting_paths:
        print("No supporting PDFs provided. Continue to main document only.")
        print_chapters(main_path)
        return
    
    # Initialize lists for storing processed texts
    texts = [preprocess_main_text]   
    topictexts = preprocess_main_text
    # Process each supporting document
    for path in supporting_paths:
        sup_start_page =  find_introduction_page(path)
        supporting_text = extract_text_from_introduction_onwards(path, sup_start_page)
        preprocess_supporting_text = preprocess_text(supporting_text)
        texts.append(preprocess_supporting_text)
        topictexts += preprocess_supporting_text


    # Create BoW for all texts including the main document
    _, _, bow = create_bow_from_text(topictexts)
    bow_matrix = bow_for_comparing(texts)
    similarity_matrix = calculate_cosine_similarity(bow_matrix)

    word_dict = {
    'adjectives': adjectives,
    'nouns': nouns,
    'verbs': verbs,
    'topics': bow
    }

    # Handle different numbers of supporting documents
    if len(supporting_paths) == 1:
        similarity_percentage = similarity_matrix[0, 1] * 100
        print(f"Comparison of Main Document with Supporting Document: {similarity_percentage}% similar.")

        # Check if either document is significantly dissimilar
        if similarity_percentage < 50.0:
            response = input("Supporting document is not significantly similar. Do you still want to continue? Y/N ")
            if response.lower() == 'y':
                print("Continuing...")
            else:
                print("Operation cancelled.")
                return
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
    
    # Generate course description
    generated_description = generate_description(templates, word_dict, title)

    # Get paraphrased description
    paraphrased_description = paraphrase_description(generated_description)
    print("\nCourse Description:", paraphrased_description)

    # Print chapters for the main document and each supporting document
    print_chapters(main_path, 'Main Document')
    for idx, path in enumerate(supporting_paths, start=1):
        print_chapters(path, f'Supporting Document {idx}')


       
if __name__ == "__main__":
    user_title = "Data analytics"
    mainPdfPath = 'Anil Maheshwari - Data analytics-McGraw-Hill Education (2017).pdf'
    supPdfPath1 = 'Data Science Algorithms in a Week.pdf'
    supPdfPath2 = None
    supPdfPath = [supPdfPath1, supPdfPath2]
    checked_path = [i for i in supPdfPath if i is not None]
    supPdfPath = checked_path
    main(user_title, mainPdfPath, supPdfPath)
