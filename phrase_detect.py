from sklearn.feature_extraction.text import CountVectorizer
import re

def clean_text(text):
    """ Lower text and remove punctuation for initial cleaning """
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

# Sample data
documents = [
    "Data mining, and data science are closely related fields.",
    "Data mining! involves discovering patterns in large data sets.",
    "Machine learning is a fascinating field that uses data mining techniques.",
    "Data science is evolving as data mining advances.",
    "Studies in data mining have increased with data science applications.",
    "Data is valuable in many fields."
]

# Clean documents
cleaned_documents = [clean_text(doc) for doc in documents]

# Initialize CountVectorizer with n-gram range to include bi-grams and unigrams
vectorizer = CountVectorizer(ngram_range=(1, 2))
X = vectorizer.fit_transform(cleaned_documents)
feature_names = vectorizer.get_feature_names_out()
frequencies = X.toarray().sum(axis=0)
bi_gram_frequencies = dict(zip(feature_names, frequencies))

# Find and print the most frequent bi-gram
most_frequent_bi_gram = max(bi_gram_frequencies, key=bi_gram_frequencies.get)
print("Most Frequent Bi-gram:", most_frequent_bi_gram, bi_gram_frequencies[most_frequent_bi_gram])

# Filter documents to remove the specific bi-gram "data mining"
pattern = r'\bdata mining\b'
filtered_documents = [' '.join(re.sub(pattern, '', doc).split()) for doc in cleaned_documents]

# Re-run CountVectorizer on the filtered documents
vectorizer_filtered = CountVectorizer(ngram_range=(1, 2))
X_filtered = vectorizer_filtered.fit_transform(filtered_documents)
new_feature_names = vectorizer_filtered.get_feature_names_out()
new_frequencies = X_filtered.toarray().sum(axis=0)
new_bi_gram_frequencies = dict(zip(new_feature_names, new_frequencies))

# Print all bi-grams and unigrams sorted by frequency after filtering
print("\nAll terms sorted by frequency after removal of 'data mining':")
sorted_terms = sorted(new_bi_gram_frequencies.items(), key=lambda item: item[1], reverse=True)
for term, frequency in sorted_terms:
    print(f"{term}: {frequency}")
