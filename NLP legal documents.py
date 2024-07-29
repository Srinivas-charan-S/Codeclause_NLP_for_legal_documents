import nltk
nltk.download('punkt')
import spacy

nlp = spacy.load('en_core_web_sm')

def preprocess_text(text):
    doc = nlp(text)
    return ' '.join([token.lemma_ for token in doc if not token.is_stop])
def extract_entities(text):
    doc = nlp(text)
    return [(entity.text, entity.label_) for entity in doc.ents]
from transformers import pipeline

summarizer = pipeline("summarization")

def summarize_text(text):
    return summarizer(text, max_length=130, min_length=30, do_sample=False)[0]['summary_text']
import spacy
from transformers import pipeline

# Load SpaCy model
nlp = spacy.load('en_core_web_sm')

# Load summarizer
summarizer = pipeline("summarization")

def preprocess_text(text):
    doc = nlp(text)
    return ' '.join([token.lemma_ for token in doc if not token.is_stop])

def extract_entities(text):
    doc = nlp(text)
    return [(entity.text, entity.label_) for entity in doc.ents]

def summarize_text(text):
    return summarizer(text, max_length=130, min_length=30, do_sample=False)[0]['summary_text']

# Example document
document = """
This Lease Agreement ("Agreement") is made and entered into on July 1, 2024, by and between ABC Corp ("Lessor") and John Doe ("Lessee"). The Lessor agrees to lease the property located at 123 Main St, Cityville, for a term of 12 months commencing on August 1, 2024, and ending on July 31, 2025. The Lessee agrees to pay a monthly rent of $1,500 on the first day of each month. The Lessor and Lessee further agree to abide by the terms and conditions set forth in this Agreement.
"""

# Preprocess the text
preprocessed_text = preprocess_text(document)

# Extract entities
entities = extract_entities(preprocessed_text)
print("Extracted Entities:", entities)

# Summarize the text
summary = summarize_text(preprocessed_text)
print("Summary:", summary)


    
