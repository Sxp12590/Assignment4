import nltk
nltk.download('punkt')  # For tokenization
nltk.download('stopwords')  # For stopwords

import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.tokens import Doc

# Load the pre-trained English model
nlp = spacy.load('en_core_web_sm')

def preprocess_sentence_spacy(sentence):
    # Process the sentence with spaCy
    doc = nlp(sentence)

    # Step 1: Tokenize the sentence into individual words
    tokens = [token.text for token in doc]
    print("Original Tokens:", tokens)

    # Step 2: Remove stopwords
    tokens_without_stopwords = [token.text for token in doc if not token.is_stop]
    print("Tokens Without Stopwords:", tokens_without_stopwords)

    # Step 3: Apply stemming (spaCy uses lemmatization by default)
    stemmed_words = [token.lemma_ for token in doc if not token.is_stop]
    print("Stemmed Words:", stemmed_words)

# Example usage with the provided sentence
sentence = "NLP techniques are used in virtual assistants like Alexa and Siri."
preprocess_sentence_spacy(sentence)