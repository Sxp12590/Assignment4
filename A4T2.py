import spacy

# Load the pre-trained English model
nlp = spacy.load('en_core_web_sm')

def extract_named_entities(sentence):
    # Process the sentence with spaCy
    doc = nlp(sentence)
    
    # Iterate through the recognized entities
    for ent in doc.ents:
        print(f"Entity Text: {ent.text}")
        print(f"Entity Label: {ent.label_}")
        print(f"Start Character: {ent.start_char}")
        print(f"End Character: {ent.end_char}")
        print()

# Input sentence
sentence = "Barack Obama served as the 44th President of the United States and won the Nobel Peace Prize in 2009."

# Extract named entities
extract_named_entities(sentence)
