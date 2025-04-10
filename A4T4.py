from transformers import pipeline

# Load the sentiment analysis pipeline
sentiment_analyzer = pipeline('sentiment-analysis')

# Input sentence
sentence = "Despite the high price, the performance of the new MacBook is outstanding."

# Analyze the sentiment
result = sentiment_analyzer(sentence)[0]

# Extract and print the sentiment label and confidence score
label = result['label']
confidence_score = result['score']

print(f"Sentiment: {label}")
print(f"Confidence Score: {confidence_score:.4f}")
