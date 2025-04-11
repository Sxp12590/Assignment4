# Assignment4  #

**Q1: NLP Preprocessing Pipeline **

The Python function does three main NLP preprocessing steps:
Tokenization
Splits the sentence into individual words and punctuation.
Stopword Removal
Removes common, less meaningful words like "are", "in", "the".Keeps only important words.
Stemming
Reduces words to their root form.
1.	What is the difference between stemming and lemmatization? Provide examples with the word “running.”?
Stemming is a crude heuristic process that chops off word endings to reduce them to their base form. It may not result in a real word.
Lemmatization is more sophisticated and reduces words to their dictionary form (lemma), considering the context and part of speech.
Example with "running":
Stemming: "running" → "run" (with PorterStemmer)
Lemmatization: "running" → "run" (but only if it's used as a verb; context matters)
So both can yield "run", but stemming just chops off "ing" mechanically, while lemmatization understands the word grammatically.
2.	Why might removing stop words be useful in some NLP tasks, and when might it actually be harmful?
 Useful:
Removing stopwords can reduce noise in data and improve performance in tasks like:
Document classification
Topic modeling
Keyword extraction
Harmful:
In tasks where word semantics or grammar matters, stopwords are essential:
Sentiment analysis ("not happy" vs. "happy")
Question answering ("Who are you?")
Text summarization (missing conjunctions or negations can distort meaning)


**Q2: Named Entity Recognition with SpaCy**
Load spaCy Model
Loads a small, pre-trained English language model.
Used for tasks like tokenization, POS tagging, and NER.
Define Function: extract_named_entities()
Takes a sentence as input.
Passes it through spaCy's NLP pipeline to analyze linguistic features.
Named Entity Extraction
Iterates over all recognized named entities (doc.ents).
1. How does NER differ from POS tagging in NLP?
Named Entity Recognition (NER):
Identifies and classifies real-world entities in text such as names of people, organizations, locations, dates, etc.
Part-of-Speech (POS) Tagging:
Assigns grammatical roles to each word (like noun, verb, adjective, etc.).
NER focuses on what the word represents in the real world, while POS tagging focuses on how the word functions grammatically in a sentence.
2. Describe two applications that use NER in the real world
a. Financial News Analysis
NER is used to extract company names, stock symbols, dates, and monetary values from articles.
Example: A system reading "Tesla shares rose 5% on Monday" can identify "Tesla" (ORG), "5%" (PERCENT), and "Monday" (DATE) for financial trend analysis.
b. Search Engines
NER helps understand user queries and return relevant results.
Example: A search for "weather in Paris next Monday" detects "Paris" (GPE) and "next Monday" (DATE), enabling location and time-based query handling.



**Q3: Scaled Dot-Product Attention **

The code implements Scaled Dot-Product Attention, a core mechanism in transformer models used for tasks like machine translation and text generation.
It begins by computing the dot product between the query matrix Q and the transpose of the key matrix K, which measures similarity between query and key vectors.
The result is then scaled by the square root of the dimension of the key vectors to stabilize gradients during training.
After scaling, a softmax function is applied to produce normalized attention weights that represent the importance of each key in relation to the query.
These attention weights are then used to compute the final weighted sum of the value vectors V, producing the attention output.
The function returns both the attention weights and the final output. 
The sample test demonstrates this mechanism using simple Q, K, and V matrices, and prints both the attention weight matrix and the resulting output matrix.
1.Why do we divide the attention score by √d in the scaled dot-product attention formula?
In scaled dot-product attention, we divide the raw attention scores (dot products of Q and K) by √d (where d is the dimension of the key vectors) to prevent the values from becoming too large.
Without scaling, when the dimension d is large, the dot products can have high magnitudes, which pushes the softmax function into regions with extremely small gradients. This can make learning unstable or slow. Scaling stabilizes the softmax computation and keeps the gradients in a healthy range for training.
2. How does self-attention help the model understand relationships between words in a sentence?
Self-attention allows the model to weigh the importance of each word in a sentence relative to every other word, including itself. For each word, it computes attention scores with respect to all other words and combines them accordingly.
This means the model can:
Capture contextual dependencies, even for words far apart in the sentence.
Learn things like subject-verb agreement, co-reference, and semantic roles.



**Q4: Sentiment Analysis using HuggingFace Transformers**

The code performs sentiment analysis on a sentence using the Hugging Face transformers library. It loads a pre-trained sentiment analysis pipeline, which typically uses a model like BERT fine-tuned for sentiment classification (e.g., positive or negative).
Load the pipeline:
Automatically downloads and initializes a pre-trained sentiment model.
Analyze sentiment and extract result:
Returns a dictionary with:
'label': e.g., "POSITIVE" or "NEGATIVE"
'score': Confidence level (between 0 and 1)

1. What is the main architectural difference between BERT and GPT? Which uses an encoder and which uses a decoder?
BERT (Bidirectional Encoder Representations from Transformers):
Uses only the Transformer encoder architecture.
It processes the input bidirectionally, meaning it looks at both the left and right context of a word at once.
Designed for understanding tasks like question answering, sentence classification, and named entity recognition.
GPT (Generative Pre-trained Transformer):
Uses only the Transformer decoder architecture.
Processes text left-to-right (causally), focusing on generating the next word in a sequence.
Best suited for generation tasks like text completion, chatbots, and creative writing.

2. Why is using pre-trained models (like BERT or GPT) beneficial for NLP applications instead of training from scratch?
Using pre-trained models offers several key benefits:
Saves Time and Resources:
Training large models like BERT or GPT from scratch requires massive datasets, computing power, and time. Pre-trained models already learned general language patterns.
Better Performance with Less Data:
You can fine-tune these models on your specific task using much smaller datasets, achieving strong performance with minimal training.
Transfer Learning Power:
These models are trained on huge text corpora (like Wikipedia and books), so they already understand grammar, context, and semantics—this knowledge transfers well to a wide variety of NLP tasks.









