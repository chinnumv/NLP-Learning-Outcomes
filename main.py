import spacy
import gensim
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from nltk.sentiment import SentimentIntensityAnalyzer

# Load pre-trained spaCy model
nlp = spacy.load("en_core_web_sm")

# Sample unstructured text data (replace with your actual data)
course_descriptions = ["This course covers basic Python programming concepts such as loops, conditionals, and functions.",
                      "Students will learn about machine learning algorithms including linear regression and decision trees.",
                      "User feedback: The course content was well-structured and easy to follow. However, more practice exercises would be helpful."]

# Entity recognition
for text in course_descriptions:
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    print("Entities:", entities)

# Sentiment analysis
sid = SentimentIntensityAnalyzer()
for text in course_descriptions:
    sentiment_score = sid.polarity_scores(text)
    print("Sentiment Score:", sentiment_score)

# Topic modeling
# Tokenize and preprocess text
tokenized_texts = [[token.text for token in nlp(text.lower()) if not token.is_stop and not token.is_punct] for text in course_descriptions]

# Create dictionary and corpus
dictionary = Dictionary(tokenized_texts)
corpus = [dictionary.doc2bow(tokens) for tokens in tokenized_texts]

# Train LDA model
lda_model = LdaModel(corpus, id2word=dictionary, num_topics=3, passes=10)

# Print topics
for idx, topic in lda_model.print_topics():
    print("Topic {}: {}".format(idx, topic))