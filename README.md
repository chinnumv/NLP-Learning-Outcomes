# NLP Analysis Toolkit

## Overview
This Python code utilizes various natural language processing (NLP) techniques to analyze unstructured text data. It includes functionalities for entity recognition, sentiment analysis, and topic modeling.

## Requirements
- Python 3.x
- spaCy library
- gensim library
- NLTK library

## Usage
- Entity Recognition:
Utilizes the spaCy library to recognize entities (e.g., persons, organizations, locations) in the provided text data.
- Sentiment Analysis:
Employs the NLTK library to perform sentiment analysis on the text data. It calculates sentiment scores indicating the positivity, negativity, and neutrality of the text.
- Topic Modeling:
Uses the gensim library to perform topic modeling on the text data. It extracts latent topics from the corpus and assigns topics to documents.

## Example
- Entity Recognition:
Entities recognized in the text data are printed along with their respective labels.
- Sentiment Analysis:
Sentiment scores for each text sample are printed, indicating the positivity, negativity, and neutrality of the text.
- Topic Modeling:
The Latent Dirichlet Allocation (LDA) model is trained on the text data to extract topics. The topics and their corresponding words are printed.