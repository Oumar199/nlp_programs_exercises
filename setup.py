from setuptools import setup

setup(name="nlp_project", version="0.0.1", author="Oumar Kane", author_email="oumar.kane@univ-thies.sn", 
      description="Make fast text processing with a text processing pipeline. Contains utils and useful classes with tokenizer with many options, corpus extraction, tokens and n-grams frequency processing and visualization, etc. The processes can be recuperated as a pipeline for a different corpus.",
      requires=['spacy', 'nltk', 'gensim'])
