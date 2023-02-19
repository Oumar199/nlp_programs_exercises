import pandas as pd
import pandarallel as prl
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
from typing import Union, List, Tuple, Callable
from tqdm import tqdm
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer, word_tokenize, wordpunct_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
from math import log2
import spacy
import re


plt.style.use('ggplot')