from nlp_project import *
from typing import List

# We transform each document to tokens and remove stopwords and 
# words with length less than 3 in one single line
text_transformation_alpha = lambda text, tokenizer, stop_words, min_length: [text_ for\
    text_ in tokenizer.tokenize(text.lower())\
        if text_ not in stop_words and len(text_) > min_length and text_.isalpha()]

text_transformation = lambda text, tokenizer, stop_words, min_length: [text_ for\
    text_ in tokenizer.tokenize(text.lower())\
        if text_ not in stop_words and len(text_) > min_length]

def pre_transformation1(texts, stop_words: list, min_length: int = 3, regexp:str = r"\w+", only_alpha: bool = True):
    """Transforms a list of texts to a list of tokens where each text represents a document

    Args:
        texts (Union[list, pd.Series, tuple]): A list of documents
        stop_words (list): Words to remove from the texts
        min_length (int, optional): The minimal length of a word. Defaults to 3.
        regexp (str, optional): The regular expression to tokenize the documents. Defaults to r"\w+".
        only_alpha (bool, optional): Indicates if we want to take only the alphanumerical words as tokens. Defaults to True.

    Returns:
        list: List of lists of tokens
    """
    
    tokenizer = RegexpTokenizer(regexp)
    
    tokens = []
    for text in tqdm(texts):
        if only_alpha:
            tokens.append(text_transformation_alpha(text, tokenizer, stop_words, min_length))
        else:
            tokens.append(text_transformation(text, tokenizer, stop_words, min_length))
            
    return tokens


def pre_transformation2(texts: Union[list, pd.Series, tuple], stop_words: list, min_length: int = 3, regexp:str = r"\w+", only_alpha: bool = True):
    """Transforms a list of texts to a list of tokens where each text represents a document

    Args:
        texts (Union[list, pd.Series, tuple]): A list of documents
        stop_words (list): Words to remove from the texts
        min_length (int, optional): The minimal length of a word. Defaults to 3.
        regexp (str, optional): The regular expression to tokenize the documents. Defaults to r"\w+".
        only_alpha (bool, optional): Indicates if we want to take only the alphanumerical words as tokens. Defaults to True.

    Returns:
        list: List of lists of tokens
    """
    
    tokenizer = RegexpTokenizer(regexp)
    
    tokens = []
    
    if type(texts) is pd.Series: texts = texts.tolist()
    
    for text in tqdm(texts):
        if only_alpha:
            doc_tokens = text_transformation_alpha(text, tokenizer, stop_words, min_length)
        else:
            doc_tokens = text_transformation(text, tokenizer, stop_words, min_length)
            
        if doc_tokens:
            tokens.append(doc_tokens)
    
    return tokens

def guess_limitations(data_frame: pd.DataFrame, column: str):
    q1 = data_frame[column].quantile(0.25)
    q3 = data_frame[column].quantile(0.75)
    eq = q3 - q1
    limit1 = q1 - 1.5 * eq
    limit2 = q3 + 1.5 * eq
    return limit1, limit2

def get_freqs_from_text(text: Union[list, tuple, pd.Series]):
    
    if type(text) is pd.Series:
        text = text.tolist()
    
    freqs = nltk.FreqDist(" ".join(text).split(" "))
    
    return pd.Series(freqs), freqs

def wordcloud(text: str, figsize: tuple = (8, 8), max_font_size: int = 60, max_words: int = 100, background_color = "white"):
    """Generate a wordcloud from a given text

    Args:
        text (str): The text from which we want to make a wordcloud
        figsize (tuple, optional): The size of the figure. Defaults to (8, 8).
        max_font_size (int, optional): The max font size. Defaults to 60.
        max_words (int, optional): The max number of words on top. Defaults to 100.
        background_color (str, optional): The background color. Defaults to "white".
    """
    
    plt.figure(figsize=figsize)
    
    word_cloud = WordCloud(
        max_font_size=max_font_size,
        max_words=max_words,
        background_color=background_color).generate(text)
    
    plt.imshow(word_cloud)
    
    plt.axis('off')
    
    plt.show()
    
# Create a new wordcloud function
def wordcloud_hue(data_frame: pd.DataFrame, target: str = "target", text: str = "text", figsize: tuple = (8, 8), max_font_size: int = 60, max_words: int = 100, background_color = "white"):
    
    unique_classes = data_frame[target].unique().tolist()
    
    fig, axs = plt.subplots(1, len(unique_classes), figsize = figsize)
    
    axs = axs.flat
    
    for i, class_ in enumerate(unique_classes):
        
        text_ = data_frame[data_frame[target] == class_][text].tolist()
        
        word_cloud = WordCloud(
            max_font_size=max_font_size,
            max_words=max_words,
            background_color=background_color).generate(" ".join(text_))
        
        axs[i].set_title(class_)
        
        axs[i].imshow(word_cloud)
        
        axs[i].axis('off')
        
    plt.show()

# recuperate words appearing in both of the two Series
def words_w_same_classes(freqs_series: List[pd.Series], top: int = 10):
    """Recuperate words appearing in two different classes

    Args:
        freqs_series (List[pd.Series]): A list of words frequencies as pandas Series where words are indexes and each of them represents a target class
        top (int, optional): The number of words on the top. Defaults to 10.

    Returns:
        list: The list of words appearing in both of the two classes
    """
    
    assert len(freqs_series) > 1
    
    f_serie = freqs_series[0].sort_values(ascending=False).head(top)
    
    words = set(f_serie.index)
    commons = set()
    
    for freq_serie in freqs_series[1:]:
        
        f_serie = freq_serie.sort_values(ascending=False).head(top)

        for word in f_serie.index:
            
            if word in words:
                
                commons.add(word)
            
            else:
                
                words.add(word)
   
    return list(commons)