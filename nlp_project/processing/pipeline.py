
from nlp_project.processing.utils import *
from typing import List
from nltk.corpus import stopwords

class TextPipeProcessing:
    """The pipeline is composed by (* if obligatory processing):
    - tokenize_text*
    - create_corpus*
    - print_frequency*
    - print_frequency_out_limits
    - print_most_common_words
    - plot_frequency_histogram
    - print_n_time_frequency_words
    - delete_n_time_frequency_words
    - lemmatize_words / stem_words
    - corpus_filter 
    - get_words_w_same_targets
    - remove_words
    - remove_words_w_same_targets
    - recuperate_results*
    - add_results_to_data_frame*
    - plot_wordcloud
    """
    def __init__(self, data_frame, text_column: str, target_column: str = 'target'):
        
        self.data_frame = data_frame
        
        self.text_column = text_column
        
        self.target_column = target_column
        
        self.lemmatizer = None
        
        self.stemmer = None
    
    def tokenize_text(self,
                      language: str = 'english',
                      regex: str = r"\w+",
                      min_word_length: int = 3,
                      only_alpha: bool = True,
                      ):
        
        texts = self.data_frame[self.text_column]
        
        texts = texts.tolist()
        
        self.stopwords = set(stopwords.words(language))
        
        self.tokens = pre_transformation1(texts, self.stopwords, min_word_length, regex, only_alpha)
    
        return self.tokens
        
    def create_corpus(self):
        
        self.corpus = []
        
        for document in tqdm(self.tokens):
            
            self.corpus.extend(document)
        
        self.corpus_text = nltk.text.Text(self.corpus)
        
        print(f"Number of words: {len(self.corpus):->16}")
        print(f"Number of unique words: {len(self.corpus_text.vocab()):->16}")
        
        return self.corpus, self.corpus_text
    
    def print_frequency(self):
        
        self.frequency = pd.DataFrame.from_dict(self.corpus_text.vocab(), 'index')
        
        self.frequency.rename({0: 'frequency'}, inplace=True, axis=1)
        
        self.frequency.reset_index(level=0, inplace=True)
        
        print(self.frequency.head())
    
    def print_frequency_out_limits(self):
        
        px.box(data_frame=self.frequency, x="frequency", hover_data=['index', 'frequency']) 
        
        self.low, self.high = guess_limitations(self.frequency, 'frequency') 
        
        print(f"Low limit: {self.low:->16}")
        print(f"High limit: {self.high:->16}")
    
    def print_most_common_words(self, lower_bound: int = 400, n_words: int = 20):
        
        self.freq_total = nltk.Counter(self.corpus_text.vocab())
        
        self.stopwords_common = list(zip(*self.freq_total.most_common(lower_bound)))[0]
        
        print("Most common words are:")
        print(self.stopwords_common[:20])
    
    def plot_frequency_histogram(self, bottom: int = 8):
        
        f_values = self.frequency['frequency'].sort_values().unique()        
    
        bottom_ = self.frequency[self.frequency['frequency'].isin(f_values[:bottom])]
        
        fig = px.histogram(data_frame = bottom_, x = 'frequency', title=f"Frequency histogram for {bottom} frequency on the bottom")
        
        fig.show()
        
    def print_n_time_frequency_words(self, n_time_freq: Union[int, list] = 1, n_words: int = 100):
        
        n_time_freq = [n_time_freq] if type(n_time_freq) is int else n_time_freq
        
        size = self.frequency[self.frequency['frequency'].isin(n_time_freq)].shape[0]
        
        n_time_frequency = self.frequency[self.frequency['frequency'].isin(n_time_freq)]
        
        print(f"Percentage of words appearing {'/'.join([str(freq) for freq in n_time_freq])} times in the dataset: {size / self.frequency.shape[0]}%")
        
        print(f"Words appearing {'/'.join([str(freq) for freq in n_time_freq])} times:")
        print(n_time_frequency.iloc[:n_words,:])
    
    def delete_n_time_frequency_words(self, n_time_freq: Union[int, list] = 1):
        
        n_time_freq = [n_time_freq] if type(n_time_freq) is int else n_time_freq
        
        n_time_frequency = self.frequency[self.frequency['frequency'].isin(n_time_freq)]
        
        self.new_frequency = self.frequency.loc[~self.frequency['index'].isin(n_time_frequency['index'].to_list()), :]
        
        print("The new frequency data frame is stored in `new_frequency` variable.")
        
        print(f"The number of deleted observations: {n_time_frequency.shape[0]:->16}")
        
    def lemmatize_words(self, lemmatizer = nltk.WordNetLemmatizer()):
        
        self.lemmatizer = lemmatizer
        
        self.new_frequency = self.new_frequency.copy()
        
        self.new_frequency.loc['index'] = self.new_frequency['index'].apply(lambda idx: lemmatizer.lemmatize(idx))
        
        self.new_frequency.dropna(axis=0, inplace=True)
        
        self.new_frequency['frequency'] = self.new_frequency['frequency'].astype('int32')
    
    def stem_words(self, stemmer = nltk.stem.PorterStemmer()):
        
        self.stemmer = stemmer
        
        self.new_frequency = self.new_frequency.copy()
        
        self.new_frequency.loc["index"] = self.new_frequency["index"].apply(lambda idx: stemmer.stem(idx))
        
        self.new_frequency.dropna(axis = 0, inplace=True)
        
        self.new_frequency['frequency'] = self.new_frequency['frequency'].astype('int32')
    
    def corpus_filter(self, corpus: nltk.corpus.words.words()):
        
        if self.lemmatizer:
        
            corpus = [self.lemmatizer.lemmatize(word) for word in corpus]
        
        elif self.stemmer:
            
            corpus = [self.stemmer.stem(word) for word in corpus]
        
        self.new_frequency = self.new_frequency[self.new_frequency['index'].isin(corpus)]
    
    def remove_words(self, words_to_remove: List[str]):
        
        self.new_frequency.drop(index=self.new_frequency[self.new_frequency['index'].isin(words_to_remove)].index, inplace = True)
    
    def get_words_w_same_targets(self, top: int = 30):
        
        target = self.data_frame[self.target_column].unique().tolist()

        freqs_series = []
        
        for class_ in target:
            
            class_df = self.data_frame[self.data_frame[self.target_column] == class_]
            
            freqs_serie, _ = get_freqs_from_text(class_df[self.text_column])
            
            freqs_series.append(freqs_serie)
            
        words_s_classes = words_w_same_classes(freqs_series, top = top)

        return words_s_classes
    
    def remove_words_w_same_targets(self, top: int = 30):
        
        words_s_classes = self.get_words_w_same_targets(top)
        
        self.remove_words(words_s_classes)
    
    def recuperate_results(self):
        try:
            frequency = self.new_frequency.copy()
        except:
            frequency = self.frequency.copy()
        finally:
            print("The recuperate results method recuperates the last version of the frequency data frame as a freqDist. Make sure to add transformations before calling this method!")
        
        frequency.set_index('index', inplace = True)
        
        frequency = frequency.to_dict()
        
        frequency = frequency['frequency']
        
        self.results = nltk.FreqDist(frequency)
        
        return self.results
    
    def add_results_to_data_frame(self, new_text_column_name: Union[str, None] = None):
        
        if not new_text_column_name: new_text_column_name = self.text_column
        
        def clean_text(index: int, words: Union[nltk.FreqDist, list, set, tuple] = self.results):
            """Clean a given document by taking only words that are chosen as representative of the target

            Args:
                index (int): The index of the document
                words (Union[nltk.FreqDist, dict, list, set, tuple]): The words that we want preserve

            Returns:
                str: The new document
            """
            tokens = self.tokens[index]
            tokens = [token for token in tokens if token in words]
            return " ".join(tokens)
        
        self.data_frame.index = list(range(self.data_frame.shape[0]))
        
        self.data_frame[new_text_column_name] = self.data_frame.index.map(clean_text)
        
        self.text_column = new_text_column_name
        
        # self.data_frame.drop(index=self.data_frame[self.data_frame['text'] == ""].index, inplace=True)
        
        # print(self.data_frame.sample(30))
        
    def plot_wordcloud(self, by_target: bool = True, figsize: tuple = (8, 8), max_font_size: int = 60, max_words: int = 100, background_color = "white"):
        try:
            
            if not by_target:
               
                text = self.data_frame[self.text_column].tolist()
               
                wordcloud(" ".join(text), figsize=figsize, max_font_size=max_font_size, max_words=max_words)
            
            else:
                
                wordcloud_hue(self.data_frame, target=self.target_column, text = self.text_column, figsize=figsize, max_font_size=max_font_size, max_words=max_words)
                
        except:
            raise AttributeError("Don't forget to add result `add_results_to_data_frame` method before plotting the wordcloud(s)!")
    
            
            
      
