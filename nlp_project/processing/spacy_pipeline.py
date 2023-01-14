
from nlp_project.processing.utils import *
from typing import List

class TextPipeProcessing:
    """The pipeline is composed by (* if obligatory processing; ** if most important):
    - tokenize_text**
    - create_corpus*
    - create_n_grams
    - set_n_grams_as_corpus
    - reset_corpus
    - create_frequency*
    - show_frequency_out_limits
    - show_most_common_words
    - plot_frequency_histogram
    - show_n_time_frequency_words
    - delete_n_time_frequency_words
    - stem words (if no lemming)
    - corpus_filter 
    - get_words_w_same_targets
    - remove_words
    - remove_words_w_same_targets
    - recuperate_results*
    - add_results_to_data_frame*
    - plot_wordcloud
    - get_tf_idf
    - some other issues...
    - use context manager to store a pipeline
    """
    pipeline = {}
    iteration = 0
    
    def __init__(self, data_frame: pd.DataFrame, text_column: str, target_column: str = 'target', name: str = "nlp_pipeline"):
        """Initialize main attributes

        Args:
            data_frame (pd.DataFrame): A dataframe containing a column for documents and column for target
            text_column (str): The name of the text column
            target_column (str, optional): The name of the target column. Defaults to 'target'.
        """
        
        self.name = name
        
        self.data_frame = data_frame
        
        self.text_column = text_column
        
        self.target_column = target_column
        
        self._lem = False
        
        self._stemmer = None
        
        self._corpus = None
        
        self._n_grams = None
        
        self._old_corpus = None
        
        self._grams_active = False
        
        self.bigrams = None
        
        self.trigrams = None

    def __enter__(self):
        
        self.current_pipe = []
        
        return self
    
    def __call__(self, method: Callable, get_results: bool = True, *args, **kwargs):
    
        self.current_pipe.append({"method": method, "args": args, "kwargs": kwargs, "result": get_results})
        
    def tokenize_text(self,
                      nlp, 
                      rm_stopwords: bool = True,
                      rm_punctations: bool = True,
                      rm_non_ascii: bool = True,
                      only_alpha: bool = True,
                      rm_spaces: bool = True,
                      keep_urls: bool = False,
                      keep_emoji: bool = False,
                      keep_html: bool = False,
                      keep_mentions: bool = False,
                      keep_upper: bool = True,
                      lem: bool = True,
                      min_length: Union[int, None] = None,
                      max_length: Union[int, None] = None,
                      entities_to_remove: Union[List[str], None] = None,
                      pos_tags: Union[List[str], None] = None,
                      ):
        
        self._nlp = nlp
        
        if lem:
            self._lem = True
        
        texts = self.data_frame[self.text_column]
        
        self._texts = texts.tolist()
        
        self._tokenizer = lambda texts: tokenization(nlp,
                      texts,
                      rm_stopwords,
                      rm_punctations,
                      rm_non_ascii,
                      only_alpha,
                      rm_spaces,
                      keep_urls,
                      keep_emoji,
                      keep_html,
                      keep_mentions,
                      keep_upper,
                      lem,
                      min_length,
                      max_length,
                      entities_to_remove,
                      pos_tags,
                    )
        
        self._tokens, self._pos_tags = self._tokenizer(self._texts)
    
        # self.iteration += 1
        
        return self._tokens, self._pos_tags
    
    def get_tf_idf(self):
        
        tf_idf = TfidfVectorizer(tokenizer = self._tokenizer)
        
        values = tf_idf.fit_transform(self._texts)
        
        return values
    
    def create_corpus(self):
        
        self._corpus = []
        
        for document in tqdm(self._tokens):
            
            self._corpus.extend(document)
        
        self._corpus_text = nltk.text.Text(self._corpus)
        
        print(f"Number of words: {len(self._corpus):->16}")
        print(f"Number of unique words: {len(self._corpus_text.vocab()):->16}")
        
        # self.pipeline[self.iteration] = "create_corpus"
        
        return self._corpus, self._corpus_text
    
    def create_n_grams(self, n: int = 2):
        
        assert n >= 2
        
        self._n_grams = []
        
        for document in tqdm(self._tokens):
            
            n_gram = get_n_grams(document, n)
            
            self._n_grams.extend(n_gram)
        
        self._n_grams_text = nltk.text.Text(self._n_grams)
        
        print(f"Number of {n} grams: {len(self._n_grams):->16}")
        print(f"Number of unique {n} grams: {len(self._n_grams_text.vocab()):->16}")

        return self._n_grams, self._n_grams_text
        
    def set_n_grams_as_corpus(self):
        
        self._old_corpus = self._corpus
        
        self._old_corpus_text = self._corpus_text
        
        if not self._n_grams:
            
            raise AttributeError("You didn't create the n grams with the `create_n_grams` method!")
        
        self._corpus = self._n_grams
        
        self._corpus_text = self._n_grams_text
        
        self._grams_active = True

    def reset_corpus(self):
        
        if not self._old_corpus:
            
            raise AttributeError("The corpus was not properly created. To create a new corpus from tokens use the `create_corpus` method!")
        
        self._corpus = self._old_corpus
        
        self._corpus_text = self._old_corpus_text
        
        self._grams_active = False
    
    def create_frequency(self):
        
        self._frequency = pd.DataFrame.from_dict(self._corpus_text.vocab(), 'index')
        
        self._frequency.rename({0: 'frequency'}, inplace=True, axis=1)
        
        self._frequency.reset_index(level=0, inplace=True)
        
        print(self._frequency.head())
    
    def show_frequency_out_limits(self):
        
        px.box(data_frame=self._frequency, x="frequency", hover_data=['index', 'frequency']) 
        
        self.low, self.high = guess_limitations(self._frequency, 'frequency') 
        
        print(f"Low limit: {self.low:->16}")
        print(f"High limit: {self.high:->16}")
    
    def show_most_common_words(self, lower_bound: int = 400, n_words: int = 20):
        
        self._freq_total = nltk.Counter(self._corpus_text.vocab())
        
        self._stopwords_common = list(zip(*self._freq_total.most_common(lower_bound)))[0]
        
        print("Most common words are:")
        print(self._stopwords_common[:20])
    
    def plot_frequency_histogram(self, bottom: int = 8):
        
        f_values = self._frequency['frequency'].sort_values().unique()        
    
        bottom_ = self._frequency[self._frequency['frequency'].isin(f_values[:bottom])]
        
        fig = px.histogram(data_frame = bottom_, x = 'frequency', title=f"Frequency histogram for {bottom} frequency on the bottom")
        
        fig.show()
        
    def show_n_time_frequency_words(self, n_time_freq: Union[int, list] = 1, n_words: int = 100):
        
        n_time_freq = [n_time_freq] if type(n_time_freq) is int else n_time_freq
        
        size = self._frequency[self._frequency['frequency'].isin(n_time_freq)].shape[0]
        
        n_time_frequency = self._frequency[self._frequency['frequency'].isin(n_time_freq)]
        
        print(f"Percentage of words appearing {'/'.join([str(freq) for freq in n_time_freq])} times in the dataset: {size / self._frequency.shape[0]}%")
        
        print(f"Words appearing {'/'.join([str(freq) for freq in n_time_freq])} times:")
        print(n_time_frequency.iloc[:n_words,:])
    
    def delete_n_time_frequency_words(self, n_time_freq: Union[int, list] = 1):
        
        n_time_freq = [n_time_freq] if type(n_time_freq) is int else n_time_freq
        
        n_time_frequency = self._frequency[self._frequency['frequency'].isin(n_time_freq)]
        
        self._new_frequency = self._frequency.loc[~self._frequency['index'].isin(n_time_frequency['index'].to_list()), :]
        
        print("The new frequency data frame is stored in `_new_frequency` attribute.")
        
        print(f"The number of deleted observations: {n_time_frequency.shape[0]:->16}")
        
    def stem_words(self, stemmer = nltk.stem.PorterStemmer(), force: bool = False):
        
        if not self._lem or force:
            
            self._stemmer = stemmer
            
            self._new_frequency = self._new_frequency.copy()
            
            self._new_frequency.loc["index"] = self._new_frequency["index"].apply(lambda idx: stemmer.stem(idx))
            
            self._new_frequency.dropna(axis = 0, inplace=True)
            
            self._new_frequency['frequency'] = self._new_frequency['frequency'].astype('int32')
        
        else:
            
            print("You have already made lemmatization on the tokens. The stemming is disabled! You can make `force = True` to force the stemming (not recommended).")
    
    def corpus_filter(self, corpus: nltk.corpus.words.words()):
        
        if self._lem:
        
            corpus = [word.lemma_ for word in Doc(self._nlp.vocab, words=corpus)]
        
        elif self._stemmer:
            
            corpus = [self._stemmer.stem(word) for word in corpus]
        
        self._new_frequency = self._new_frequency[self._new_frequency['index'].isin(corpus)]
    
    def remove_words(self, words_to_remove: List[str]):
        
        self._new_frequency = self._new_frequency.copy()
        
        self._new_frequency.drop(index=self._new_frequency[self._new_frequency['index'].isin(words_to_remove)].index, inplace = True)
    
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
            frequency = self._new_frequency.copy()
        except:
            frequency = self._frequency.copy()
        finally:
            print("The recuperate results method recuperates the last version of the frequency data frame as a freqDist. Make sure to add transformations before calling this method!")
        
        frequency.set_index('index', inplace = True)
        
        frequency = frequency.to_dict()
        
        frequency = frequency['frequency']
        
        self._results = nltk.FreqDist(frequency)
        
        if self._grams_active:
            
            keys = list(self._results.keys())
            
            if len(keys[0].split(" ")) == 2:
                
                self._bigrams = self._results
            
            elif len(keys[0].split(" ")) == 3:
                
                self._trigrams = self._results
        
        return self._results
    
    def add_results_to_data_frame(self, new_text_column_name: Union[str, None] = None):
        
        if self._grams_active:
                
                print("You didn't reset the corpus with the `reset_corpus` method!")
        
        if not new_text_column_name: new_text_column_name = self.text_column
        
        def clean_text(index: int, words: Union[nltk.FreqDist, list, set, tuple] = self._results):
            """Clean a given document by taking only words that are chosen as representative of the target

            Args:
                index (int): The index of the document
                words (Union[nltk.FreqDist, dict, list, set, tuple]): The words that we want to preserve

            Returns:
                str: The new document
            """
            
            if len(list(words.keys())[0].split(" ")) != 1:
                
                raise ValueError("Only uni grams can be provide as results to the data frame text column!")

            tokens = self._tokens[index]
            
            if self._stemmer:
                
                tokens = [self._stemmer.stem(token) for token in tokens if self._stemmer.stem(token) in words]
            
            else:
            
                tokens = [token for token in tokens if token in words]
            
            return " ".join(tokens)
        
        self.data_frame.index = list(range(self.data_frame.shape[0]))
        
        self.data_frame[new_text_column_name] = self.data_frame.index.map(clean_text)
        
        self.text_column = new_text_column_name
        
        # self.data_frame.drop(index=self.data_frame[self.data_frame['text'] == ""].index, inplace=True)
        
        # print(self.data_frame.sample(30))
        
    def plot_wordcloud(self, by_target: bool = True, figsize: tuple = (8, 8), max_font_size: int = 60, max_words: int = 100, background_color = "white"):
        """Plot a wordcloud. Can be done by target class

        Args:
            by_target (bool, optional): Define if a wordcloud will be traced for each target class. Defaults to True.
            figsize (tuple, optional): The figure size with width and height. Defaults to (8, 8).
            max_font_size (int, optional): The maximum size of the font. Defaults to 60.
            max_words (int, optional): The maximum number of words on top of frequencies. Defaults to 100.
            background_color (str, optional): The background color. Defaults to "white".

        Raises:
            AttributeError: It is recommended to call the `add_results_to_data_frame` method before 
        """
        
        try:
            
            if not by_target:
               
                text = self.data_frame[self.text_column].tolist()
               
                wordcloud(" ".join(text), figsize=figsize, max_font_size=max_font_size, max_words=max_words)
            
            else:
                
                wordcloud_hue(self.data_frame, target=self.target_column, text = self.text_column, figsize=figsize, max_font_size=max_font_size, max_words=max_words)
                
        except:
            raise AttributeError("Don't forget to add result `add_results_to_data_frame` method before plotting the wordcloud(s)!")
    
    def predict_next_word(self, text: str):

            if self._bigrams and self._trigrams:
                
                bigram = " ".join(text.split(" ")[-2:])
                
                co_occs = []
                
                trigrams = []
                
                for trigram in self._trigrams:
                    
                    if bigram in trigram[:len(bigram)]:
                        
                        if text in set(self._bigrams.keys()):
                            
                            freq1 = self._bigrams[bigram]
                            
                            freq2 = self._trigrams[trigram]
                            
                            co_occs.append(freq2 / freq1)
                            
                            trigrams.append(trigram)

                        else:
                
                            raise KeyError(f"The bigram {text} is not identified in the registered bigrams!")
                
                try:
                
                    max_co_occ = np.array([co_occs]).argmax()
                    
                    max_trigram = trigrams[max_co_occ]
                
                    return max_trigram.split(" ")[-1], co_occs[max_co_occ]
                
                except ValueError:
                    
                    return "", None
            
            else:
                
                raise ValueError("You must create bigrams and trigrams before using them to predict the next word of your text!")
    
    def display(self, text: str, style = "dep"):
        
        # Create a container object
        doc = self._nlp(text)
        
        # Render frame with displacy
        spacy.displacy.render(doc, style=style)
    
    def execute_pipeline(self, name: str = "nlp_pipeline"):
        
        results = []
        
        try:
        
            pipeline = self.pipeline[name]
            
            for pipe in tqdm(pipeline):
                
                args = pipe['args']
                
                kwargs = pipe['kwargs']
                
                method = pipe['method']
                
                result = pipe['result']
                
                results_ = method(*args, **kwargs)
                
                if result:
                    
                    results.append(results_)
            
            return results
        
        except KeyError:
            
            raise ValueError("The pipeline that you specified doesn't exist!")
    
    def __exit__(self, ctx_ept, ctx_value, ctx_tb):
        
        self.pipeline[self.name] = self.current_pipe
        
        print("You can execute the pipeline with the `pipeline_name.execute_pipeline`! The pipelines are available in the attribute `pipeline`.")
        
        return ctx_value 
      
