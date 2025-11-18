"""
Enhanced Text Analytics Module for AI Data Analysis Platform
Comprehensive NLP and text analysis with advanced preprocessing and analytics.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
import re
import warnings
warnings.filterwarnings('ignore')

# NLP Libraries
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer, PorterStemmer, LancasterStemmer, SnowballStemmer
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from nltk import ngrams
import spacy
from textblob import TextBlob
import gensim
from gensim import corpora
from gensim.models import LdaModel, Word2Vec, Doc2Vec
from gensim.models.coherencemodel import CoherenceModel

# Indonesian NLP
try:
    from Sastrawi.Stemmer import StemmerFactory
    from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
    SASTRAWI_AVAILABLE = True
except ImportError:
    SASTRAWI_AVAILABLE = False

# Advanced NLP
try:
    import transformers
    from transformers import pipeline, AutoTokenizer, AutoModel
    from sentence_transformers import SentenceTransformer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# Language Detection
try:
    from langdetect import detect, DetectorFactory
    from polyglot.detect import Detector
    LANGUAGE_DETECTION_AVAILABLE = True
except ImportError:
    LANGUAGE_DETECTION_AVAILABLE = False

# Topic Modeling
try:
    import bertopic
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import NMF, LatentDirichletAllocation
    TOPIC_MODELING_AVAILABLE = True
except ImportError:
    TOPIC_MODELING_AVAILABLE = False

# Text Visualization
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter

try:
    from utils.error_handler import handle_errors, TextAnalyticsError, ValidationError
except ImportError:
    def handle_errors(component_name):
        def decorator(func):
            def wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    print(f"Error in {component_name}: {str(e)}")
                    return None
            return wrapper
        return decorator
    
    class TextAnalyticsError(Exception):
        pass
    
    class ValidationError(Exception):
        pass

try:
    from utils.rate_limiter import rate_limit
except ImportError:
    def rate_limit():
        def decorator(func):
            return func
        return decorator


class AdvancedTextAnalyzer:
    """Advanced text analytics with comprehensive NLP capabilities."""
    
    def __init__(self):
        """Initialize advanced text analyzer."""
        self.data = None
        self.processed_texts = []
        self.language = 'en'
        self.nlp_models = {}
        self.stop_words = set()
        self.stemmers = {}
        self.lemmatizers = {}
        self._initialize_nlp_resources()
    
    def _initialize_nlp_resources(self):
        """Initialize NLP resources and models."""
        try:
            # Download NLTK resources
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
            nltk.download('maxent_ne_chunker', quiet=True)
            nltk.download('words', quiet=True)
            
            # Initialize stemmers
            self.stemmers = {
                'porter': PorterStemmer(),
                'lancaster': LancasterStemmer(),
                'snowball': SnowballStemmer('english')
            }
            
            # Initialize lemmatizer
            self.lemmatizers['wordnet'] = WordNetLemmatizer()
            
            # Load spaCy model
            try:
                self.nlp_models['en'] = spacy.load('en_core_web_sm')
            except OSError:
                print("Warning: spaCy English model not found. Run: python -m spacy download en_core_web_sm")
            
            # Indonesian resources
            if SASTRAWI_AVAILABLE:
                stemmer_factory = StemmerFactory()
                self.stemmers['indonesian'] = stemmer_factory.create_stemmer()
                
                stopword_factory = StopWordRemoverFactory()
                self.stop_words.update(stopword_factory.create_stop_word_remover().stopword_list)
            
            # English stop words
            self.stop_words.update(stopwords.words('english'))
            
        except Exception as e:
            print(f"Warning: Some NLP resources failed to load: {e}")
    
    @handle_errors("text")
    def set_data(self, data: pd.Series, text_column: str = None):
        """Set text data for analysis."""
        if isinstance(data, pd.DataFrame):
            if text_column is None:
                raise ValidationError("text_column must be specified when data is DataFrame")
            self.data = data[text_column].dropna()
        elif isinstance(data, pd.Series):
            self.data = data.dropna()
        else:
            raise ValidationError("Data must be pandas Series or DataFrame")
        
        # Detect language
        self.language = self._detect_language(self.data.iloc[0] if len(self.data) > 0 else "")
    
    def _detect_language(self, text: str) -> str:
        """Detect language of text."""
        if not LANGUAGE_DETECTION_AVAILABLE:
            return 'en'
        
        try:
            # Use langdetect for primary detection
            lang = detect(text)
            if lang in ['id', 'ms']:  # Indonesian or Malay
                return 'id'
            elif lang == 'en':
                return 'en'
            else:
                return lang
        except:
            return 'en'  # Default to English
    
    @handle_errors("text")
    def preprocess_texts(self, config: Dict[str, Any] = None) -> List[str]:
        """Comprehensive text preprocessing."""
        if self.data is None:
            raise ValidationError("No data set for preprocessing")
        
        config = config or self._get_default_preprocessing_config()
        
        processed_texts = []
        
        for text in self.data:
            # Basic cleaning
            processed_text = self._basic_cleaning(text, config)
            
            # Tokenization
            tokens = self._tokenize(processed_text, config)
            
            # Stop word removal
            if config.get('remove_stopwords', True):
                tokens = self._remove_stopwords(tokens, config)
            
            # Stemming/Lemmatization
            if config.get('stemming', False):
                tokens = self._apply_stemming(tokens, config)
            elif config.get('lemmatization', True):
                tokens = self._apply_lemmatization(tokens, config)
            
            # Filter tokens
            tokens = self._filter_tokens(tokens, config)
            
            processed_texts.append(' '.join(tokens))
        
        self.processed_texts = processed_texts
        return processed_texts
    
    def _get_default_preprocessing_config(self) -> Dict[str, Any]:
        """Get default preprocessing configuration."""
        return {
            'lowercase': True,
            'remove_punctuation': True,
            'remove_numbers': False,
            'remove_special_chars': True,
            'remove_stopwords': True,
            'stemming': False,
            'lemmatization': True,
            'min_word_length': 2,
            'max_word_length': 20,
            'stemmer_type': 'porter',
            'language': self.language
        }
    
    def _basic_cleaning(self, text: str, config: Dict[str, Any]) -> str:
        """Basic text cleaning."""
        if not isinstance(text, str):
            text = str(text)
        
        # Lowercase
        if config.get('lowercase', True):
            text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove phone numbers
        text = re.sub(r'\+?\d{1,3}[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove punctuation
        if config.get('remove_punctuation', True):
            text = re.sub(r'[^\w\s]', ' ', text)
        
        # Remove numbers
        if config.get('remove_numbers', True):
            text = re.sub(r'\d+', '', text)
        
        # Remove special characters
        if config.get('remove_special_chars', True):
            text = re.sub(r'[^\w\s]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _tokenize(self, text: str, config: Dict[str, Any]) -> List[str]:
        """Tokenize text."""
        if config.get('language') == 'id' and SASTRAWI_AVAILABLE:
            # Use simple tokenization for Indonesian
            tokens = text.split()
        else:
            # Use NLTK tokenization
            try:
                tokens = word_tokenize(text)
            except:
                tokens = text.split()
        
        return tokens
    
    def _remove_stopwords(self, tokens: List[str], config: Dict[str, Any]) -> List[str]:
        """Remove stopwords from tokens."""
        lang = config.get('language', 'en')
        
        # Get appropriate stop words
        if lang == 'id' and SASTRAWI_AVAILABLE:
            stopword_factory = StopWordRemoverFactory()
            stop_words = set(stopword_factory.create_stop_word_remover().stopword_list)
        else:
            stop_words = set(stopwords.words(lang))
        
        # Add custom stop words
        custom_stopwords = config.get('custom_stopwords', [])
        stop_words.update(custom_stopwords)
        
        return [token for token in tokens if token.lower() not in stop_words]
    
    def _apply_stemming(self, tokens: List[str], config: Dict[str, Any]) -> List[str]:
        """Apply stemming to tokens."""
        stemmer_type = config.get('stemmer_type', 'porter')
        lang = config.get('language', 'en')
        
        if lang == 'id' and SASTRAWI_AVAILABLE:
            stemmer = self.stemmers['indonesian']
            return [stemmer.stem(token) for token in tokens]
        elif stemmer_type in self.stemmers:
            stemmer = self.stemmers[stemmer_type]
            return [stemmer.stem(token) for token in tokens]
        else:
            return tokens
    
    def _apply_lemmatization(self, tokens: List[str], config: Dict[str, Any]) -> List[str]:
        """Apply lemmatization to tokens."""
        if 'wordnet' not in self.lemmatizers:
            return tokens
        
        lemmatizer = self.lemmatizers['wordnet']
        
        # Get POS tags for better lemmatization
        try:
            pos_tags = pos_tag(tokens)
            lemmatized_tokens = []
            
            for token, pos_tag in pos_tags:
                pos = self._get_wordnet_pos(pos_tag)
                if pos:
                    lemmatized_tokens.append(lemmatizer.lemmatize(token, pos=pos))
                else:
                    lemmatized_tokens.append(lemmatizer.lemmatize(token))
            
            return lemmatized_tokens
        except:
            return [lemmatizer.lemmatize(token) for token in tokens]
    
    def _get_wordnet_pos(self, treebank_tag: str) -> str:
        """Convert Treebank POS tags to WordNet POS tags."""
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            return None
    
    def _filter_tokens(self, tokens: List[str], config: Dict[str, Any]) -> List[str]:
        """Filter tokens based on length and other criteria."""
        min_length = config.get('min_word_length', 2)
        max_length = config.get('max_word_length', 20)
        
        filtered_tokens = []
        for token in tokens:
            if min_length <= len(token) <= max_length and token.isalpha():
                filtered_tokens.append(token)
        
        return filtered_tokens
    
    @handle_errors("text")
    def extract_entities(self, text: str = None) -> Dict[str, List[str]]:
        """Extract named entities from text."""
        if text is None:
            if self.processed_texts:
                text = ' '.join(self.processed_texts[:5])  # Use first 5 processed texts
            elif self.data is not None and len(self.data) > 0:
                text = self.data.iloc[0]
            else:
                raise ValidationError("No text available for entity extraction")
        
        entities = {
            'persons': [],
            'organizations': [],
            'locations': [],
            'dates': [],
            'miscellaneous': []
        }
        
        # Use spaCy for entity extraction
        if self.language in self.nlp_models:
            doc = self.nlp_models[self.language](text)
            
            for ent in doc.ents:
                if ent.label_ == 'PERSON':
                    entities['persons'].append(ent.text)
                elif ent.label_ == 'ORG':
                    entities['organizations'].append(ent.text)
                elif ent.label_ in ['GPE', 'LOC']:
                    entities['locations'].append(ent.text)
                elif ent.label_ == 'DATE':
                    entities['dates'].append(ent.text)
                else:
                    entities['miscellaneous'].append(ent.text)
        
        # Use NLTK as fallback
        else:
            try:
                tokens = word_tokenize(text)
                pos_tags = pos_tag(tokens)
                chunks = ne_chunk(pos_tags)
                
                for chunk in chunks:
                    if hasattr(chunk, 'label'):
                        entity_text = ' '.join([token for token, pos in chunk.leaves()])
                        if chunk.label() == 'PERSON':
                            entities['persons'].append(entity_text)
                        elif chunk.label() == 'ORGANIZATION':
                            entities['organizations'].append(entity_text)
                        elif chunk.label() in ['GPE', 'LOCATION']:
                            entities['locations'].append(entity_text)
                        else:
                            entities['miscellaneous'].append(entity_text)
            except:
                pass
        
        # Remove duplicates
        for key in entities:
            entities[key] = list(set(entities[key]))
        
        return entities
    
    @handle_errors("text")
    def analyze_sentiment(self, method: str = 'textblob') -> Dict[str, Any]:
        """Analyze sentiment of texts."""
        if self.data is None:
            raise ValidationError("No data set for sentiment analysis")
        
        sentiments = {
            'polarity': [],
            'subjectivity': [],
            'sentiment_labels': [],
            'confidence': []
        }
        
        for text in self.data:
            if method == 'textblob':
                blob = TextBlob(str(text))
                polarity = blob.sentiment.polarity
                subjectivity = blob.sentiment.subjectivity
                
                sentiments['polarity'].append(polarity)
                sentiments['subjectivity'].append(subjectivity)
                
                # Classify sentiment
                if polarity > 0.1:
                    label = 'positive'
                elif polarity < -0.1:
                    label = 'negative'
                else:
                    label = 'neutral'
                
                sentiments['sentiment_labels'].append(label)
                sentiments['confidence'].append(abs(polarity))
            
            elif method == 'vader' and TRANSFORMERS_AVAILABLE:
                # Use VADER sentiment analyzer
                from nltk.sentiment import SentimentIntensityAnalyzer
                sia = SentimentIntensityAnalyzer()
                scores = sia.polarity_scores(str(text))
                
                sentiments['polarity'].append(scores['compound'])
                sentiments['subjectivity'].append(0)  # VADER doesn't provide subjectivity
                
                if scores['compound'] >= 0.05:
                    label = 'positive'
                elif scores['compound'] <= -0.05:
                    label = 'negative'
                else:
                    label = 'neutral'
                
                sentiments['sentiment_labels'].append(label)
                sentiments['confidence'].append(abs(scores['compound']))
        
        # Calculate summary statistics
        sentiment_summary = {
            'average_polarity': np.mean(sentiments['polarity']),
            'average_subjectivity': np.mean(sentiments['subjectivity']),
            'sentiment_distribution': dict(Counter(sentiments['sentiment_labels'])),
            'detailed_sentiments': sentiments
        }
        
        return sentiment_summary
    
    @handle_errors("text")
    def extract_keywords(self, method: str = 'tfidf', top_k: int = 10) -> Dict[str, List[str]]:
        """Extract keywords from texts."""
        if not self.processed_texts:
            self.preprocess_texts()
        
        keywords = {}
        
        if method == 'tfidf':
            keywords = self._extract_keywords_tfidf(top_k)
        elif method == 'yake':
            keywords = self._extract_keywords_yake(top_k)
        elif method == 'textrank':
            keywords = self._extract_keywords_textrank(top_k)
        elif method == 'frequency':
            keywords = self._extract_keywords_frequency(top_k)
        
        return keywords
    
    def _extract_keywords_tfidf(self, top_k: int) -> Dict[str, List[str]]:
        """Extract keywords using TF-IDF."""
        try:
            vectorizer = TfidfVectorizer(
                max_features=1000,
                ngram_range=(1, 2),
                stop_words='english' if self.language == 'en' else None
            )
            
            tfidf_matrix = vectorizer.fit_transform(self.processed_texts)
            feature_names = vectorizer.get_feature_names_out()
            
            # Get average TF-IDF scores
            mean_scores = np.mean(tfidf_matrix.toarray(), axis=0)
            
            # Get top keywords
            top_indices = np.argsort(mean_scores)[::-1][:top_k]
            keywords = [feature_names[i] for i in top_indices]
            
            return {'tfidf_keywords': keywords}
        except Exception as e:
            return {'tfidf_keywords': []}
    
    def _extract_keywords_yake(self, top_k: int) -> Dict[str, List[str]]:
        """Extract keywords using YAKE algorithm."""
        try:
            import yake
            kw_extractor = yake.KeywordExtractor(
                lan=self.language,
                n=2,
                dedupLim=0.7,
                top=top_k,
                features=None
            )
            
            all_keywords = []
            for text in self.processed_texts:
                keywords = kw_extractor.extract_keywords(text)
                all_keywords.extend([kw for kw, score in keywords])
            
            # Get most common keywords
            keyword_counts = Counter(all_keywords)
            top_keywords = [kw for kw, count in keyword_counts.most_common(top_k)]
            
            return {'yake_keywords': top_keywords}
        except:
            return {'yake_keywords': []}
    
    def _extract_keywords_textrank(self, top_k: int) -> Dict[str, List[str]]:
        """Extract keywords using TextRank algorithm."""
        try:
            # Simple TextRank implementation
            all_text = ' '.join(self.processed_texts)
            words = all_text.split()
            
            # Create word co-occurrence matrix
            word_set = list(set(words))
            word_index = {word: i for i, word in enumerate(word_set)}
            
            co_matrix = np.zeros((len(word_set), len(word_set)))
            
            window_size = 4
            for i in range(len(words) - window_size + 1):
                window = words[i:i + window_size]
                for j in range(len(window)):
                    for k in range(j + 1, len(window)):
                        word1, word2 = window[j], window[k]
                        if word1 in word_index and word2 in word_index:
                            idx1, idx2 = word_index[word1], word_index[word2]
                            co_matrix[idx1][idx2] += 1
                            co_matrix[idx2][idx1] += 1
            
            # Apply PageRank-like algorithm
            scores = np.ones(len(word_set))
            for _ in range(10):  # 10 iterations
                new_scores = np.zeros(len(word_set))
                for i in range(len(word_set)):
                    for j in range(len(word_set)):
                        if co_matrix[i][j] > 0:
                            new_scores[i] += co_matrix[i][j] * scores[j] / np.sum(co_matrix[j])
                    new_scores[i] = 0.85 + 0.15 * new_scores[i]
                scores = new_scores
            
            # Get top keywords
            top_indices = np.argsort(scores)[::-1][:top_k]
            keywords = [word_set[i] for i in top_indices]
            
            return {'textrank_keywords': keywords}
        except:
            return {'textrank_keywords': []}
    
    def _extract_keywords_frequency(self, top_k: int) -> Dict[str, List[str]]:
        """Extract keywords using frequency analysis."""
        all_words = []
        for text in self.processed_texts:
            words = text.split()
            all_words.extend(words)
        
        word_counts = Counter(all_words)
        top_keywords = [word for word, count in word_counts.most_common(top_k)]
        
        return {'frequency_keywords': top_keywords}
    
    @handle_errors("text")
    def perform_topic_modeling(self, method: str = 'lda', num_topics: int = 5) -> Dict[str, Any]:
        """Perform topic modeling on texts."""
        if not self.processed_texts:
            self.preprocess_texts()
        
        if method == 'lda':
            return self._lda_topic_modeling(num_topics)
        elif method == 'nmf':
            return self._nmf_topic_modeling(num_topics)
        elif method == 'bertopic' and TOPIC_MODELING_AVAILABLE:
            return self._bertopic_modeling(num_topics)
        else:
            raise ValidationError(f"Unknown topic modeling method: {method}")
    
    def _lda_topic_modeling(self, num_topics: int) -> Dict[str, Any]:
        """Perform LDA topic modeling."""
        # Create dictionary and corpus
        tokenized_texts = [text.split() for text in self.processed_texts]
        dictionary = corpora.Dictionary(tokenized_texts)
        corpus = [dictionary.doc2bow(text) for text in tokenized_texts]
        
        # Train LDA model
        lda_model = LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=num_topics,
            random_state=42,
            passes=10,
            alpha='auto',
            per_word_topics=True
        )
        
        # Extract topics
        topics = []
        for idx, topic in lda_model.print_topics(-1):
            topics.append({
                'topic_id': idx,
                'words': topic,
                'word_list': [word.split('*')[1].strip('"') for word in topic.split(' + ')]
            })
        
        # Calculate coherence
        coherence_model = CoherenceModel(
            model=lda_model,
            texts=tokenized_texts,
            dictionary=dictionary,
            coherence='c_v'
        )
        coherence_score = coherence_model.get_coherence()
        
        return {
            'method': 'lda',
            'topics': topics,
            'coherence_score': coherence_score,
            'model': lda_model,
            'dictionary': dictionary,
            'corpus': corpus
        }
    
    def _nmf_topic_modeling(self, num_topics: int) -> Dict[str, Any]:
        """Perform NMF topic modeling."""
        try:
            vectorizer = TfidfVectorizer(
                max_features=1000,
                ngram_range=(1, 2),
                stop_words='english' if self.language == 'en' else None
            )
            
            tfidf_matrix = vectorizer.fit_transform(self.processed_texts)
            
            nmf_model = NMF(
                n_components=num_topics,
                random_state=42,
                max_iter=200
            )
            
            nmf_model.fit(tfidf_matrix)
            
            feature_names = vectorizer.get_feature_names_out()
            topics = []
            
            for topic_idx, topic in enumerate(nmf_model.components_):
                top_words_idx = topic.argsort()[-10:][::-1]
                top_words = [feature_names[i] for i in top_words_idx]
                
                topics.append({
                    'topic_id': topic_idx,
                    'word_list': top_words,
                    'weights': topic[top_words_idx].tolist()
                })
            
            return {
                'method': 'nmf',
                'topics': topics,
                'model': nmf_model,
                'vectorizer': vectorizer
            }
        except Exception as e:
            return {'error': f"NMF topic modeling failed: {str(e)}"}
    
    def _bertopic_modeling(self, num_topics: int) -> Dict[str, Any]:
        """Perform BERTopic modeling."""
        try:
            from bertopic import BERTopic
            
            topic_model = BERTopic(
                nr_topics=num_topics,
                language=self.language,
                calculate_probabilities=True,
                verbose=True
            )
            
            topics, probs = topic_model.fit_transform(self.processed_texts)
            
            # Get topic info
            topic_info = topic_model.get_topic_info()
            
            return {
                'method': 'bertopic',
                'topics': topic_info.to_dict('records'),
                'topic_assignments': topics,
                'probabilities': probs,
                'model': topic_model
            }
        except Exception as e:
            return {'error': f"BERTopic modeling failed: {str(e)}"}
    
    @handle_errors("text")
    def analyze_ngrams(self, n: int = 2, top_k: int = 20) -> Dict[str, List[Tuple[str, int]]]:
        """Analyze n-grams in texts."""
        if not self.processed_texts:
            self.preprocess_texts()
        
        all_ngrams = []
        
        for text in self.processed_texts:
            words = text.split()
            if len(words) >= n:
                text_ngrams = list(ngrams(words, n))
                all_ngrams.extend([' '.join(ngram) for ngram in text_ngrams])
        
        ngram_counts = Counter(all_ngrams)
        top_ngrams = ngram_counts.most_common(top_k)
        
        return {
            f'{n}_grams': top_ngrams,
            'total_unique_ngrams': len(ngram_counts),
            'total_ngrams': len(all_ngrams)
        }
    
    @handle_errors("text")
    def calculate_readability(self) -> Dict[str, Any]:
        """Calculate readability scores for texts."""
        if self.data is None:
            raise ValidationError("No data set for readability analysis")
        
        readability_scores = {
            'flesch_reading_ease': [],
            'flesch_kincaid_grade': [],
            'gunning_fog': [],
            'coleman_liau': [],
            'automated_readability': [],
            'dale_chall': []
        }
        
        for text in self.data:
            text = str(text)
            
            # Basic text statistics
            sentences = sent_tokenize(text)
            words = word_tokenize(text)
            syllables = sum(self._count_syllables(word) for word in words)
            
            num_sentences = len(sentences)
            num_words = len(words)
            
            if num_sentences == 0 or num_words == 0:
                continue
            
            # Flesch Reading Ease
            flesch_score = 206.835 - 1.015 * (num_words / num_sentences) - 84.6 * (syllables / num_words)
            readability_scores['flesch_reading_ease'].append(flesch_score)
            
            # Flesch-Kincaid Grade Level
            flesch_kincaid = 0.39 * (num_words / num_sentences) + 11.8 * (syllables / num_words) - 15.59
            readability_scores['flesch_kincaid_grade'].append(flesch_kincaid)
            
            # Gunning Fog Index
            complex_words = sum(1 for word in words if self._count_syllables(word) >= 3)
            gunning_fog = 0.4 * ((num_words / num_sentences) + 100 * (complex_words / num_words))
            readability_scores['gunning_fog'].append(gunning_fog)
        
        # Calculate averages
        avg_scores = {}
        for metric, scores in readability_scores.items():
            if scores:
                avg_scores[metric] = {
                    'average': np.mean(scores),
                    'min': np.min(scores),
                    'max': np.max(scores),
                    'std': np.std(scores)
                }
            else:
                avg_scores[metric] = {'average': 0, 'min': 0, 'max': 0, 'std': 0}
        
        return avg_scores
    
    def _count_syllables(self, word: str) -> int:
        """Count syllables in a word."""
        word = word.lower()
        vowels = "aeiouy"
        syllable_count = 0
        prev_char_was_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_char_was_vowel:
                syllable_count += 1
            prev_char_was_vowel = is_vowel
        
        # Handle silent 'e'
        if word.endswith('e'):
            syllable_count -= 1
        
        # Ensure at least one syllable
        return max(1, syllable_count)
    
    @handle_errors("text")
    def create_word_cloud(self, max_words: int = 100, background_color: str = 'white') -> WordCloud:
        """Create word cloud from texts."""
        if not self.processed_texts:
            self.preprocess_texts()
        
        all_text = ' '.join(self.processed_texts)
        
        wordcloud = WordCloud(
            width=800,
            height=400,
            max_words=max_words,
            background_color=background_color,
            colormap='viridis'
        ).generate(all_text)
        
        return wordcloud
    
    @handle_errors("text")
    def generate_text_embeddings(self, method: str = 'sentence_transformers') -> np.ndarray:
        """Generate text embeddings."""
        if not self.processed_texts:
            self.preprocess_texts()
        
        if method == 'sentence_transformers' and TRANSFORMERS_AVAILABLE:
            try:
                model = SentenceTransformer('all-MiniLM-L6-v2')
                embeddings = model.encode(self.processed_texts)
                return embeddings
            except:
                pass
        
        elif method == 'word2vec':
            # Train Word2Vec model
            tokenized_texts = [text.split() for text in self.processed_texts]
            model = Word2Vec(
                sentences=tokenized_texts,
                vector_size=100,
                window=5,
                min_count=1,
                workers=4
            )
            
            # Get document embeddings by averaging word vectors
            embeddings = []
            for text in tokenized_texts:
                word_vectors = [model.wv[word] for word in text if word in model.wv]
                if word_vectors:
                    embeddings.append(np.mean(word_vectors, axis=0))
                else:
                    embeddings.append(np.zeros(100))
            
            return np.array(embeddings)
        
        elif method == 'tfidf':
            # Use TF-IDF as embeddings
            vectorizer = TfidfVectorizer(max_features=1000)
            embeddings = vectorizer.fit_transform(self.processed_texts).toarray()
            return embeddings
        
        else:
            raise ValidationError(f"Unknown embedding method: {method}")
    
    @handle_errors("text")
    def classify_text(self, texts: List[str], categories: List[str] = None) -> Dict[str, Any]:
        """Classify texts into categories."""
        if categories is None:
            categories = ['positive', 'negative', 'neutral']
        
        classifications = {
            'predictions': [],
            'probabilities': [],
            'confidence': []
        }
        
        for text in texts:
            # Simple rule-based classification as fallback
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            
            if polarity > 0.1:
                prediction = 'positive'
            elif polarity < -0.1:
                prediction = 'negative'
            else:
                prediction = 'neutral'
            
            confidence = abs(polarity)
            
            classifications['predictions'].append(prediction)
            classifications['probabilities'].append({prediction: confidence})
            classifications['confidence'].append(confidence)
        
        return classifications
    
    @rate_limit()
    def get_text_analysis_summary(self) -> Dict[str, Any]:
        """Get comprehensive text analysis summary."""
        if self.data is None:
            raise ValidationError("No data set for analysis")
        
        summary = {
            'basic_stats': self._get_basic_text_stats(),
            'language_detected': self.language,
            'preprocessing_applied': len(self.processed_texts) > 0,
            'sample_entities': self.extract_entities(),
            'sentiment_summary': self.analyze_sentiment(),
            'top_keywords': self.extract_keywords(top_k=10),
            'readability_scores': self.calculate_readability()
        }
        
        return summary
    
    def _get_basic_text_stats(self) -> Dict[str, Any]:
        """Get basic text statistics."""
        if self.data is None:
            return {}
        
        text_lengths = self.data.astype(str).str.len()
        word_counts = self.data.astype(str).str.split().str.len()
        
        return {
            'total_texts': len(self.data),
            'avg_text_length': text_lengths.mean(),
            'max_text_length': text_lengths.max(),
            'min_text_length': text_lengths.min(),
            'avg_word_count': word_counts.mean(),
            'max_word_count': word_counts.max(),
            'min_word_count': word_counts.min(),
            'total_characters': text_lengths.sum(),
            'total_words': word_counts.sum()
        }
    
    def export_results(self, filename: str, format: str = 'json') -> str:
        """Export text analysis results."""
        results = self.get_text_analysis_summary()
        
        if format == 'json':
            import json
            filepath = f"exports/{filename}.json"
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2, default=str)
        elif format == 'csv':
            # Create DataFrame with results
            df_results = pd.DataFrame({
                'original_text': self.data,
                'processed_text': self.processed_texts if self.processed_texts else '',
                'text_length': self.data.astype(str).str.len(),
                'word_count': self.data.astype(str).str.split().str.len()
            })
            
            filepath = f"exports/{filename}.csv"
            df_results.to_csv(filepath, index=False)
        else:
            raise ValidationError(f"Unsupported export format: {format}")
        
        return filepath


class IndonesianTextAnalyzer(AdvancedTextAnalyzer):
    """Specialized text analyzer for Indonesian text."""
    
    def __init__(self):
        """Initialize Indonesian text analyzer."""
        super().__init__()
        self.language = 'id'
        self._initialize_indonesian_resources()
    
    def _initialize_indonesian_resources(self):
        """Initialize Indonesian-specific NLP resources."""
        if SASTRAWI_AVAILABLE:
            # Indonesian stemmer
            stemmer_factory = StemmerFactory()
            self.stemmers['indonesian'] = stemmer_factory.create_stemmer()
            
            # Indonesian stop words
            stopword_factory = StopWordRemoverFactory()
            self.stop_words.update(stopword_factory.create_stop_word_remover().stopword_list)
        
        # Add custom Indonesian stop words
        indonesian_stopwords = {
            'yang', 'dan', 'di', 'ke', 'pada', 'untuk', 'dengan', 'adalah',
            'ini', 'itu', 'tersebut', 'dari', 'sebagai', 'dalam', 'akan',
            'telah', 'juga', 'atau', 'karena', 'jika', 'seperti', 'oleh',
            'bisa', 'dapat', 'yaitu', 'yakni', 'yaitu', 'adalah'
        }
        self.stop_words.update(indonesian_stopwords)
    
    def _detect_language(self, text: str) -> str:
        """Override to default to Indonesian."""
        return 'id'


# Standalone functions for backward compatibility
def analyze_text_column(df, text_column):
    """
    Analyze text column (standalone function)
    """
    try:
        analyzer = AdvancedTextAnalyzer()
        analyzer.set_data(df, text_column)
        
        # Get basic statistics
        stats = analyzer._get_basic_text_stats()
        
        # Get sentiment analysis
        sentiment = analyzer.analyze_sentiment()
        
        # Get keywords
        keywords = analyzer.extract_keywords(top_k=10)
        
        # Create word cloud
        wordcloud = analyzer.create_word_cloud()
        
        return {
            'statistics': stats,
            'sentiment': sentiment,
            'keywords': keywords,
            'wordcloud': wordcloud
        }
    except Exception as e:
        return {
            'error': str(e),
            'statistics': {},
            'sentiment': pd.DataFrame(),
            'keywords': [],
            'wordcloud': None
        }


def analyze_sentiment(texts, method='textblob'):
    """
    Analyze sentiment of texts (standalone function)
    """
    try:
        if isinstance(texts, str):
            texts = [texts]
        
        sentiments = []
        for text in texts:
            if method == 'textblob':
                blob = TextBlob(text)
                polarity = blob.sentiment.polarity
                subjectivity = blob.sentiment.subjectivity
                
                # Classify sentiment
                if polarity > 0.1:
                    label = 'positive'
                elif polarity < -0.1:
                    label = 'negative'
                else:
                    label = 'neutral'
                
                sentiments.append({
                    'text': text,
                    'polarity': polarity,
                    'subjectivity': subjectivity,
                    'label': label
                })
        
        return pd.DataFrame(sentiments)
    except Exception as e:
        return pd.DataFrame()