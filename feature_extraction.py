import numpy as np
import pandas as pd
from itertools import islice

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.inspection import permutation_importance
from sklearn.utils import parallel_backend
from sklearn.decomposition import PCA
from scipy.stats import spearmanr
from scipy.cluster import hierarchy as hc
from collections import defaultdict
import matplotlib.pyplot as plt
from classification import RandomForestClassifier

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences

class textVectorizer:

    def __init__(self, vectorizer_type, vocab_size=10_000) -> None:
        
        self.vocab_size = vocab_size
        if vectorizer_type == 'tfidf':
            self.initialize_tfidf_vectorizer()
        elif vectorizer_type == 'count':
            self.initialize_count_vectorizer()
        else:
            raise NotImplementedError
    
    def initialize_tfidf_vectorizer(self):
        self.vectorizer = TfidfVectorizer(
            min_df=0.0, 
            max_df=1.0, 
            use_idf=True, 
            ngram_range=(1,3), 
            max_features=self.vocab_size,
        )

    def initialize_count_vectorizer(self):
        self.vectorizer = CountVectorizer(
            min_df=0.0, 
            max_df=1.0, 
            binary=False, 
            ngram_range=(1,3), 
            max_features=self.vocab_size,
        )

    def apply_transform_train(self, reviews):
        return self.vectorizer.fit_transform(reviews)

    def apply_transform_test(self, reviews):
        return self.vectorizer.transform(reviews)

def truncate_or_pad_sequences(sequences, max_sequence_length, padding_value=0):
    """
    Truncate or pad sequences to a specified maximum sequence length.
    """
    truncated_padded_sequences = []
    for sequence in sequences:
        if len(sequence) > max_sequence_length:
            # Truncate if sequence is longer than max_sequence_length
            truncated_sequence = sequence[:max_sequence_length]
        else:
            # Pad if sequence is shorter than max_sequence_length
            padding_length = max_sequence_length - len(sequence)
            padded_sequence = np.pad(sequence, (0, padding_length), mode='constant', constant_values=padding_value)
            truncated_sequence = padded_sequence
        truncated_padded_sequences.append(truncated_sequence)
    return np.dstack(truncated_padded_sequences)

class GloveVectorizer:
    def __init__(self, embedding_dim, vocab_size, max_sequence_length, glove_emb_path):
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length
        self.glove_emb_path = glove_emb_path
        self.word_to_index = {}
        self.embedding_matrix = None

        self.load_glove_embeddings()

    def load_glove_embeddings(self):
        # Initialize word_to_index and embedding_matrix
        self.word_to_index["<PAD>"] = 0  # Reserved for padding
        self.embedding_matrix = np.zeros((self.vocab_size, self.embedding_dim))

        # Load GloVe embeddings
        with open(self.glove_emb_path, 'r', encoding='utf-8') as glove_file:
            for index, line in enumerate(glove_file):
                if index >= self.vocab_size - 1:
                    break

                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], dtype='float32')

                # Ensure the vector has the expected dimension
                if len(vector) == self.embedding_dim:
                    self.word_to_index[word] = index + 1  # Start index from 1, reserve 0 for padding
                    self.embedding_matrix[index + 1] = vector

    def text_to_sequence(self, text):
        words = text.split()
        sequence = [self.word_to_index.get(word, 0) for word in words][:self.max_sequence_length]
        while len(sequence) < self.max_sequence_length:
            sequence.append(0)  # Pad sequence if it's shorter than max_sequence_length
        return sequence

    def vectorize_text(self, reviews):
        review_vectors = []
        for review_text in reviews:
            sequence = self.text_to_sequence(review_text)
            review_vectors.append(self.embedding_matrix[sequence])
        return np.transpose(np.dstack(review_vectors), (2, 0, 1))