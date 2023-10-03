from wordcloud import WordCloud, STOPWORDS
from nltk.util import ngrams
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class EDA_movie_reviews:

    def __init__(self, df_movie_reviews, reviews_col, target_col) -> None:
        """
         Initialize the class.
         
         @param df_movie_reviews - pandas DataFrame with the reviews for the movie.
         @param reviews_col - string column name of the column that contains the list of ratings.
         @param target_col - string column name of the target column.
        """
        self.df_movie_reviews = df_movie_reviews
        self.reviews_col = reviews_col
        self.target_col = target_col
        self.df_movie_reviews["n_tokens"] = self.df_movie_reviews[self.reviews_col].apply(lambda x: len(x.split(' ')))

        self.df_positive_reviews = self.df_movie_reviews.loc[self.df_movie_reviews[self.target_col] == 1]
        self.df_negative_reviews = self.df_movie_reviews.loc[self.df_movie_reviews[self.target_col] == 0]

    def visualize_wordcloud(self):
        """
         Visualize wordcloud for each review type.
        """
        fig, axs = plt.subplots(ncols=2, figsize=(18, 10))
        # Generate a Wordcloud plot for each review type.
        for i, (reviews_type, df_reviews) in enumerate([("POSITIVE", self.df_positive_reviews), ("NEGATIVE", self.df_negative_reviews)]):
            wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white", stopwords = STOPWORDS).generate(str(df_reviews[self.reviews_col]))
            ax = axs[i]
            ax.set_title(f"{reviews_type} Reviews - Wordcloud")
            ax.imshow(wordcloud, interpolation="bilinear")
            ax.axis("off")
        plt.show()

    def visulaize_class_distribution(self):
        """
         Visualize the class distribution of the reviews.
        """
        self.df_movie_reviews[self.target_col].value_counts().plot(kind="bar")
        plt.title("Class Distribution\n(0=NEGATIVE, 1=POSITIVE)")
        plt.ylabel("Count")
        plt.show()

    def statistics_on_review_text(self):
        """
         Print statistics on number of tokens per review.
        """
        # Statistics on count of tokens per movie review
        for reviews_type, df_reviews in [("POSITIVE", self.df_positive_reviews), ("NEGATIVE", self.df_negative_reviews)]:
            print(f"Statistics on count of tokens per {reviews_type} movie review")
            display(df_reviews["n_tokens"].describe(percentiles=[0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]))

    def show_common_n_grams(self, n=3, show_count=20):
        """
         Show the most common n - grams for each reviews type.

         @param n - number of words to use for ngrams
         @param show_count - number of words to show per type
        """
        # prints the most common ngrams for the given type
        for reviews_type, df_reviews in [("POSITIVE", self.df_positive_reviews), ("NEGATIVE", self.df_negative_reviews)]:
            text_ngrams = [ngram for review_text in df_reviews[self.reviews_col] for ngram in ngrams(review_text.split(), n)]
            most_common_ngrams = Counter(text_ngrams).most_common(show_count)
            print('-'*50)
            print(f"most common {n}-grams for {reviews_type}:")
            print('-'*50)
            print("n_grams -- occurence_counts")
            # Print the most common n grams and occurence counts.
            for n_grams, occurence_counts in most_common_ngrams:
                print(n_grams, "--", occurence_counts)
