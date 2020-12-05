import sys
import pandas as pd
import numpy as np

import re
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'stopwords'])

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from sklearn.base import BaseEstimator, TransformerMixin


class StartingVerbExtractor(BaseEstimator, TransformerMixin):

    def starting_verb(self, text):
        """
        A new feature that judge the starting word of text is verb or not.

        INPUT:
            text - a string.

        OUTPUT:
            1 - an int that represents the starting word is verb.
            0 - an int that represents the starting word is not verb.
        """
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(word_tokenize(sentence))
            if len(pos_tags) > 0:
                first_word, first_tag = pos_tags[0]
                if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                    return 1
        return 0

    def fit(self, x, y=None):
        """
        Fit method.
        """
        return self

    def transform(self, X):
        """
        Transform method, applying starting_verb function to all messages in training data.

        INPUT:
            X - a DataFrame.

        OUTPUT:
            A DataFrame that includes 0/1 for each messages in training data.
        """
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)


class LengthCountor(BaseEstimator, TransformerMixin):

    def length_count(self, text):
        """
        A new feature that counts the number of words in text, excluding stopwords.

        INPUT:
            text - a string.

        OUTPUT:
            length - an int that represents the number of words in text.
        """
        length = 0
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            tokens = word_tokenize(sentence)
            tokens = [w for w in tokens if w not in stopwords.words('english')]
            length += len(tokens)
        return length

    def fit(self, x, y=None):
        """
        Fit method.
        """
        return self

    def transform(self, X):
        """
        Transform method, applying length_count function to all messages in training data.

        INPUT:
            X - a DataFrame.

        OUTPUT:
            A DataFrame that includes number of words for each messages in training data.
        """
        X_tagged = pd.Series(X).apply(self.length_count)
        return pd.DataFrame(X_tagged)