import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

import re
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'stopwords'])

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.externals import joblib


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
            pos_tags = nltk.pos_tag(tokenize(sentence))
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


def load_data(database_filepath):
    """
    Load cleaned data from database.

    INPUT:
        database_filepath - a string that descibes the file path of database.

    OUTPUT:
        X - a DataFrame that includes only messages data.
        Y - a DataFrame that includes all categories data.
        category_names - an array that stores names of categories.
    """
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('DisasterResponse', engine)
    X = df['message']
    Y = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    category_names = Y.columns
    return X, Y, category_names


def tokenize(text):
    """
    Tokenize the text and drop stopwords.

    INPUT:
        text - a string.

    OUTPUT:
        clean_tokens - a list that stores words after tokenized.
    """
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        if clean_tok not in stopwords.words('english'):
            clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():      
    """
    Build ML pipeline and make grid search.

    INPUT:
        None.

    OUTPUT:
        cv - grid search model.
    """
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

            ('starting_verb', StartingVerbExtractor()),
            ('length_count', LengthCountor())
        ])),

        ('clf', MultiOutputClassifier(RandomForestClassifier(random_state=42)))
    ])
    parameters = {'clf__estimator__n_estimators': [10]}
    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate the model and print precision, recall, f1 score for each category.

    INPUT:
        model - grid search model.
        X_test - a DataFrame, testing data.
        Y_test - a DataFrame, labels of testing data.
        category_names - an array that stores names of categories.

    OUTPUT:
        None.
    """
    Y_test_pred = model.predict(X_test)
    for i, col in enumerate(category_names):
        test_true = Y_test[col].values
        test_pred = Y_test_pred[:, i]
        print(classification_report(test_true, test_pred))


def save_model(model, model_filepath):
    """
    Save the model into pickle file.

    INPUT:
        model - grid search model.
        model_filepath - a string that descibes the file path of model.

    OUTPUT:
        None.
    """
    joblib.dump(model, model_filepath)


def main():
    """
    Main program that load cleaned data, build moel, train model, evaluate model and save model.

    INPUT:
        None.

    OUTPUT:
        None.
    """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()