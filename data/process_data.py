import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Load massages and categories data.
    
    INPUT:
        messages_filepath - a string that descibes the file path of messages.
        categories_filepath - a string that descibes the file path of categories.
        
    OUTPUT:
        df - a DataFrame which is merged by messages and categories data.
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, left_on='id', right_on='id')
    return df


def clean_data(df):
    """
    Clean the df DataFrame.
    
    INPUT:
        df - a DataFrame.
        
    OUTPUT:
        df - a DataFrame which has been processed from df.
    """
    categories = pd.DataFrame(df['categories'].str.split(';').tolist())
    row = categories.loc[0, :]
    category_colnames = row.apply(lambda x: x[:-2]).tolist()
    categories.columns = category_colnames
    for column in categories:
        categories[column] = categories[column].str[-1]
        categories[column] = pd.to_numeric(categories[column])
    df.drop('categories', axis=1, inplace=True)
    df = pd.concat([df, categories], axis=1)
    df.drop_duplicates(inplace=True)
    return df


def save_data(df, database_filename):
    """
    Store the cleaned data into database.
    
    INPUT:
        df - a DataFrame which has been cleaned.
        database_filename - a string that descibes the file name of database.
        
    OUTPUT:
        None.
    """
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('DisasterResponse', engine, if_exists='replace', index=False) 


def main():
    """
    Main program that load, clean and save the data.
    
    INPUT:
        None.
        
    OUTPUT:
        None.
    """
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()