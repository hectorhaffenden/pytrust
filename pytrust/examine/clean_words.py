import nltk 
# Need to run nltk.download('wordnet') if haven't before
import string
import re
import pandas as pd
import numpy as np

def remove_punct(text):
    text  = "".join([char for char in text if char not in string.punctuation])
    text = re.sub('[0-9]+', '', text)
    return text


def tokenization(text):
    text = re.split('\W+', text)
    return text



def remove_stopwords(text, stopword):
    text = [word for word in text if word not in stopword]
    return text
    

def lemmatizer(text, wn):
    text = [wn.lemmatize(word) for word in text]
    return text



def clean_data(df):
    ## Step 1: Remove duplicate entries
    df.drop_duplicates(inplace=True)
    
    ## Step 2: Remove punctuation
    punctuation = ['.', '?', '!', '$', '£', '\'', ',']
    word_cols = ['title', 'content']
    for col in word_cols:
        for punc in punctuation:
            df[col] = df[col].str.replace(punc, '')

        # And make all lower case
        df[col] = df[col].str.lower()
        
        
        df[f'{col}_punct'] = df[col].apply(lambda x: remove_punct(x))

        df[f'{col}_tokenized'] = df[f'{col}_punct'].apply(lambda x: tokenization(x.lower()))

        stopword = nltk.corpus.stopwords.words('english')
        df[f'{col}_nonstop'] = df[f'{col}_tokenized'].apply(lambda x: remove_stopwords(x, stopword))


        wn = nltk.WordNetLemmatizer()
        df[f'{col}_lemmatized'] = df[f'{col}_nonstop'].apply(lambda x: lemmatizer(x, wn))


        df[f'{col}_clean'] = df[f'{col}_lemmatized'].str.join(' ')
        
        df.drop([f'{col}_punct', f'{col}_tokenized', f'{col}_nonstop', f'{col}_lemmatized'],
                axis = 1, inplace=True)
        
    
    df = df.drop(['name'], axis = 1)
    
    
    
    return df

def create_fea(df):
    df['title_num_words'] = df['title'].str.split(' ').str.len()
    df['content_num_words'] = df['content'].str.split(' ').str.len()
    
    df['title_num_char'] = df['title'].str.len()
    df['content_num_char'] = df['content'].str.len()
    return df


def remove_custom_stopwords(df, custom_stop = ['morrisons'],
                            cols = ['title', 'content', 'title_clean', 'content_clean']):
    for col in cols:
        for punc in custom_stop:
            df[col] = df[col].str.replace(punc, '')
    return df


