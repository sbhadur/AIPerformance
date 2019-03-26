import numpy as np
import pandas as pd
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import re

def load():
    df = pd.read_json("DataSets/reddit_jokes.json")
    df2 = pd.read_json("DataSets/stupidstuff.json")
    df3 = pd.read_json("DataSets/wocka.json")

    # Convert into binary Y based on score
    df['score'] = np.where(df['score'] > 10, 1, -1)
    df = df.drop(columns=['id','title'])

    df2 = df2.drop(columns=['id', 'category'])
    df2['score'] = np.where(df2['rating'] > 2.5, 1, -1)
    df2 = df2.drop(columns=['rating'])

    df = pd.concat([df, df2])

    df3 = df3.drop(columns=['category', 'id', 'title'])
    df3['score'] = 1

    df = pd.concat([df, df3])

    return df

def clean():
    df = load()
    stemmer = SnowballStemmer('english')
    words = stopwords.words('english')

    df['clean'] = df['body'].apply(lambda x: " ".join([stemmer.stem(i) for i in re.sub("[^a-zA-Z]", " ", x).split() if i not in words]))
    df = df.drop(columns=['body'])

    return df
