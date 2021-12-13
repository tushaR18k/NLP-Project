import pandas as pd
import numpy as np
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
#!pip install flair
import flair
flair_sentiment = flair.models.TextClassifier.load('en-sentiment')



def vader_sentiment(split='train',path_to_dataset='./Data/TRAINING'):
    data = pd.read_csv(f'./{split}_split.csv',sep="\t")
    text = np.asarray(data.iloc[:, 6]).tolist()
    sid = SentimentIntensityAnalyzer()
    sentiments = {'pos':[],'neg':[],'neu':[]}
    for i in range(len(text)):
        c = sid.polarity_scores(text[i])
        key=0
        v=0
        ans=100000
        for k in ['neg','neu','pos']:
            if ans > c[k]:
                key = k
                v = c[k]
                ans = c[k]
        sentiments[key].append(i)

    return sentiments


def flair(split='train',path_to_dataset='./Data/TRAINING'):
    data = pd.read_csv(f'./{split}_split.csv',sep="\t")
    text = np.asarray(data.iloc[:, 6]).tolist()
    sentiments = {'POSITIVE':[],'NEGATIVE':[],'NEUTRAL':[]}
    for i in range(len(text)):
        s = flair.data.Sentence(text[i])
        flair_sentiment.predict(s)
        total_sentiment = s.labels
        ans = str(total_sentiment[0]).split()
        sentiments[ans[0]].append((i,ans[1]))
    return sentiments


