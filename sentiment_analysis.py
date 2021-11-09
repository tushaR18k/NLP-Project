import pandas as pd
import numpy as np
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer




def func(split='training',path_to_dataset='./Data/TRAINING'):
    data = pd.read_csv('./train_split.csv',sep="\t")
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

