import nltk
nltk.download('stopwords')
nltk.download('wordnet')
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn import feature_extraction, preprocessing
from pdb import set_trace as breakpoint

import pandas as pd
import re
import torch

def get_part_of_speech(df):
    def pos_per_sent(x):
        return ' '.join([pos for _, pos in nltk.pos_tag(x)])
    def pos_text_pair_per_sent(x):
        return [(pos,t) for t, pos in nltk.pos_tag(x)]
    new_df=pd.DataFrame()
    new_df['text'] = df['text']
    new_df['pos']=df['text'].apply(pos_per_sent)
    new_df['pos_text_pair']=new_df['text'].apply(pos_text_pair_per_sent)

    return new_df

def vectorize(df):
    c_vectorizer=feature_extraction.text.CountVectorizer(lowercase=False)
    df = df.join(pd.DataFrame(c_vectorizer.fit_transform(df['pos']).todense(),
                          columns=c_vectorizer.get_feature_names(),
                          index=df.index))
    return c_vectorizer, df

def plot_histogram(pd_series, bins='auto', x_label='Sentence Length Counts', y_label='Frequency', bar_chart=False):
    # code modified from https://realpython.com/python-histograms/
    import matplotlib.pyplot as plt
    plt.clf()
    title = f'{y_label} of {x_label}'
    if bar_chart:
        #breakpoint()
        plt.bar(pd_series.index[:10], pd_series.values[:10], color='#607c8e') # take first ten
    else:
        pd_series.plot.hist(grid=True, bins=bins, rwidth=0.9,
                   color='#607c8e', density=True, cumulative=True)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(axis='y', alpha=0.75)
    '''
    n, bins, patches = plt.hist(x=d, bins='auto', color='#0504aa', alpha=0.7, rwidth=0.85)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.text(23, 45, r'$\mu=15, b=3$')
    maxfreq = n.max()
    # Set a clean upper y-axis limit.
    plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
    '''
    return plt

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return tuple(res)
    
def remove_rare_words(tokens, common_tokens, max_len):
    return [token if token in common_tokens else '<UNK>' for token in tokens][-max_len:]

def replace_numbers(tokens):
    return [re.sub(r'[0-9]+', '<NUM>', token) for token in tokens]

def tokenize(text, stop_words, lemmatizer):
    text = re.sub(r'[^\w\s]', '', text) # remove special characters
    text = text.lower() # lowercase
    tokens = wordpunct_tokenize(text) # tokenize
    tokens = [lemmatizer.lemmatize(token) for token in tokens] # noun lemmatizer
    tokens = [lemmatizer.lemmatize(token, "v") for token in tokens] # verb lemmatizer
    tokens = [token for token in tokens if token not in stop_words] # remove stopwords
    return tokens

def build_bow_vector(sequence, idx2token):
    vector = [0] * len(idx2token)
    for token_idx in sequence:
        if token_idx not in idx2token:
            raise ValueError('Wrong sequence index found!')
        else:
            vector[token_idx] += 1
    return vector