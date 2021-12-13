import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch
import os
import re
from collections import Counter
import pdb
# nltk text processors
from PIL import Image
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize
from nltk.stem import WordNetLemmatizer
from functools import partial
from tqdm.notebook import tqdm

tqdm.pandas()
from sklearn.feature_extraction.text import TfidfVectorizer  # TF-IDF
import einops
from utils import *
from params import *
<<<<<<< HEAD
import matplotlib.pyplot as plt
import random
import pickle
class MAMIDataset(Dataset):
    def __init__(self,max_len,max_vocab,transform_apply=False, path_to_dataset='./Data/TRAINING', split='training', transform=None):
        self.path_to_dataset=path_to_dataset
        
        self.transforms=transform

        self.to_tensor = transforms.ToTensor()
        df = data = pd.read_csv(os.path.join(path_to_dataset,f'{split}_split.csv'),sep="\t")
        df=data=data[data.apply(lambda x: os.path.exists(os.path.join(path_to_dataset,x['file_name'])), axis=1)]
        self.image_arr = np.asarray(data.iloc[:, 0]).tolist()
        self.text =  np.asarray(data.iloc[:, 6]).tolist()
=======


class MAMIDataset(Dataset):
    def __init__(self, max_len, max_vocab, transform_apply=False, path_to_dataset='./Data/TRAINING', split='training', transform=None):
        self.path_to_dataset = path_to_dataset

        self.transforms = transform

        self.to_tensor = transforms.ToTensor()
        df = data = pd.read_csv(os.path.join(path_to_dataset, f'{split}_split.csv'), sep="\t")
        self.image_arr = np.asarray(data.iloc[:, 0]).tolist()
        self.text = np.asarray(data.iloc[:, 6]).tolist()
>>>>>>> da5de8aaeb66abdbb2af19c3efed604ac65990ea
        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()
        df['tokens'] = df['Text Transcription'].apply(
            partial(
                tokenize,
                stop_words=stop_words,
                lemmatizer=lemmatizer,
            ),
        )
        all_tokens = [token for doc in list(df.tokens) for token in doc]

        common_tokens = set(
            list(
                zip(*Counter(all_tokens).most_common(max_vocab))
            )[0]
        )
        # pdb.set_trace()
        df.loc[:, 'tokens'] = df.tokens.progress_apply(
            partial(
                remove_rare_words,
                common_tokens=common_tokens,
                max_len=max_len,
            ),
        )

        df.loc[:, 'tokens'] = df.tokens.progress_apply(replace_numbers)

        # Remove sequences with only <UNK>
        df = df[df.tokens.progress_apply(
            lambda tokens: any(token != '<UNK>' for token in tokens),
        )]

        # Build vocab
        vocab = sorted(set(
            token for doc in list(df.tokens) for token in doc
        ))
        self.token2idx = {token: idx for idx, token in enumerate(vocab)}
        self.idx2token = {idx: token for token, idx in self.token2idx.items()}

        # Convert tokens to indexes
        df['indexed_tokens'] = df.tokens.progress_apply(
            lambda doc: [self.token2idx[token] for token in doc],
        )

        # Build BoW vector
        df['bow_vector'] = df.indexed_tokens.progress_apply(
            build_bow_vector, args=(self.idx2token,)
        )

        # Build TF-IDF vector
        vectorizer = TfidfVectorizer(
            analyzer='word',
            tokenizer=lambda doc: doc,
            preprocessor=lambda doc: doc,
            token_pattern=None,
        )
        vectors = vectorizer.fit_transform(df.tokens).toarray()
        df['tfidf_vector'] = [vector.tolist() for vector in vectors]

        # self.text = df.review.tolist()
        self.sequences = df.indexed_tokens.tolist()
        self.bow_vector = df.bow_vector.tolist()
        self.tfidf_vector = df.tfidf_vector.tolist()
        self.label_arr = np.asarray(df.iloc[:, 1:6]).tolist()
<<<<<<< HEAD
        #self.targets = df.label.tolist()
        #print(self.tfidf_vector[:5])
=======
        # self.targets = df.label.tolist()
        # print(self.tfidf_vector[:5])
>>>>>>> da5de8aaeb66abdbb2af19c3efed604ac65990ea

    def __len__(self):
        return len(self.sequences)
        
        
    def __getitem__(self, i):
        try:
            single_image_name = self.image_arr[i]  # TODO: need to edit
            img_as_img = Image.open(os.path.join(self.path_to_dataset, single_image_name))
<<<<<<< HEAD
            # img_as_img = transforms.Resize((112,112))(img_as_img)
            # img_as_tensor = self.to_tensor(img_as_img)
            img_as_tensor = self.transforms(img_as_img)
            if img_as_tensor.shape[0] == 1:
                img_as_tensor=einops.repeat(img_as_tensor, 'c h w -> (repeat c) h w', repeat=3)
            
=======
            # img_as_img = transforms.Resize((112, 112))(img_as_img)
            # img_as_tensor = self.to_tensor(img_as_img)
            img_as_tensor = self.transforms(img_as_img)
            if img_as_tensor.shape[0] == 1:
                img_as_tensor = einops.repeat(img_as_tensor, 'c h w -> (repeat c) h w', repeat=3)

>>>>>>> da5de8aaeb66abdbb2af19c3efed604ac65990ea
            return (
                img_as_tensor,
                torch.Tensor(self.sequences[i]),
                torch.Tensor(self.bow_vector[i]),
                torch.Tensor(self.tfidf_vector[i]),
<<<<<<< HEAD
                [self.label_arr[i][0]], #only misogyny 
                ' '.join(self.text[i].split()[:76]),
=======
                [self.label_arr[i][0]],  # only misogyny
                self.text[i],
>>>>>>> da5de8aaeb66abdbb2af19c3efed604ac65990ea
                single_image_name
            )
        except Exception as e:
            #pdb.set_trace()
            print(e)
            exit(1)


<<<<<<< HEAD
class ReduceMAMIDataset(Dataset):
    def __init__(self,max_len,max_vocab,transform_apply=False, path_to_dataset='./Data/TRAINING', split='training', transform=None):
        self.path_to_dataset=path_to_dataset
        
        self.transforms=transform

        self.to_tensor = transforms.ToTensor()
        df = data = pd.read_csv(os.path.join(path_to_dataset,f'{split}_split.csv'),sep="\t")
        df=data=data[data.apply(lambda x: os.path.exists(os.path.join(path_to_dataset,x['file_name'])), axis=1)]
        self.image_arr = np.asarray(data.iloc[:, 0]).tolist()
        self.text =  np.asarray(data.iloc[:, 6]).tolist()
        
    def __len__(self):
        return len(self.text)
    
    def __getitem__(self, i):
        single_image_name = self.image_arr[i] #TODO: need to edit
        return (
            os.path.join(self.path_to_dataset, single_image_name),
            single_image_name,
            ' '.join(self.text[i].split()[:76])
        )
def collate2(batch):
    img_ids = [item[1] for item in batch]
    img_paths = [item[0] for item in batch]
    text = [item[2] for item in batch]
    return img_ids, img_paths, text
=======

>>>>>>> da5de8aaeb66abdbb2af19c3efed604ac65990ea
def collate(batch):
    img = torch.stack([item[0] for item in batch], dim=0)
    seq = [item[1] for item in batch]
    bow = torch.stack([item[2] for item in batch], dim=0)
    tfidf = torch.stack([item[3] for item in batch], dim=0)
    target = torch.stack([torch.FloatTensor(item[4]) for item in batch], dim=0)
    text = [item[5] for item in batch]
    img_ids = [item[6] for item in batch]

<<<<<<< HEAD
    return img, text, img_ids

class COCODataset(Dataset):
    def __init__(self, path='/own_files/datasets/mscoco'):
        with open(f'/afs/cs.pitt.edu/usr0/arr159/erhan_code/t/t/data/COCO2017_train_capdata.pkl', 'rb') as f:
            df=pickle.load(f)
            
        self.ids=df['path']
        self.captions=df['capdata']
        self.img_paths=[self.get_path(path, img_pth)  for img_pth in df['path']]
    def __len__(self):
        return len(self.ids)
    def get_path(self, path, filename):
        return os.path.join(path, ('train2014' if 'train2014' in filename else 'val2014'), filename)
    def __getitem__(self, i):
        caption, _, _, _= random.choice(self.captions[i])# extract cap
        return (self.ids[i], caption, self.img_paths[i])
    @staticmethod
    def collate_fn(batch):
        ids=[data[0] for data in batch]
        img_paths=[data[2] for data in batch]
        captions=[data[1] for data in batch]
        return ids, img_paths, captions
=======
    return img, seq, bow, tfidf, target, text, img_ids
>>>>>>> da5de8aaeb66abdbb2af19c3efed604ac65990ea
