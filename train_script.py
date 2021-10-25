import torch
from Dataset import MAMIDataset, collate
from params import *
from torch.utils.data import DataLoader
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_dataset = MAMIDataset(MAX_LEN,MAX_VOCAB)
#val_dataset = MAMIDataset(MAX_LEN,MAX_VOCAB, val_set=True)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=collate)


for img,seq,bow,tfidf,target,text,img_id in train_loader:
    print(img_id)
    print(text)
    #print(bow)
    break