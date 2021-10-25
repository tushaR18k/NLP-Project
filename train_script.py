import torch
from Dataset import MAMIDataset, collate
from params import *
from torch.utils.data import DataLoader
from tqdm import tqdm

def train_epoch(dataloader, model, optim, scheduler, epoch):
    pb=tqdm(dataloader)
    for img,seq,bow,tfidf,target,text,img_id in pb:
        print(img_id)
        print(text)
        #print(bow)
        break 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# initialize datasets
train_dataset = MAMIDataset(MAX_LEN,MAX_VOCAB, split='train')
val_dataset = MAMIDataset(MAX_LEN,MAX_VOCAB, split='val')
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=collate, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=collate)

# initialize model and optimizers

# train and save model
'''
for epoch in range(MAX_EPOCHS)
    loss = train_epoch(train_loader, model, optim, scheduler, epoch)
'''
    

