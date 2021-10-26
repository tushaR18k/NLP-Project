import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau

from Dataset import MAMIDataset, collate
from params import *
from tqdm import tqdm
from models import LogisticRegression
from utils import accuracy
import pdb
import os

exp_name='log_res'
os.makedirs(f'results/{exp_name}', exist_ok=True)

def train_epoch(dataloader, model, optimizer, scheduler, epoch):
    try:
        pb=tqdm(dataloader)
        acc_loss=0
        accuracy_list=[]
        count=1
        for img,seq,bow,tfidf,target,text,img_id in pb:
            bow=bow.to(device)
            target=target.to(device)
            outputs=model(bow)
            #pdb.set_trace()
            loss=model.calculate_loss(outputs, target)
            loss.backward()
            acc_loss+=loss

            #batch_top1 = accuracy(outputs, target, topk=[1])[0]
            #accuracy_list.append(batch_top1)
            scheduler.step(loss)
            optimizer.step()
            model.zero_grad()
            pb.set_description(f'Loss: {acc_loss/count}')
            count+=1
    except Exception as e:
        pdb.set_trace()
        
    torch.save({'model_state':model.state_dict(),
                'optim_state':optimizer.state_dict(),
                'scheduler_state':scheduler.state_dict()}, f'results/{exp_name}/checkpoint.pth')
    return acc_loss/len(dataloader)#, sum(accuracy_list)/len(accuracy_list)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# initialize datasets
train_dataset = MAMIDataset(MAX_LEN,MAX_VOCAB, split='train')
val_dataset = MAMIDataset(MAX_LEN,MAX_VOCAB, split='val')
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=collate, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=collate)
#pdb.set_trace()

# initialize model and optimizers
model=LogisticRegression(in_dim=train_dataset[0][2].shape[-1], out_dim=1)
model.to(device)
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, nesterov=True)
scheduler=ReduceLROnPlateau(optimizer, 'min')

# train and save model
for epoch in range(MAX_EPOCHS):
    loss = train_epoch(train_loader, model, optimizer, scheduler, epoch)
    
