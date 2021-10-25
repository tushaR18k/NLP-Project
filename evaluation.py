import torch
from models import LogisticRegression
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR,ReduceLROnPlateau
import torch.nn as nn

from Dataset import MAMIDataset, collate
from params import *
from tqdm import tqdm
from models import LogisticRegression
from utils import accuracy
import pdb
import os

val_dataset = MAMIDataset(MAX_LEN,MAX_VOCAB, split='val')
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=collate)

path_to_model='./results/log_res/checkpoint.pth'
model=LogisticRegression(in_dim=val_dataset[0][2].shape[-1], out_dim=1)
model.load_state_dict(torch.load(path_to_model)['model_state'])

#perform inference
predicted_labels=[round(i.detach().numpy()[0]) for i in nn.Sigmoid()(model(torch.Tensor(val_dataset.bow_vector)))]
ground_truth_labels=torch.Tensor([labels[0] for labels in val_dataset.label_arr])

from sklearn.metrics import f1_score
print(f1_score(ground_truth_labels, predicted_labels))
