from IPython.display import clear_output, Image, display
import PIL.Image
import io
import json
import torch
import numpy as np
from _lxmert.processing_image import Preprocess
from _lxmert.visualizing_image import SingleImageViz
from _lxmert.modeling_frcnn import GeneralizedRCNN
from _lxmert._utils import Config
import _lxmert._utils
from transformers import LxmertForQuestionAnswering, LxmertTokenizer
import wget
import pickle
import os
from transformers import LxmertForPreTraining
import importlib 
import Dataset
importlib.reload(Dataset)
from Dataset import ReduceMAMIDataset, collate2, COCODataset

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE=2
DATASET_TO_USE='coco'
def showarray(a, fmt="jpeg"):
    a = np.uint8(np.clip(a, 0, 255))
    f = io.BytesIO()
    PIL.Image.fromarray(a).save(f, fmt)
    display(Image(data=f.getvalue()))
    
lxmert_base = LxmertForPreTraining.from_pretrained("unc-nlp/lxmert-base-uncased").to(device)
# load models and model components
frcnn_cfg = Config.from_pretrained("unc-nlp/frcnn-vg-finetuned")

frcnn = GeneralizedRCNN.from_pretrained("unc-nlp/frcnn-vg-finetuned", config=frcnn_cfg)

image_preprocess = Preprocess(frcnn_cfg)

lxmert_tokenizer = LxmertTokenizer.from_pretrained("unc-nlp/lxmert-base-uncased")

import _lxmert
OBJ_URL = "https://raw.githubusercontent.com/airsplay/py-bottom-up-attention/master/demo/data/genome/1600-400-20/objects_vocab.txt"
objids = _lxmert._utils.get_data(OBJ_URL)

import torch.nn as nn
import random

def get_object_detection_output(img_paths):
    images, sizes, scales_yx = image_preprocess(img_paths)
    output_dict = frcnn(
        images,
        sizes,
        scales_yx=scales_yx,
        padding="max_detections",
        max_detections=frcnn_cfg.max_detections,
        return_tensors="pt",
    )
    return output_dict
def get_objects(output_dict):
    objects=[]
    for tmp in zip(output_dict.get("obj_ids"),output_dict.get("obj_probs")):
        objects.append([objids[i] for i, p in zip(tmp[0].tolist(), tmp[1].tolist()) if p > 0.5])
    return objects
def pretrained_model_fwd_pass(img_paths, txt):
    output_dict=get_object_detection_output(img_paths)
    objects=get_objects(output_dict)
    normalized_boxes = output_dict.get("normalized_boxes").to(device)
    features = output_dict.get("roi_features").to(device)
    inputs = lxmert_tokenizer(
        txt,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_token_type_ids=True,
        return_attention_mask=True,
        add_special_tokens=True,
        return_tensors="pt",
    ).to(device)
    cross_relationship_score= nn.Sigmoid()(lxmert_base(input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        visual_feats=features,
        visual_pos=normalized_boxes,
        token_type_ids=inputs.token_type_ids,
        output_attentions=False)['cross_relationship_score'])
    shuffled_ids=[1,0]#list(range(len(txt)))
    #random.shuffle(shuffled_ids)
    inputs = lxmert_tokenizer(
        [txt[_id] for _id in shuffled_ids],
        padding="max_length",
        max_length=77,
        truncation=True,
        return_token_type_ids=True,
        return_attention_mask=True,
        add_special_tokens=True,
        return_tensors="pt",
    ).to(device)
    actually_different=[shuffled_ids[_id] != _id for _id in list(range(len(txt)))]
    random_cross_relationship_score= nn.Sigmoid()(lxmert_base(input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        visual_feats=features,
        visual_pos=normalized_boxes,
        token_type_ids=inputs.token_type_ids,
        output_attentions=False)['cross_relationship_score'])
    return objects, cross_relationship_score.tolist(), random_cross_relationship_score.tolist(), actually_different

def save_as_df(dataloader, path):
    original_images = []
    images = []
    texts = []
    # plt.figure(figsize=(16, 5))
    sims=[]
    random_sims=[]
    import pandas as pd
    df=pd.DataFrame()
    img_ids=[]
    from tqdm import tqdm 
    underlying_shuffle_info=[]
    all_objects=[]
    total_counter=0
    for batch_img_id, batch_img_paths, batch_text in tqdm(dataloader):
        #breakpoint()
        total_counter+=1
        objects, cross_relationship_score, random_cross_relationship_score, actually_different = pretrained_model_fwd_pass(batch_img_paths, batch_text)
        img_ids.extend(batch_img_id)
        sims.extend(cross_relationship_score)
        random_sims.extend(random_cross_relationship_score)
        underlying_shuffle_info.extend(actually_different)
        all_objects.extend(objects)
        if total_counter % 500 == 0:
            df=pd.DataFrame()
            df['img_ids']=img_ids
            df['sim']=sims
            df['random_sims']=random_sims
            df['img_random_sims']=underlying_shuffle_info
            df['objects']=all_objects
            df.to_csv(path)
            if total_counter % 1000 == 0:
                return
    df=pd.DataFrame()
    df['img_ids']=img_ids

    df['sim']=sims
    df['random_sims']=random_sims
    df['img_random_sims']=underlying_shuffle_info
    df['objects']=all_objects
    df.to_csv(path)

from params import *
from torch.utils.data import DataLoader


if DATASET_TO_USE=='coco':
    # enter dataloading code
    train_dataset = COCODataset()
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=4, collate_fn=train_dataset.collate_fn, shuffle=True)
    save_as_df(train_loader, path='coco_lxmert.csv')
else:
    num_epochs = 30

    # transform = transforms.Compose([
    #     transforms.RandomHorizontalFlip(),
    #     transforms.Resize((256, 256)),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.485, 0.456, 0.406),
    #                           (0.229, 0.224, 0.225))])
    train_dataset = ReduceMAMIDataset(MAX_LEN, MAX_VOCAB, split='train', path_to_dataset='./Data/MASKED_TEXT_TRAINING', transform=None)
    val_dataset = ReduceMAMIDataset(MAX_LEN, MAX_VOCAB, split='val', path_to_dataset='./Data/MASKED_TEXT_TRAINING', transform=None)


    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=8, collate_fn=collate2, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=collate2)

    dataloader = {
        'train': train_loader,
        'val': val_loader
    }

    num_classes = 2


    # call training
    save_as_df(dataloader['train'], path='masked_df_mami_lxmert2.csv')

    from params import *
    train_dataset = ReduceMAMIDataset(MAX_LEN, MAX_VOCAB, split='train', path_to_dataset='./Data/TRAINING', transform=None)
    val_dataset = ReduceMAMIDataset(MAX_LEN, MAX_VOCAB, split='val', path_to_dataset='./Data/TRAINING', transform=None)

    from torch.utils.data import DataLoader

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=8, collate_fn=collate2, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=collate2)

    dataloader = {
        'train': train_loader,
        'val': val_loader
    }

    save_as_df(dataloader['train'], path='not_masked_df_mami_lxmert2.csv')

    # call training
