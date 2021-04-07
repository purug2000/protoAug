import re
import torch
import random 
import numpy as np
import torch.nn.functional as F
from .processor import NegationDataset
import math
import time
import torch.optim as optim
from  tqdm import tqdm
import torch
import time
import numpy as np
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from torch.utils.data import DataLoader,TensorDataset



def get_entropy(train_dataset_path ,model, tokenizer):
    # for k,v in model.named_parameters():
    #     v.requires_grad = False
    with torch.no_grad():
      st = time.time()
      input_ids = []
      attention_masks = []
      output_label = []
      train_dataset,_ = NegationDataset.from_tsv(train_dataset_path, tokenizer)
      y_true = torch.tensor([]).cuda()
      entropyl = torch.tensor([]).cuda()
      m=torch.nn.Softmax(dim=1)
      cnt=0
      for feat in train_dataset.features:
          input_ids.append(feat.input_ids)
          attention_masks.append(feat.attention_mask)
          out = model(torch.tensor([feat.input_ids]).cuda(),torch.tensor([feat.attention_mask]).cuda())

          pred = m(out[0])
          entropy = torch.sum(- pred * torch.log(pred), dim=1, keepdim=True)
          entropy_norm = entropy / np.log(pred.size(1))
          entropy_norm = entropy_norm.squeeze(1)
          entropyl = torch.cat((entropyl,entropy_norm),0)
          z = torch.argmax(out[0],dim=1)
          z = z.type(torch.cuda.LongTensor)
          y_true = torch.cat((y_true,z),0)
          cnt+=1
          # print(y_true)

      ed = time.time()
      print("time taken - ",ed-st)
    return  entropyl, y_true



def get_top_points_idx(model, tokenizer,dataset_dir, quantile = 0.5):
    entropyl, y_true =  get_entropy(dataset_dir, model,tokenizer)
    print(entropyl.shape)
    pos_entropyl = torch.sort(entropyl[y_true==1])
    # print(pos_entropyl)
    threshold = torch.quantile(pos_entropyl.values,torch.tensor([quantile]).cuda())
    
    mask = (((entropyl<threshold) & (y_true==1)) | ((entropyl<threshold) & (y_true==0)))

    idx = torch.where(mask==True)[0]

    return idx,y_true[idx]

def get_dataloader(model, tokenizer,dataset_dir, threshold = 0.5, extra_aug = 3,batch_size = 16):
    idx, y_true_ = get_top_points_idx(model = model, dataset_dir= dataset_dir, tokenizer=tokenizer, quantile= threshold)
    new_train_dataset,y_true_= NegationDataset.from_tsv(dataset_dir, tokenizer,1,idx.tolist(),y_true_,extra_aug)
    
    
    y_true_ = torch.tensor(y_true_)
    y_true_ = y_true_.type(torch.cuda.LongTensor)
    
    
    # batch_size=16 # 32
  
    

    new_input_ids = []
    new_attention_masks = []

    for feat in new_train_dataset.features:
        new_input_ids.append(feat.input_ids)
        new_attention_masks.append(feat.attention_mask)

    new_input_ids = torch.tensor(new_input_ids)
    new_attention_masks = torch.tensor(new_attention_masks)
    dataset = TensorDataset(new_input_ids, new_attention_masks,y_true_)
    train_dataloader = DataLoader(
              dataset,  # The training samples.
              # sampler = RandomSampler(dataset), # Select batches randomly        ### was commented for 0.8759
              shuffle = True,  ### not here for 0.8759
              batch_size = batch_size # Trains with this batch size.
          )
    return train_dataloader

def op_copy(optimizer):
    '''
    This function has been copied from https://github.com/tim-learn/SHOT/blob/master/object/image_target.py 
    '''
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer

def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    '''
    This function has been copied from https://github.com/tim-learn/SHOT/blob/master/object/image_target.py 
    '''
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    # decay = 1/math.sqrt(iter_num)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] =0.0005
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer

def tensor_l2normalization(q):
	'''
	This function has been copied from https://github.com/youngryan1993/PrDA-Progressive-Domain-Adaptation-from-a-Source-Pre-trained-Model/blob/master/lib.py
	'''
    qn = torch.norm(q, p=2, dim=1).detach().unsqueeze(1)
    q = q.div(qn.expand_as(q))
    return q
    
def train_normal(model, train_dataloader, train_min_step = 3204, train_lr=0.0005, 
        train_weight_decay = 0.0005, train_momentum = 0.9, train_update_freq = 96) :
    optimizer = optim.SGD(list(filter(lambda p: p.requires_grad, model.parameters())), 
                        lr=train_lr, weight_decay=train_weight_decay, momentum=train_momentum, nesterov=True)
    optimizer = op_copy(optimizer)
    global_step = 0
    epoch_id = 0
    max_epoch = math.ceil((train_min_step)/len(train_dataloader))
    
    model.zero_grad()
    while global_step < train_min_step:
        epoch_id += 1
        max_iter = max_epoch * len(train_dataloader)
        # print(max_iter)
        epoch_start_time=time.time()

        for i, dt in tqdm(enumerate(train_dataloader)):
        
            input_id = dt[0].cuda()
            attention_mask = dt[1].cuda()
            pseudo_label = dt[2].cuda()
            
            ## trainable target model
            model.train()
            out_t = model(input_id,attention_mask=attention_mask)
            
            logit_t = out_t[0]
            
            ce_from_s2t = F.cross_entropy(logit_t,pseudo_label)
            ce_total = ce_from_s2t
            
            if global_step%2==0:
                batch_loss = ce_total
            else:
                batch_loss+=ce_total
            global_step+=1 
            if global_step%2==0:
                optimizer.zero_grad()
                batch_loss /= 2
                batch_loss.backward(retain_graph=True)
                optimizer.step()
                lr_scheduler(optimizer, iter_num=global_step, max_iter=max_iter)

def train_adap(model, tokenizer, train_dir,  thresh = 0.6,extra_aug = 1, max_epoch = 5, thresh_range = 0.2, train_lr=0.0005, 
        train_weight_decay = 0.0005, train_momentum = 0.9, train_update_freq = 96):
    optimizer = optim.SGD(list(filter(lambda p: p.requires_grad, model.parameters())),     
                        lr=train_lr, weight_decay=train_weight_decay, momentum=train_momentum, nesterov=True)

    optimizer = op_copy(optimizer)
    epoch_id = 0
    model.zero_grad()
    global_step = 0

    while epoch_id < max_epoch:
        thresh = 0.5+(thresh_range*epoch_id/max_epoch)
        
        epoch_id += 1

        model.eval()
        train_dataloader = get_dataloader(model, tokenizer,train_dir, thresh, extra_aug)
        max_iter = max_epoch * len(train_dataloader)
        epoch_start_time=time.time()
        for i, dt in tqdm(enumerate(train_dataloader)):
            input_id = dt[0].cuda()
            attention_mask = dt[1].cuda()
            pseudo_label = dt[2].cuda()
            ## trainable target model
            model.train()
            out_t = model(input_id,attention_mask=attention_mask)
            logit_t = out_t[0]
            loss = F.cross_entropy(logit_t,pseudo_label)# + nn.CrossEntropyLoss()(logit_t,pseudo_label_t))
            
            
            if global_step%2==0:
                batch_loss = loss
            else:
                batch_loss+=loss
            global_step+=1 
            if global_step%2==0:
                optimizer.zero_grad()
                batch_loss /= 2
                batch_loss.backward(retain_graph=True)
                optimizer.step()
                lr_scheduler(optimizer, iter_num=global_step, max_iter=max_iter)
        epoch_end_time=time.time()
        print("time taken for epoch no. {} is {}".format(epoch_id,epoch_end_time-epoch_start_time))
   

