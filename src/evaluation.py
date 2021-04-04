import torch
import logging

# import NegationDataset
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn.metrics import f1_score,precision_score,recall_score,confusion_matrix
from tabulate import tabulate
from .processor import NegationDataset
import numpy as np
logger = logging.getLogger(__name__)

def load_testdata(file_name,tokenizer):
    test_dataset,_ = NegationDataset.from_tsv(file_name, tokenizer)

    input_ids = []
    attention_masks = []
    for feat in test_dataset.features:
        input_ids.append(feat.input_ids)
        attention_masks.append(feat.attention_mask)

    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)
    # print(input_ids.shape,attention_masks.shape)

    dataset = TensorDataset(input_ids, attention_masks,torch.arange(input_ids.size(0)))
    
    batch_size=32 # 32
    test_dataloader = DataLoader(dataset, batch_size = batch_size)
    # print(batch_size)
    return test_dataloader


def predict(target_trainable_net,tokenizer,test_file,output_test_file,threshold = 0.5):
  # target_trainable_net = 
  # tokenizer =
  start=True
  test_dataloader = load_testdata(test_file,tokenizer)
  for data in test_dataloader:
    # out = fixed_source_net(data[0].cuda(),data[1].cuda())
    out = target_trainable_net(data[0].cuda(),data[1].cuda())
    if start:
        predict = out[0].cpu().detach().numpy()
        start = False
    else:
        predict = np.concatenate((predict,out[0].cpu().detach().numpy()))
  m=torch.nn.Softmax(dim=1)
  p = m(torch.tensor(predict))
  p1 = np.ones(p.shape[0],np.int)
  for i in range(p.shape[0]):
    prob = p[i][1]
    if prob>threshold:
      p1[i] = 1
    else:
      p1[i] = -1
  # predict = np.argmax(predict,1)
  with open(output_test_file, "w") as writer:
    logger.info("***** Test results *****")
    for index, item in enumerate(p1):
        writer.write("%s\n" % p1[index])

  return p1

def read_tsv(file):
    output = []
    with open(file, 'r') as f_output:
        for record in f_output:
            output.append(int(record))
    return output

def score_negation(ref_domain,res_domain):
    ref = read_tsv(ref_domain)
    res = read_tsv(res_domain)
    assert len(ref) == len(res)
    trainable_f1_score = f1_score(ref,res)
    trainable_prec = precision_score(ref,res)
    trainable_recall = recall_score(ref,res)
    scores = [['trained',trainable_f1_score,trainable_prec,trainable_recall]]
    print(tabulate(scores,headers=['model','f1 score','precision','recall']))
    return trainable_f1_score,trainable_prec,trainable_recall

