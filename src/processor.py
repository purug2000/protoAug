import torch
import numpy as np
from torch.utils.data.dataset import Dataset
from transformers.data.processors.utils import InputExample, InputFeatures
from transformers.data.processors.glue import glue_convert_examples_to_features
from transformers.data.processors.utils import DataProcessor
import re
import random
labels = ["-1", "1"]
max_length = 128
def text_augmentation(lines,y_true_,extra_size_ratio=5):
  lt = 6
  rt = 3
  y_true_= y_true_.unsqueeze(1)
  p = re.compile('<e>.*</e>')
  concept_terms = []
  for line in lines:
    line=line[0]
    concept_terms.append(re.findall('<e>.*</e>', line)[0])
  sz = len(lines)
  for i in range(int(extra_size_ratio*sz)):
    idx = random.choice(range(sz))
    line = lines[idx][0]
    n=len(line)
    ## taking only a part of sentence that contains the marked term with max 6 words on left and 3 word on right
    m=p.search(line)
    (start,end) = m.span()

    ## --------
    ## replacing concept term with some other concept term
    idx2 = random.choice(range(len(concept_terms)))
    new_line2 = line[:start] + concept_terms[idx2] + line[end:]
    lines.append([new_line2])
    # lines.append([line])
    # print(torch.tensor([y_true_[idx].cpu().numpy()]))
    y_true_ = torch.vstack((y_true_,torch.tensor(y_true_[idx].cpu().numpy()).cuda()))
    ## -------
    # print("original line : ",line)
    # print("new first line : ",new_line)
    # print("new second line : ",new_line2)
  y_true_ = y_true_.squeeze(1)
  return lines,y_true_

class NegationDataset(Dataset):
    def __init__(self, features):
        self.features = features
        self.label_list = ["-1", "1"]

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i) -> InputFeatures:
        return self.features[i]

    def get_labels(self):
        return self.label_list


    @classmethod
    def from_tsv(cls, tsv_file, tokenizer,flag=0,idx = [],y_true_ = [],extra_ratio=0):
        """Creates examples for the test set."""
        lines = DataProcessor._read_tsv(tsv_file)
        # print(len(lines))
        pos_lines = []
        neg_lines = []
        extra_lines  = 0
        pos_n = 0
        neg_n = 0
        if idx!=[]:
          lines = [line for (i,line) in enumerate(lines) if i in idx]
          print(len(lines))
          pos_lines = [line for (i,line) in enumerate(lines) if y_true_[i]==1]
          neg_lines = [line for (i,line) in enumerate(lines) if y_true_[i]==0]
          pos_n = len(pos_lines)
          neg_n = len(neg_lines)
          extra_pos_ratio = ((len(neg_lines)/len(pos_lines))*(1+extra_ratio)-1)
          extra_neg_ratio = extra_ratio
          # print(extra_pos_ratio)

        # print(lines)
        if flag==1:
          pos_y_true_ = torch.ones(pos_n).cuda()
          neg_y_true_ = torch.zeros(neg_n).cuda()
          pos_lines,pos_y_true_ = text_augmentation(pos_lines,pos_y_true_,extra_pos_ratio)
          neg_lines,_ = text_augmentation(neg_lines,neg_y_true_,extra_neg_ratio)
          lines = []
          print(len(pos_y_true_))
          y_true_ = pos_y_true_.unsqueeze(1)
          for line in pos_lines:
            lines.append(line)
          for line in neg_lines:
            lines.append(line)
            y_true_ = torch.vstack((y_true_,torch.tensor(0).cuda()))
          
          y_true_ = y_true_.squeeze(1)
            
        examples = []
        for (i, line) in enumerate(lines):
            # if idx==None or i in idx:
            guid = 'instance-%d' % i
            if line[0] in labels:
                text_a = '\t'.join(line[1:])
            else:
                text_a = '\t'.join(line)
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label = None))

        features = glue_convert_examples_to_features(
            examples,
            tokenizer,
            max_length=max_length,
            label_list=labels,
            output_mode='classification',
        )
        return cls(features),y_true_
