from utils import get_top_points_idx
from processor import NegationDataset
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
import torch
from torch.utils.data import DataLoader,TensorDataset
from utils import train
from evaluation import score_negation, predict

if __name__ == '__main__' :
    
    # model_name = "tmills/roberta_sfda_sharpseed"
    model_name = "model/"
    train_dir = "drive/My Drive/Team6/sfda/negation/practice_text/train.tsv"

    test_dir = "drive/My Drive/Team6/sfda/negation/practice_text/dev.tsv"

    # idx, y_true_ = get_top_points_idx(model_dir= model_name, dataset_dir= train_dir)
    
    config = AutoConfig.from_pretrained(model_name,output_hidden_states=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                            config=config)


    # new_train_dataset,y_true_= NegationDataset.from_tsv(train_dir, tokenizer,1,idx.tolist(),y_true_,3)
    
    
    # y_true_ = torch.tensor(y_true_)
    # y_true_ = y_true_.type(torch.cuda.LongTensor)
    
    model = AutoModelForSequenceClassification.from_pretrained(model_name,
                                                                config=config)
    batch_size=16 # 32
  
    model = model.cuda()
    for k,v in model.named_parameters():
        x = k.split('.')
        if x[0]=='classifier':
            v.requires_grad=False

    # new_input_ids = []
    # new_attention_masks = []

    # for feat in new_train_dataset.features:
    #     new_input_ids.append(feat.input_ids)
    #     new_attention_masks.append(feat.attention_mask)

    # new_input_ids = torch.tensor(new_input_ids)
    # new_attention_masks = torch.tensor(new_attention_masks)
    # dataset = TensorDataset(new_input_ids, new_attention_masks,y_true_)
    # train_dataloader = DataLoader(
    #           dataset,  # The training samples.
    #           # sampler = RandomSampler(dataset), # Select batches randomly        ### was commented for 0.8759
    #           shuffle = True,  ### not here for 0.8759
    #           batch_size = batch_size # Trains with this batch size.
    #       )

    # train(model,train_dataloader,train_min_step = 250)
    # model.save_pretrained('model/')
    # tokenizer.save_pretrained('model/')

    predict(model,test_dir, "out.tsv")
    score_negation( "drive/My Drive/Team6/sfda/negation/practice_text/dev_labels.txt", "out.tsv")



    

    