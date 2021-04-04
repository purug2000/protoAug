from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from src.utils import train_normal, train_adap, get_dataloader
from src.evaluation import score_negation, predict

if __name__ == '__main__' :
    

    ## Hyperparameters #######
    model_name = "tmills/roberta_sfda_sharpseed"
    train_path = "drive/My Drive/Team6/sfda/negation/practice_text/train.tsv"
    test_path = "drive/My Drive/Team6/sfda/negation/practice_text/dev.tsv"
    model_save_dir = "model/"
    ref_pred = "drive/My Drive/Team6/sfda/negation/practice_text/dev_labels.txt"
    res_pred = "out.tsv"
    adaptive = True
    extra_aug =  3
    threshold =  0.5
    max_epoch = 5
    batch_size = 16
    do_predict = True
    thresh_range = 0.2
    ##########################

    
    config = AutoConfig.from_pretrained(model_name,output_hidden_states=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                            config=config)
    model = AutoModelForSequenceClassification.from_pretrained(model_name,
                                                                config=config)
    model = model.cuda()
    for k,v in model.named_parameters():
        x = k.split('.')
        if x[0]=='classifier':
            v.requires_grad=False

    if adaptive:
        train_adap(model,tokenizer,train_path, threshold, extra_aug, max_epoch,thresh_range)
    else:
        train_dataloader = get_dataloader(model, tokenizer,train_path,threshold, extra_aug,batch_size)
        min_steps = max_epoch*len(train_dataloader)
        train_normal(model, train_dataloader, min_steps)
    
    
    model.save_pretrained(model_save_dir)
    tokenizer.save_pretrained(model_save_dir)

    if do_predict:
        predict(model,tokenizer,test_path, res_pred)
        # # "drive/My Drive/Team6/sfda/negation/practice_text/dev_labels.txt"
        score_negation( ref_pred, res_pred)



    

    