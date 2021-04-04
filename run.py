from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from src.utils import train_normal, train_adap, get_dataloader
from src.evaluation import score_negation, predict
import argparse

def getArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_name", type=str, default="tmills/roberta_sfda_sharpseed", help="Model Name")
    parser.add_argument("-trp", "--train_path", type=str, default="drive/My Drive/Team6/sfda/negation/practice_text/train.tsv", help="Train data path")
    parser.add_argument("-tep", "--test_path", type=str, default="drive/My Drive/Team6/sfda/negation/practice_text/dev.tsv", help="Test data path")
    parser.add_argument("-msd", "--model_save_dir", type=str, default="model/", help="Directory to save model")
    parser.add_argument("-ref", "--ref_pred", type=str, default="drive/My Drive/Team6/sfda/negation/practice_text/dev_labels.txt", help="Reference predictions")
    parser.add_argument("-res", "--res_pred", type=str, default="out.tsv", help="Result predictions")
    parser.add_argument("-a","--adaptive", type=bool, default = True, help="Adaptive Protoaug")
    parser.add_argument("-e","--extra_aug", type=int, default = 3, help="Extra Augmentation")
    parser.add_argument("-t","--threshold", type=float, default = 0.5, help="Threshold")
    parser.add_argument("-me","--max_epoch", type=int, default = 5, help="Max number of epochs")
    parser.add_argument("-bs","--batch_size", type=int, default = 16, help="Batch Size")
    parser.add_argument("-dp","--do_predict", type=bool, default = True, help="Make Predictions")
    parser.add_argument("-tr","--thresh_range", type=float, default = 0.2, help="Threshold Range")
    return parser


if __name__ == '__main__' :
    
    #argument parsing
    args = getArgs().parse_args()
    model_name = args.model_name
    train_path = args.train_path
    test_path = args.test_path
    model_save_dir = args.model_save_dir
    ref_pred = args.ref_pred
    res_pred = args.res_pred
    adaptive = args.adaptive
    extra_aug =  args.extra_aug
    threshold =  args.threshold
    max_epoch = args.max_epoch
    batch_size = args.batch_size
    do_predict = args.do_predict
    thresh_range = args.thresh_range
    
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
