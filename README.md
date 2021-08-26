# PyTorch Implementation of ProtoAUG and Adapt-ProtoAUG
This repository provides a [PyTorch](https://pytorch.org/) implementation of the *ProtoAUG* and *Adapt-ProtoAUG* methods presented in our
SemEval 2021 paper ”IITK at SemEval-2021 Task 10: Source-Free Unsupervised DomainAdaptation using Class Prototypes”.


## Citation and Contact
You can find the paper at 
[https://aclanthology.org/2021.semeval-1.53/](https://aclanthology.org/2021.semeval-1.53/).

If you use our work, please also cite the paper:
```
@inproceedings{kumar-etal-2021-iitk,
    title = "{IITK} at {S}em{E}val-2021 Task 10: Source-Free Unsupervised Domain Adaptation using Class Prototypes",
    author = "Kumar, Harshit  and
      Shah, Jinang  and
      Hegde, Nidhi  and
      Gupta, Priyanshu  and
      Jindal, Vaibhav  and
      Modi, Ashutosh",
    booktitle = "Proceedings of the 15th International Workshop on Semantic Evaluation (SemEval-2021)",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.semeval-1.53",
    doi = "10.18653/v1/2021.semeval-1.53",
    pages = "438--444",
    abstract = "Recent progress in deep learning has primarily been fueled by the availability of large amounts of annotated data that is obtained from highly expensive manual annotating pro-cesses. To tackle this issue of availability of annotated data, a lot of research has been done on unsupervised domain adaptation that tries to generate systems for an unlabelled target domain data, given labeled source domain data. However, the availability of annotated or labelled source domain dataset can{'}t always be guaranteed because of data-privacy issues. This is especially the case with medical data, as it may contain sensitive information of the patients. Source-free domain adaptation (SFDA) aims to resolve this issue by us-ing models trained on the source data instead of using the original annotated source data. In this work, we try to build SFDA systems for semantic processing by specifically focusing on the negation detection subtask of the SemEval2021 Task 10. We propose two approaches -ProtoAUGandAdapt-ProtoAUGthat use the idea of self-entropy to choose reliable and high confidence samples, which are then used for data augmentation and subsequent training of the models. Our methods report an improvement of up to 7{\%} in F1 score over the baseline for the Negation Detection subtask.",
}
```
<!-- 
If you would like to get in touch, please contact [contact@lukasruff.com](mailto:contact@lukasruff.com). -->


## Abstract
>> Recent progress in deep learning has primarily been fueled by the availability of large amounts of annotated
>> data that is obtained from highly expensive manual annotating processes. To tackle this issue of availability 
>> of annotated data, a lot of research has been done on unsupervised domain adaptation that tries to generate systems
>> for an unlabelled target domain data, given labelled source domain data. However, the availability of annotated or 
>> labelled source domain dataset can't always be guaranteed because of data-privacy issues. This is especially the 
>> case with medical data, as it may contain sensitive information of the patients. Source-free domain adaptation (SFDA) 
>> aims to resolve this issue by using models trained on the source data instead of using the original annotated source 
>> data. In this work, we try to build SFDA systems for semantic processing by specifically focusing on the negation detection 
>> subtask of the SemEval 2021 Task 10. We propose two approaches - *ProtoAUG* and *Adapt-ProtoAUG* that use 
>> the idea of self-entropy to choose reliable and high confidence samples, which are then used for data augmentation and 
>> subsequent training of the models. Our methods report an improvement of up to 7\% in F1 score over the baseline for the Negation Detection subtask.


## Installation
This code is written in `Python 3.7` and requires the packages listed in `requirements.txt`.

Clone the repository to your local machine and directory of choice:
```
git clone https://github.com/purug2000/protoAug.git
```
Install the required libraries/packages using:
```
pip install -r requirements.txt
```

## Dataset Preparation

Since the competition uses sensitive medical data, the data used for this project is not easily available. Please use the 
following links to obtain the required data:

* Competetition Codalab Page ([https://competitions.codalab.org/competitions/26152](https://competitions.codalab.org/competitions/26152))
* Organisers' github repo ([https://github.com/Machine-Learning-for-Medical-Language/source-free-domain-adaptation](https://github.com/Machine-Learning-for-Medical-Language/source-free-domain-adaptation))

## Running the experiments
To get help regarding any of the parameters, use:
```
python3 run.py -h
```

### Running ProtoAUG
To run *ProtoAUG*, set --adaptive to False:
```
python3 run.py --adaptive False --batch_size 16 --do_predict True --extra_aug 3 --max_epoch 5 --model_name "tmills/roberta_sfda_sharpseed" --model_save_dir "model/" --ref_pred "drive/My Drive/Team6/sfda/negation/practice_text/dev_labels.txt" --res_pred "out.tsv" --test_path "drive/My Drive/Team6/sfda/negation/practice_text/dev.tsv" --threshold 0.5 --train_path "drive/My Drive/Team6/sfda/negation/practice_text/train.tsv" 
```

### Running Adapt-ProtoAUG
To run *Adapt-ProtoAUG*, set --adaptive to True:
```
python3 run.py --adaptive True --batch_size 16 --do_predict True --extra_aug 3 --max_epoch 5 --model_name "tmills/roberta_sfda_sharpseed" --model_save_dir "model/" --ref_pred "drive/My Drive/Team6/sfda/negation/practice_text/dev_labels.txt" --res_pred "out.tsv" --test_path "drive/My Drive/Team6/sfda/negation/practice_text/dev.tsv" --thresh_range 0.2 --threshold 0.5 --train_path "drive/My Drive/Team6/sfda/negation/practice_text/train.tsv" 
```
