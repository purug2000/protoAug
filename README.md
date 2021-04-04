# PyTorch Implementation of ProtoAUG and Adapt-ProtoAUG
This repository provides a [PyTorch](https://pytorch.org/) implementation of the *ProtoAUG* and *Adapt-ProtoAUG* methods presented in our
SemEval 2021 paper ”IITK at SemEval-2021 Task 10: Source-Free Unsupervised DomainAdaptation using Class Prototypes”.


## Citation and Contact
You can find a PDF of the paper at 
[http://proceedings.mlr.press/v80/ruff18a.html](http://proceedings.mlr.press/v80/ruff18a.html).

If you use our work, please also cite the paper:
```
@InProceedings{pmlr-v80-ruff18a,
  title     = {Deep One-Class Classification},
  author    = {Ruff, Lukas and Vandermeulen, Robert A. and G{\"o}rnitz, Nico and Deecke, Lucas and Siddiqui, Shoaib A. and Binder, Alexander and M{\"u}ller, Emmanuel and Kloft, Marius},
  booktitle = {Proceedings of the 35th International Conference on Machine Learning},
  pages     = {4393--4402},
  year      = {2018},
  volume    = {80},
}
```

If you would like to get in touch, please contact [contact@lukasruff.com](mailto:contact@lukasruff.com).


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
git clone https://github.com/lukasruff/Deep-SVDD-PyTorch.git
```

To run the code, we recommend setting up a virtual environment, e.g. using `virtualenv` or `conda`:

### `virtualenv`
```
# pip install virtualenv
cd <path-to-Deep-SVDD-PyTorch-directory>
virtualenv myenv
source myenv/bin/activate
pip install -r requirements.txt
```

### `conda`
```
cd <path-to-Deep-SVDD-PyTorch-directory>
conda create --name myenv
source activate myenv
while read requirement; do conda install -n myenv --yes $requirement; done < requirements.txt
```


## Running ProtoAUG

We currently have implemented the MNIST ([http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/)) and 
CIFAR-10 ([https://www.cs.toronto.edu/~kriz/cifar.html](https://www.cs.toronto.edu/~kriz/cifar.html)) datasets and 
simple LeNet-type networks.

Have a look into `main.py` for all possible arguments and options.

## Running Adapt-ProtoAUG
```
cd <path-to-Deep-SVDD-PyTorch-directory>

# activate virtual environment
source myenv/bin/activate  # or 'source activate myenv' for conda

# create folder for experimental output
mkdir log/mnist_test

# change to source directory
cd src

# run experiment
python main.py mnist mnist_LeNet ../log/mnist_test ../data --objective one-class --lr 0.0001 --n_epochs 150 --lr_milestone 50 --batch_size 200 --weight_decay 0.5e-6 --pretrain True --ae_lr 0.0001 --ae_n_epochs 150 --ae_lr_milestone 50 --ae_batch_size 200 --ae_weight_decay 0.5e-3 --normal_class 3;
```
This example trains a One-Class Deep SVDD model where digit 3 (`--normal_class 3`) is considered to be the normal class. Autoencoder
pretraining is used for parameter initialization.
