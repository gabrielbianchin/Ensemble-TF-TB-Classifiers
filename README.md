# Ensemble of Template-Free and Template-Based Classifiers for Protein Secondary Structure Prediction

## Introduction
Protein secondary structures are important inmany biological processes and applications. Due to the advances in the sequencing methods, there are many proteins sequenced, but fewer proteins with the secondary structures defined by laboratory methods. With the development of computer technology, computational methods started to become the most important methodology for predicting secondary structures. Driven by the recent results obtained by computational methods in this task, we evaluate two different approaches to this problem, which are template free classifiers, based on machine learning techniques, and template-based classifiers, based on searching tools. Each of them is formed by different sub classifiers, six for template-free and two for template-based, each with a specific view of the protein. Our results showed that these ensembles improve the results of each approach individually.

## Classifier
The protein secondary structure classifier presented in the paper is composed of an ensemble of template-free classifiers (bidirecional recurrent neural networks, random forests, Inception-v4 blocks, inception recurrent networks, BERT and convolutional neural networks), template-based classifiers (BLAST), and the ensemble between template-free and template-based classifiers. More details can be found in the [paper](https://www.mdpi.com/1422-0067/22/21/11449).

## Dataset
The dataset used in the paper can be found [here](https://www.princeton.edu/~jzthree/datasets/ICML2014/).

## Reproducibility
The code of template-free and template-based classifiers is available in the folders.

## Citation
This repository contains the source codes of Ensemble of Template-Free and Template-Based Classifiers for Protein Secondary Structure Prediction, as given in the paper:

Gabriel Bianchin de Oliveira, Helio Pedrini, Zanoni Dias. "Ensemble of Template-Free and Template-Based Classifiers for Protein Secondary Structure Prediction". International Journal of Molecular Sciences, 22(21), 11449.

```
@article{ijms222111449,
AUTHOR = {Oliveira, Gabriel Bianchin de and Pedrini, Helio and Dias, Zanoni},
TITLE = {Ensemble of Template-Free and Template-Based Classifiers for Protein Secondary Structure Prediction},
JOURNAL = {International Journal of Molecular Sciences},
VOLUME = {22},
YEAR = {2021},
NUMBER = {21},
ARTICLE-NUMBER = {11449},
URL = {https://www.mdpi.com/1422-0067/22/21/11449},
ISSN = {1422-0067},
DOI = {10.3390/ijms222111449}
}
```