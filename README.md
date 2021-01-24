  

# Data-driven discovery of multiscale chemical reactions

## Abstract

We propose a method to discover multiscale chemical reactions governed by the law of mass action from data.
 -  We use one matrix to represent the stoichiometric coefficients for both the reactants and products in a system without catalysis reactions. The negative entries in the matrix denote the stoichiometric coefficients for the reactants and the positive ones denote the products.
 
 - We find that the conventional optimization methods usually get stuck in the local minima and could not find the true solution in learning multiscale chemical reactions. To overcome this difficulty, we propose to perform a round operation on the stoichiometric coefficients which are closed to integers and do not update them in the afterwards training. With such a treatment, the dimension of the searching space is greatly reduced and the global mimina is eventually obtained. Several numerical experiments including the classical Michaelis–Menten kinetics and the hydrogen oxidation reactions verify the good performance of our algorithm in learning multiscale chemical reactions.

  

## Structure of this repository

  

 ### regression 
this folder constains the code for multiscale nonlinear regression problem in Section 2 in the [paper](https://arxiv.org/abs/2101.06589)
```sh
cd regression
python regression.py
```

### reaction
this folder constains the code for learning multiscale chemical reactions in Section 4 in the [paper](https://arxiv.org/abs/2101.06589)
```sh
cd reaction
python multiscale_reaction.py
```
  

## Cite
Juntao Huang, Yizhou Zhou, and Wen-An Yong. [Data-driven discovery of multiscale chemical reactions governed by the law of mass action](https://arxiv.org/abs/2101.06589), arXiv preprint arXiv:2101.06589 (2021).