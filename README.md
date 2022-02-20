  

# Data-driven discovery of multiscale chemical reactions

## Abstract

We propose a ***partial-parameters-freezing (PPF)*** technique to discover ***multiscale*** chemical reactions governed by the law of mass action from data. 

The idea of the PPF technique is to progressively determine the network parameters by using the fact that the stoichiometric coefficients are integers. With such a technique, the dimension of the searching space is gradually reduced in the training process and the global mimina can be eventually obtained.

  

## Structure of this repository

  

  

- regression: this folder constains the code for multiscale nonlinear regression problem in Section 2 in the [paper](https://arxiv.org/abs/2101.06589)

```sh
cd regression
python regression.py
```

  

- reaction: this folder constains the code for learning multiscale chemical reactions in Section 4 in the [paper](https://arxiv.org/abs/2101.06589)

```sh
cd reaction
python multiscale_reaction.py
```
  

## Cite
J. Huang, Y. Zhou, and W.-A. Yong. [Data-driven discovery of multiscale chemical reactions governed by the law of mass action](https://www.sciencedirect.com/science/article/pii/S0021999121006380), Journal of Computational Physics, 448, 110743, 2022.
