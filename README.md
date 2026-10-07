# Hyperspherical Privacy for Image VAEs: Evaluating VAEs Under Differing Privacy Frameworks

This repository contains the code for our master thesis:

> **Hyperspherical Privacy for Image VAEs: Evaluating VAEs Under Differing Privacy Frameworks**  
> Alexander J.O. Ojutkangas, Tobias Pettersson  
> Blekinge Institute of Technology, Sweden  
---

## Summary

This thesis aims to investigate the performance of hyperspherical VAEs when trained with privacy mechanisms. To achieve this, we compare hyperspherical VAEs with common Gaussian VAEs in terms of the privacy–utility trade-off through several empirical experiments. We also explore their robustness against certain privacy attacks.

## Details

Model based on code from https://github.com/AntixK/PyTorch-VAE/ and https://github.com/nicola-decao/s-vae-pytorch.

vMF-based differentially private SGD based on https://arxiv.org/abs/2211.04686 with implementation of vMF-mechanism adapted from https://hal.science/hal-04004568v2

## Installation

```
$ git clone https://github.com/topt21/vae-testing-grounds
$ cd vae-testing-grounds
$ git submodule init
$ git submodule update
$ python -m venv venv
$ source venv/bin/activate
$ pip install -r requirements.txt
```
## Usage

See `$ ./main.py -h` and `$ ./main.py {cmd} -h` for usage.

## Citation

TBD
