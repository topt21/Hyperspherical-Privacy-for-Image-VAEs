# VAE Testing Grounds

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
