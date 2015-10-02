# ParseNet: Looking Wider to See Better

By Wei Liu, Andrew Rabinovich, and Alex Berg.

### Introduction

ParseNet is a unified framework for semantic segmentation with CNN. You can use the package to train/evaluate a network for segmentation. For more details, please refer to our [arXiv paper](http://arxiv.org/abs/1506.04579).

### Citing ParseNet

Please cite ParseNet in your publications if it helps your research:

    @article{liu15parsenet,
      Author = {Liu, Wei and Rabinovich, Andrew and Berg, Alexander},
      Journal = {arXiv preprint arXiv:1506.04579},
      Title = {ParseNet: Looking Wider to See Better},
      Year = {2015}
    }

### Contents
1. [Installation](#installation)
2. [Model](#model)

### Installation
1. Get the code
  ```Shell
  git clone https://github.com/weiliu89/caffe.git
  git checkout fcn
  ```

2. Build the code. Please follow [Caffe instruction](http://caffe.berkeleyvision.org/installation.html) to install all necessary packages and build it.
  ```Shell
  make -j8
  make mat
  make py
  make test -j8
  make runtest -j8
  ```

  **Note:** Since ParseNet merges [#2016](https://github.com/BVLC/caffe/pull/2016), which will cause a crash on exit. You can safely ignore it as it is a known side effect.

    syncedmem.cpp:16] Check failed: error == cudaSuccess (29 vs. 0) driver shutting down

### Model
The ParseNet model and solver on PASCAL has been shared at the [Caffe's Model Zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo). Please check it out for more details.
