# SqueezeNet-PyTorch

## Overview

This repository contains an op-for-op PyTorch reimplementation
of [SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size
](https://arxiv.org/pdf/1602.07360v4.pdf).

## Table of contents

- [SqueezeNet-PyTorch](#squeezenet-pytorch)
    - [Overview](#overview)
    - [Table of contents](#table-of-contents)
    - [Download weights](#download-weights)
    - [Download datasets](#download-datasets)
    - [How Test and Train](#how-test-and-train)
        - [Test](#test)
        - [Train model](#train-model)
        - [Resume train model](#resume-train-model)
    - [Result](#result)
    - [Contributing](#contributing)
    - [Credit](#credit)
        - [SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size](#squeezenet-alexnet-level-accuracy-with-50x-fewer-parameters-and-05mb-model-size)

## Download weights

- [Google Driver](https://drive.google.com/drive/folders/17ju2HN7Y6pyPK2CC_AqnAfTOe9_3hCQ8?usp=sharing)
- [Baidu Driver](https://pan.baidu.com/s/1yNs4rqIb004-NKEdKBJtYg?pwd=llot)

## Download datasets

Contains MNIST, CIFAR10&CIFAR100, TinyImageNet_200, MiniImageNet_1K, ImageNet_1K, Caltech101&Caltech256 and more etc.

- [Google Driver](https://drive.google.com/drive/folders/1f-NSpZc07Qlzhgi6EbBEI1wTkN1MxPbQ?usp=sharing)
- [Baidu Driver](https://pan.baidu.com/s/1arNM38vhDT7p4jKeD4sqwA?pwd=llot)

Please refer to `README.md` in the `data` directory for the method of making a dataset.

## How Test and Train

Both training and testing only need to modify the `config.py` file.

### Test

- line 29: `model_arch_name` change to `squeezenet`.
- line 31: `model_mean_parameters` change to `[0.485, 0.456, 0.406]`.
- line 32: `model_std_parameters` change to `[0.229, 0.224, 0.225]`.
- line 34: `model_num_classes` change to `1000`.
- line 36: `mode` change to `test`.
- line 89: `model_weights_path` change to `./results/pretrained_models/SqueezeNet-ImageNet_1K-145ddc1c.pth.tar`.

```bash
python3 test.py
```

### Train model

- line 29: `model_arch_name` change to `squeezenet`.
- line 31: `model_mean_parameters` change to `[0.485, 0.456, 0.406]`.
- line 32: `model_std_parameters` change to `[0.229, 0.224, 0.225]`.
- line 34: `model_num_classes` change to `1000`.
- line 36: `mode` change to `train`.
- line 50: `pretrained_model_weights_path` change to `./results/pretrained_models/SqueezeNet-ImageNet_1K-145ddc1c.pth.tar`.

```bash
python3 train.py
```

### Resume train model

- line 29: `model_arch_name` change to `squeezenet`.
- line 31: `model_mean_parameters` change to `[0.485, 0.456, 0.406]`.
- line 32: `model_std_parameters` change to `[0.229, 0.224, 0.225]`.
- line 34: `model_num_classes` change to `1000`.
- line 36: `mode` change to `train`.
- line 53: `resume` change to `./samples/squeezenet-ImageNet_1K/epoch_xxx.pth.tar`.

```bash
python3 train.py
```

## Result

Source of original paper results: [https://arxiv.org/pdf/1602.07360v4.pdf](https://arxiv.org/pdf/1602.07360v4.pdf))

In the following table, the top-x error value in `()` indicates the result of the project, and `-` indicates no test.

|         Model          |   Dataset   | Top-1 error (val) | Top-5 error (val) |
|:----------------------:|:-----------:|:-----------------:|:-----------------:|
|       squeezenet       | ImageNet_1K | 42.5%(**41.9%**)  | 19.7%(**19.6%**)  |

```bash
# Download `SqueezeNet-ImageNet_1K-145ddc1c.pth.tar` weights to `./results/pretrained_models`
# More detail see `README.md<Download weights>`
python3 ./inference.py 
```

Input:

<span align="center"><img width="224" height="224" src="figure/n01440764_36.JPEG"/></span>

Output:

```text
Build `squeezenet` model successfully.
Load `squeezenet` model weights `/SqueezeNet-PyTorch/results/pretrained_models/SqueezeNet-ImageNet_1K-145ddc1c.pth.tar` successfully.
tench, Tinca tinca                                                          (80.22%)
barracouta, snoek                                                           (15.18%)
bolete                                                                      (0.68%)
armadillo                                                                   (0.67%)
reel                                                                        (0.38%)
```

## Contributing

If you find a bug, create a GitHub issue, or even better, submit a pull request. Similarly, if you have questions,
simply post them as GitHub issues.

I look forward to seeing what the community does with these models!

### Credit

#### SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size

*Forrest N. Iandola, Song Han, Matthew W. Moskewicz, Khalid Ashraf, William J. Dally, Kurt Keutzer*

##### Abstract

Recent research on deep neural networks has focused primarily on improving accuracy. For a given accuracy level, it is
typically possible to identify multiple DNN architectures that achieve that accuracy level. With equivalent accuracy,
smaller DNN architectures offer at least three advantages: (1) Smaller DNNs require less communication across servers
during distributed training. (2) Smaller DNNs require less bandwidth to export a new model from the cloud to an
autonomous car. (3) Smaller DNNs are more feasible to deploy on FPGAs and other hardware with limited memory. To provide
all of these advantages, we propose a small DNN architecture called SqueezeNet. SqueezeNet achieves AlexNet-level
accuracy on ImageNet with 50x fewer parameters. Additionally, with model compression techniques we are able to compress
SqueezeNet to less than 0.5MB (510x smaller than AlexNet).
The SqueezeNet architecture is available for download here: [this https URL](https://github.com/DeepScale/SqueezeNet)

[[Paper]](https://arxiv.org/pdf/1602.07360v4.pdf)

```bibtex
@article{SqueezeNet,
    Author = {Forrest N. Iandola and Song Han and Matthew W. Moskewicz and Khalid Ashraf and William J. Dally and Kurt Keutzer},
    Title = {SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and $<$0.5MB model size},
    Journal = {arXiv:1602.07360},
    Year = {2016}
}
```