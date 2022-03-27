# PDE-CNNs (Global Layer) 

CNNs using the Global Layer composed of Partial Differential Equations


## Brief Description  

## Installation 

Our codebase is written using [PyTorch](https://pytorch.org). You can set up the environment using [Conda](https://www.anaconda.com/products/individual) and executing the following commands.  

```
conda create --name pytorch-1.10 python=3.10
conda activate pytorch-1.10
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```

Please update the last command as per your system specifications (see [PyTorch-Install](https://pytorch.org/get-started/locally/)). Although we have not explicitly tested all the recent PyTorch versions, but you should be able to run this code on PyTorch>=1.7 and Python>=3.7

Please install the following packages used in our codebase.

```
pip install tqdm
pip install thop
pip install timm==0.5.4
pip install pyyaml
```

In order to run the MNIST Neural ODE code, you need to install torchdiffeq that provides black-box ODE solvers used in Neural ODEs. 

```
pip install torchdiffeq 
```


## Training Scripts 

Global-Layer resides in the file global\_layer.py and is the central place where PDE constrained feature transition is implemented. 

We provide train commands for various model configurations. Please uncomment the command coresponding to the model you want to train and execute. For MNIST and CIFAR datasets, we will download the data directly using PyTorch API. For Imagenet, please download the data from the official website and update the runner-imagenet.sh script.

```
bash runner-mnist10.sh
bash runner-cifar.sh
bash runner-imagenet.sh
```

## Reference (Bibtex entry)


```
@article{kag2022globallayer,
  title={Condensing CNNs with Partial Differential Equations},
  author={Kag, Anil and Saligrama, Venkatesh},
  journal={In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2022}
}

```

