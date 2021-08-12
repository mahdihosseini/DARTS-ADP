# DARTS-ADP
- [DARTS-ADP](#darts-adp)
  - [Introduction](#introduction)
  - [Datasets](#datasets)
    - [ADP](#adp)
    - [BCSS](#bcss)
    - [BACH](#bach)
    - [Osteosarcoma](#osteosarcoma)
  - [Architectures](#architectures)
    - [Overall architecture](#overall-architecture)
    - [DARTS_ADP_N2](#darts_adp_n2)
    - [DARTS_ADP_N3](#darts_adp_n3)
    - [DARTS_ADP_N4](#darts_adp_n4)
  - [Performance](#performance)
  - [Usage](#usage)
    - [Pretrained models](#pretrained-models)
    - [Training](#training)
## Introduction
We use [Differentiable Architecture Search (DARTS)](https://github.com/quark0/darts) to find the optimal network architectures for Computational Pathology (CPath) applications. This repository provides three architectures that are searched on the ADP dataset, and can be well transfered to other CPath datasets.

## Datasets
We search the architectures on the ADP dataset and transfer them to three more datasets including BCSS, BACH, and Osteosarcoma.
### ADP
ADP is a multi-label histological tissue type dataset. This is where the architectures are searched. More details can be found in the [ADP Website](https://www.dsp.utoronto.ca/projects/ADP/) and [this paper](https://openaccess.thecvf.com/content_CVPR_2019/html/Hosseini_Atlas_of_Digital_Pathology_A_Generalized_Hierarchical_Histological_Tissue_Type-Annotated_CVPR_2019_paper.html).

### BCSS
BCSS is a multi-label breast cancer tissue dataset. More details can be found [here](https://academic.oup.com/bioinformatics/article/35/18/3461/5307750).

### BACH
BACH is a single-label breast cancer histology image dataset. More details can be found in [this paper](https://www.sciencedirect.com/science/article/abs/pii/S1361841518307941).

### Osteosarcoma
This dataset contains osteosarcoma histology images and is available through [this website](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=52756935).

## Architectures
### Overall architecture
![Network structure](/figures/network_macro_structure.png)
Above is the overall structure of the searched networks, where the normal and reduction cells are searched. A cell can be represented as a directed acyclic graph with nodes and edges. Each node is a feature map, and each edge belongs to one of the candidate operations, including 3x3 and 5x5 seperable convolutions, 3x3 and 5x5 dilated separable convolutions, 3x3 max pooling, 3x3 average pooling, and skip-connection.

We alter the number of nodes and search for the optimum architectures in each configuration. Here we present three best-performing architectures with different number of nodes.

### DARTS_ADP_N2
This network contains 2 nodes in each cell.
![DARTS_ADP_N2](figures/cells_n_2_c_4.png) 

### DARTS_ADP_N3
This network contains 3 nodes in each cell.
![DARTS_ADP_N3](figures/cells_n_3_c_4.png)

### DARTS_ADP_N4
This network contains 4 nodes in each cell.
![DARTS_ADP_N4](figures/cells_n_4_c_4.png)

## Performance
The searched networks are trained in 4 datasets and compared with multiple state-of-the-art networks. Results show their superioty in prediction accuracy and computation complexity.
![Performance](figures/performance.png)

## Usage
### Pretrained models
Pretrained model weights are provided in the `/pretrained` folder, where four subfolders contains pretrained weights for three architectures on each dataset. The script `test.py` demonstrates how to load the pretrained weights of a network and test its performance.

First, download the dataset you want to train on and store them to a local directory.

Then, open `test_demo.sh` and edit the following:
- Change the path of `--data` to where you store the downloaded data.
- Change the name of `--dataset` accordingly. Valid names are `ADP`, `BCSS`, `BACH`, and `OS`.
- Select the architecture you want to train. Valid architectures are `DARTS_ADP_N2`, `DARTS_ADP_N3`, and `DARTS_ADP_N4`.
- Change `--model_path` according to the chosen dataset and architect name. E.g., `./pretrained/ADP/darts_adp_n4.pt` if testing `DARTS_ADP_N4` on `ADP`.

Now, simply run 
```
cd path/to/this-repo
sh test_demo.sh
```

### Training
First, download the dataset you want to train on and store them to a local directory.

Then, open `train_demo.sh` and edit the following:

- Change the path of `--data` to where you store the downloaded data.
- Change the name of `--dataset` correspondingly. Valid names are `ADP`, `BCSS`, `BACH`, and `OS`.
- Select the architecture you want to train. Valid architectures are `DARTS_ADP_N2`, `DARTS_ADP_N3`, and `DARTS_ADP_N4`.
- Other hyperparameters including learning rate, batch size and epochs, etc.

You can open `train.py` to see full details of hyperparameters.

Now, run the demo script to start training
```
cd path/to/this-repo
sh train_demo.sh
```