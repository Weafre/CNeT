# Lossless Point Cloud Geometry and Attribute Compression Using a Learned Conditional Probability Model
* **Authors**:
[Dat T. Nguyen](https://scholar.google.com/citations?hl=en&user=uqqqlGgAAAAJ),
[André Kaup](https://scholar.google.de/citations?user=0En1UwQAAAAJ&hl=de),

* **Affiliation**: Friedrich-Alexander University Erlangen-Nuremberg, 91058 Erlangen, Germany

* **Accepted to**: [[IEEE Transactions on Circuits and Systems for Video Technology]]([https://ieeexplore.ieee.org/xpl/conhome/1000349/all-proceedings](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=76))

* **Links**: [[Arxiv]](https://arxiv.org/pdf/2303.06519.pdf),  [[IEEE Xplore]](https://ieeexplore.ieee.org/abstract/document/10024999))



## Description

- Abstract: In this paper, we present an efficient lossless point cloud compression method that uses sparse tensor-based deep neural networks to learn point cloud geometry and color probability distributions. Our method represents a point cloud with both occupancy feature and three attribute features at different bit depths in a unified sparse representation. This allows us to efficiently exploit feature-wise and point-wise dependencies within point clouds using a sparse tensor-based neural network and thus build an accurate auto-regressive context model for an arithmetic coder. To the best of our knowledge, this is the first learning-based lossless point cloud geometry and attribute compression approach. Compared with the-state-of-the-art lossless point cloud compression method from Moving Picture Experts Group (MPEG), our method achieves 22.6% reduction in total bitrate on a diverse set of test point clouds while having 49.0% and 18.3% rate reduction on geometry and color attribute component, respectively. 

- This is a Pytorch implementation of attribute compression module from the paper, for the geometry compression module, please refer to [SparseVoxelDNN](https://github.com/Weafre/SparseVoxelDNN).

## Requirments

- pytorch 1.9.0, py3.9_cuda11.1_cudnn8.0.5_0 
- MinkowskiEngine 0.5
- torchac 0.9.3
- packages in mink_environment.yml

## Getting started
- Install the dependencies using the conda environment.yml file
```shell
conda env create --name mink --file environment.yml
```
## Dataset
- We use four datasets with color attributes: MPEG Owlii, MPEG 8i, MVUB, and MPEG CAT1 (except point clouds used for testing). All points clouds are partitioned into a set of points with maximum bounding box of 64. 
- Example: the following command performs the partitioning for Owlii dataset: 
```
    python3 -m DataPreprocessing.ds_pc_octree_blocks datasets/Owlii datasets/Owlii_bl64 --vg_size 1024 --level 4 
```
The partitioned blocks are written into the datasets/Owlii_bl64 directory, doing the same for other datasets
- Randomly split blocks to training and validating set: 
``` 
    python3 -m DataPreprocessing.ds_select_randomly_move datasets/ datasets/validation/ 0.3
``` 
- An example training set is located in [ExampleTrainingSet](https://github.com/Weafre/MNeT/tree/main/ExampleTrainingSet)
## Training

    python3 -m Training.cnet_attribute_training -trainset ../Datasets/CNeT_TrainingSet/33K/  -validset ../Datasets/CNeT_ValidSet/33K/ -flag test -outputmodel Model/  -lr 7 -useDA 8   --color -opt 1 -dim 2 -ngpus 1  -batch 2  -bacc 1
The argument description can be found in the Training/cnet_attribute_training.py
## Encoding

    python3 -m Encoders.cnet_attribute_encoder -level 10 -ply ply_path -output Output/   -color_voxeldnn_path checkpoint1 -color_voxeldnn_path checkpoint2 -color_voxeldnn_path checkpoint3  -signaling signal

Checkpoints can be downloaded from [here](https://faubox.rrze.uni-erlangen.de/getlink/fiWwcjc9A5mJwSNqfMhoBa/)


