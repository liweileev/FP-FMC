# FP-FMC

This is the demo code for paper: Filter Pruning via Feature Map Clustering

## Environment

All our experiments are conducted on NVIDIA GeForce RTX 3090.
The code has been tested by Python 3.8.5, Pytorch 1.8.1 and CUDA 11.1 on Ubuntu 20.04.1. 
The necessary dependencies are installed as follows:


```
pip install requirements.txt
```

## Running Code

### Prepare the dataset

The dataset used in the demo code is CIFAR-10 which can be downloaded from the official website:
[https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz)

Then unzip them wherever you like with the following structure:

```
DATASET_DIR
└───cifar-10-batches-py
    │  batches.meta
    │  data_batch_1
    │  data_batch_2
    │  data_batch_3
    │  data_batch_4
    │  data_batch_5
    │  readme.html
    │  test_batch
```


### Prepare the baseline models

The demo code involves the pruning of 5 models, and their corresponding benchmark models are downloaded from the following online public resources:

VggNet: [https://drive.google.com/file/d/1i3ifLh70y1nb8d4mazNzyC4I27jQcHrE](https://drive.google.com/file/d/1i3ifLh70y1nb8d4mazNzyC4I27jQcHrE)

DenseNet: [https://drive.google.com/file/d/12rInJ0YpGwZd\_k76jctQwrfzPubsfrZH](https://drive.google.com/file/d/12rInJ0YpGwZd\_k76jctQwrfzPubsfrZH)

GoogleNet: [https://drive.google.com/file/d/1rYMazSyMbWwkCGCLvofNKwl58W6mmg5c](https://drive.google.com/file/d/1rYMazSyMbWwkCGCLvofNKwl58W6mmg5c)

ResNet-56: [https://drive.google.com/file/d/1f1iSGvYFjSKIvzTko4fXFCbS-8dw556T](https://drive.google.com/file/d/1f1iSGvYFjSKIvzTko4fXFCbS-8dw556T)

ResNet-110: [https://drive.google.com/file/d/1uENM3S5D_IKvXB26b1BFwMzUpkOoA26m](https://drive.google.com/file/d/1uENM3S5D_IKvXB26b1BFwMzUpkOoA26m)

After downloading, place them in the pretrain folder without any changes.



### Step0: Find appropriate K for clustering

This step is the preparation stage of FP-FMC, which is used to determine the clustering hyper-parameter $K_{L_i}$ for each convolutional layer.

Take DenseNet as an example, run the following script in terminal:

```
python3 step0_findK.py --data_dir= [DATASET_DIR] --arch=densenet --batch_calcK=1 --batch_size=128 --gpu=0

```
Then, the appropriate $K_{L_i}$ and history records of clustering results corresponding to each convolutional layer are saved in the folder:  output/step0\_appropriateK/densenet/

Since this process is relatively time-consuming, we have prepared a copy of the results for different architectures in the [Google Drive](https://drive.google.com/drive/folders/1k-2c1wiBpWdnQAUu-TH8Dc9hxObds4CF?usp=sharing). You can also obtain similar results through the above script.


### Step1: Feature flustering

Cluster the feature maps layer by layer with the last step's results (appropriate Ks):

```
python3 step1_featureMap_clustering.py --data_dir=[DATASET_DIR] --arch=densenet --batch_clustering=1 --batch_size=128 --gpu=0 --Ks_file=output/step0_appropriateK/densenet/samples=128.npy 
```
We use the prepared Ks_file in the above script, you can use the result from the last step.
The outputs of the this step are numpy files per convolutional Layer. 
The results will be saved in the folder: output/step1\_clustering/densenet/samples=128/.
We also provide the copy of results for six network architectures [HERE](https://drive.google.com/drive/folders/1d6xU_I1gbo7IZ6GVnlgBq2jImk1taXsy?usp=sharing) (Google Drive). You can also obtain similar results through the above script.


### Step2: Sort filters

Calculate the redundancy indices (RI) of filters and sort filters by RI:

```
python3 step2_sortFilter_by_clusteringResult.py --clustering_dir=output/step1_clustering/densenet/samples=128 --num_clustering=128/
```

This step do not need to use GPU.
We use the prepared clustering  results numpy file in the above script, you can use the result from the last step.
The outputs of this step are RI of filters and sorted filters list for each convolutional Layer.
The results will be saved in the folder: output/step2\_sort/densenet/samples=128/.
The copy of results per layer for six network architectures are [HERE](https://drive.google.com/drive/folders/158rogZ0xSM-RtPHJvxTVupuUFJtFdK68?usp=sharing) (Google Drive).


### Step3: Prune and fine-tune the model

Prune the baseline model by sorted list from last step and fine-tune the pruned model:

```
python3 step3_pruningTrain.py --arch=densenet --use_pretrain --sorted_conv_prefix=output/step2_sort/densenet/samples=128/ --gpu=0 --lr_decay_step=150,225 --weight_decay=0.002 --compress_rate=[0.]+[0.2]*12+[0.]+[0.2]*12+[0.]+[0.2]*12 
```

We use the prepared sorted filter numpy file in the above script, you can use the result from the last step.
This step will output config.txt, fine-tuning log with net summaries and fine-tuned model weights files in the output directory.
We also provide the config text file, fine-tuning logs and pruned models of all experiments results shown in the main paper in Google Drive.
All of them can be used to test the classification accuracy of pruned FP-FMC models.
The configurations and performances of all models are as follows:

|  Arch   | Top-1%  | FLOPs | Parameters | Compress rates | files |
|  ----  | ----  |  ----  | ----  |  ----  | ---- |
| VGGNet  | 93.89 |104.78M(-66.6%)| 2.5M(-93.3%)    | [0.3]×7+[0.75]×5|<a href="https://drive.google.com/drive/folders/1JgQ-PFeULbcXV1aYtnGrHydL__KjKptB?usp=sharing"><img src="GoogleDrive.svg" height="30" alt="Google Drive Datasets"></a>|
| VGGNet |92.93 |66.95M(-78.7%) |1.90M(-87.3%) | [0.45]×7+[0.78]×5|<a href="https://drive.google.com/drive/folders/1B1Br94ss9nKIujlFFxb0Bs9R2caQdajO?usp=sharing"><img src="GoogleDrive.svg" height="30" alt="Google Drive Datasets"></a>|
|DenseNet|94.45 |173.38M(-38.5%)|0.62M(-40.4%) |[0.2]×12+[0.]+[0.2]×12+[0.]+[0.2]×12|<a href="https://drive.google.com/drive/folders/16p9XZnQS-o9k6yJzCW3_c3qjtfrOOgKs?usp=sharing"><img src="GoogleDrive.svg" height="30" alt="Google Drive Datasets"></a>|
|DenseNet|93.72 |133.17M(-52.78%) |0.45M(-56.7%) | [0.3]×12+[0.1]+[0.3]×12+[0.1]+[0.3]×12|<a href="https://drive.google.com/drive/folders/1Dxe4rFB_aSz4jCIOMbuB840SJfliikKP?usp=sharing"><img src="GoogleDrive.svg" height="30" alt="Google Drive Datasets"></a>|
|DenseNet|93.28 |106.23M(-62.3%) |0.35M(-66.3%) | [0.4]×12+[0.1]+[0.4]×12+[0.1]+[0.4]×12|<a href="https://drive.google.com/drive/folders/1QLb_Y4whv3MaXpKLTcBJUG05u_TvHPY3?usp=sharing"><img src="GoogleDrive.svg" height="30" alt="Google Drive Datasets"></a>|
|GoogleNet|95.11 |0.65B(-57.2%)     |2.86M(-53.50%)    | [0.6]×2+[0.7]×5+[0.8]×2 |<a href="https://drive.google.com/drive/folders/1XXF642rd4wqTNKUMZBNLPsoGrYnWP3hT?usp=sharing"><img src="GoogleDrive.svg" height="30" alt="Google Drive Datasets"></a>|
|GoogleNet|94.83 |0.39B(-74.3%) |2.09M(-65.80%) | [0.85]×2+[0.9]×5+[0.9]×2|<a href="https://drive.google.com/drive/folders/1KeenW40csfE0u7xLoYpZ7UFWY9Ram6yH?usp=sharing"><img src="GoogleDrive.svg" height="30" alt="Google Drive Datasets"></a>|
|ResNet-56 |95.28 |90.35M(-28.0%)     |0.66M(-22.4%)    | [0.]+[0.18]×29 |<a href="https://drive.google.com/drive/folders/1drU-ss-3esjD_72VCemY9_AUF_KDUzZQ?usp=sharing"><img src="GoogleDrive.svg" height="30" alt="Google Drive Datasets"></a>|
|ResNet-56|94.22 |65.94M(-47.4%) |0.48M(-42.8%) | [0.]+[0.15]×2+[0.4]×27|<a href="https://drive.google.com/drive/folders/1Z0LkpZ08YXT8yC1B__JTv60jMXDLVgK7?usp=sharing"><img src="GoogleDrive.svg" height="30" alt="Google Drive Datasets"></a>|
|ResNet-56|92.26 |34.78M(-74.1%) |0.24M(-70.0%) | [0.]+[0.4]×2+[0.5]×9+[0.6]×9+[0.7]×9|<a href="https://drive.google.com/drive/folders/1R70iU5PYjwA8jggH3OSLVBEWMq-avPZc?usp=sharing"><img src="GoogleDrive.svg" height="30" alt="Google Drive Datasets"></a>|
|ResNet-110 |95.32 |140.54M(-44.4%)     |1.04M(-39.1%)    | [0.]+[0.2]×2+[0.3]×18+[0.35]×36 |<a href="https://drive.google.com/drive/folders/1wHTwgeG_I-06uTkrp4U3YcXiYlrz5fnc?usp=sharing"><img src="GoogleDrive.svg" height="30" alt="Google Drive Datasets"></a>|
|ResNet-110|95.17 |101.97M(-59.6%) |0.72M(-58.1%) | [0.]+[0.25]×2+[0.4]×18+[0.55]×36|<a href="https://drive.google.com/drive/folders/1Tj2eggPkpIuVCmkWpuuDlQPbE6KzRXOM?usp=sharing"><img src="GoogleDrive.svg" height="30" alt="Google Drive Datasets"></a>|
|ResNet-110|93.15 |71.69M(-71.6%) |0.54M(-68.3%) | [0.]+[0.4]×2+[0.5]×18+[0.65]×36|<a href="https://drive.google.com/drive/folders/1PhxDrI3lajxGE_bDwttFU5uXBj_NTBgr?usp=sharing"><img src="GoogleDrive.svg" height="30" alt="Google Drive Datasets"></a>|

## Others

In addition, we also provide the code and data of Figure 2c, Figure 3, and Figure 9 in the main paper.

### RI barplot in Figure 2C and Figure 3

```
python3 plots/R_barplot.py

```

### K boxplots in Figure 9
```
python3 plots/K_boxplot.py

```