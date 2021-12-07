# AWGAN
Codes for paper: AWGAN: A Powerful Batch Effect Removal Model for scRNA-seq Data

# Download
To install this tool, please use this code:
```
pip install awgan
```
# Brief tutorial

To run our method, the first thing is to import necessary packages:
```
import scprep
import numpy as np
import pandas as pd
import graphtools as gt
import os
import scanpy as sc
from skmisc.loess import loess

import torch.autograd
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F  

from collections import Counter
```
Then we need to load the scRNA-seq data with batch information:
```
adata = sc.read_loom('/content/drive/MyDrive/data/DC.loom', sparse=False) # use human dentritic dataset as one example
# scanpy.pp.highly_variable_genes` with `flavor='seurat_v3'` expects raw count data.
adata.X = np.float32(np.int32(adata.X))
adata = awgan.preprocessing.data_preprocess(adata) #preprocessing
```
Now we can assign the training sequence and generate required input information:
```
c = Counter(adata.obs['batch'])
c_keys = list(c.keys())
adata1 = adata[adata.obs['batch'] ==c_keys[0]]
adata2 = adata[adata.obs['batch'] !=c_keys[0]]
```
We can utilize this step to train the AWGAN, and get output_results as correction matrix:
```
output_results, model = awgan.model.sequencing_train(adata1,adata2,c_keys, epoch=40)
```
For more information and other examples, please take a look at 


# Package Requirement

To run AWGAN and other benchmarks, we suggest you install the python environment with the version-specific package listed in this table. 

|                 | Package    | Version |
|-----------------|------------|---------|
| Python packages | pytorch    | 1.9.0   |
|                 | scanpy     | 1.8.1   |
|                 | scIB       | 0.1.1   |
|                 | scprep     | 1.1.0   |
|                 | umap-learn | 0.5.1   |
|                 | phate      | 1.0.7   |
|                 | imap       | 1.0.0   |
|                 | scVI       | 0.6.8   |
|                 | bbknn      | 1.5.1   |
|                 | mnnpy      | 0.1.9.5 |
|                 | Harmonypy  | 0.0.5   |
|                 | rpy2       | 3.4.5   |
| R packages      | kBET       | 0.99.6  |
|                 | LISI       | 1.0.0   |
|                 | liger      | 1.0.0   |
|                 | Seurat     | 4.0.6   |

# Reference
Please use:

AWGAN: A Powerful Batch Correction Model for scRNA-seq Data.
Tianyu Liu, Yuge Wang, Hong-yu Zhao. bioRxiv 2021.11.08.467781; doi: https://doi.org/10.1101/2021.11.08.467781
