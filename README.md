# Recommender System for Amazon Dataset

## Data set
http://jmcauley.ucsd.edu/data/amazon/

## Project structure

* graphs_* - variuos visualizations
* data_preprocessing.ipynb - preprocess review texts woth spaCy lib
* baselines_* - train baselines and gather data
* matrix_factorization_with_als.py - basic matrix factorizations using ALS
* other *.py - out utils

### CDL-based models
* cdl_conv_mf - CDL models with convolutional net for building reviews representations
* cld_sdae - general CDL model with SDAE for building reviews representations

## Evaluation

* cdl_metrics - tests CDL model using MAE\RMSE and writes recall@M\precision@M to file  

## References

* Hao Wang, Naiyan Wang, and Dit-Yan Yeung. “Collaborative deep learning for recommender systems”. In:Proceedings  of  the  21th  ACM  SIGKDD International Conference on Knowledge Discovery and Data Mining. ACM. 2015, pages 1235–124


* Surprise library http://surpriselib.com/