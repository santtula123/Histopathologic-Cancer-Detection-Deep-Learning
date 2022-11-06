## Histopatologinen syövän tunnistus 

**Tekijä:** Santeri Moilanen, santerimoilanen@gmail.com

**Tiivistelmä**: Tutkimusartikkelin [Deep Convolutional Neural Network with TensorFlow and Keras to Classify Skin Cancer Images](https://www.researchgate.net/publication/343409875_Deep_Convolutional_Neural_Network_with_TensorFlow_and_Keras_to_Classify_Skin_Cancer_Images) neuroverkkoarkkitehtuurin replikointi ja optimointi.

## Testattu seuraavilla kirjastoilla

- [Python 3.9.12](https://www.python.org/)
- [Tensorflow 2.7.0](https://www.tensorflow.org/overview/?hl=fi)
- [Pandas 1.4.3](https://pandas.pydata.org/)
- [NumPy 1.23.2](https://numpy.org/)
- [sklearn 1.1.1](https://scikit-learn.org/)
- [matplotlib 3.5.2](https://matplotlib.org/)
- [seaborn 0.12.1](https://seaborn.pydata.org)
- [opendatasets 0.1.22](https://github.com/JovianML/opendatasets)


## Työn rakenne

```
loppuprojekti
│   README.md
│   report.md    
│   param_grid.txt
│   kaggle_submission.csv
│   1_data_download_preprocessing.ipynb
│   2_grid_search.py
│   3_model_diagnostics_and_predictions.ipynb
│
└───data
└───data_limited
└───gs_dnn_ensemble_20221023T2328
└───gs_dnn_ensemble_20221025T1510
└───gs_dnn_ensemble_20221028T1838
└───histopathologic-cancer-detection
└───resources
```

ks. [raportti](./report.md).

## Tulokset

Parhaan mallin (ks. [2D-CNN-2](./gs_dnn_ensemble_20221023T2328/model_infos/2D-CNN-2.png)) ja 
[summary](./gs_dnn_ensemble_20221023T2328/model_infos/2D-CNN-2_summary.txt) suorituskyky evaluointidataa vastaan:

| TP | TN | FN | FP | ACC | PREC | REC | FSCORE
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| 12490 | 7914 | 671 | 942 | 0.92674 | 0.92987 | 0.94902 | 0.93934 |

Parhaan mallin suorituskyky testidataa vastaan:

| TP | TN | FN | FP | ACC | PREC | REC | FSCORE
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| 12185 | 7934 | 660 | 938 | 0.92642 | 0.92852 | 0.94862 | 0.93846 |

