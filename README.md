![UTA-DataScience-Logo](https://user-images.githubusercontent.com/89792487/208189079-d4fc4d67-01bc-4397-891e-52f05330eb12.png) ![image](https://github.com/lemaurK/SantanderBankBinaryClassification/assets/89792487/e1c3b1bc-794b-487e-b4de-f38c23d6f449)


# Santander Customer Transaction Prediction

* This repository holds an attempt to complete the [Santander Customer Transaction Prediction](https://www.kaggle.com/competitions/santander-customer-transaction-prediction/overview) using a few dimension reduction techniques and three models: Logistic Regression, Deep Learning, and K-Nearest Neighbors.

## Overview
**Definition of Challenge** 
* The overarching question this challenge asks is "Can you identify who will make a transaction?"
* Given 199 features of anonymized data on 200,000 customers, including a **binary** feature outlining whether the customer made the purchase or not, use ML/DL to create a model that predicts, with great accuracy, if a given customer will make a purchase.

**My Approach**
* Due to the high dimensionality of this dataset, I employed a few dimension redcution algorithms in order to feed my models cleaned, balanced, and appropriate data. 
* Dimension Reduction Techniques
  * Principal Component Analysis (PCA)
  * Random Forest
  * Variance Inflation Factor (VIF)
* Models
  * Logistic Regression 
  * Deep Learning Neural Network
  * K-Nearest Neighbors

**Performance Achieved**
* The highest accuracy was achieved by Logistic Regression at 91.2% and the highest Kaggle score was achieved by Deep Learning at 0.629. The higest Kaggle score achieved by a competitor was 0.92573.

## Summary of Work Done

### Data

* Data:
  * Type: For example
    * Input: medical images (1000x1000 pixel jpegs), CSV file: image filename -> diagnosis
    * Input: CSV file of features, output: signal/background flag in 1st column.
  * Size: How much data?
  * Instances (Train, Test, Validation Split): how many data points? Ex: 1000 patients for training, 200 for testing, none for validation

#### Preprocessing / Clean up

* Describe any manipulations you performed to the data.

#### Data Visualization

Show a few visualization of the data and say a few words about what you see.

### Problem Formulation

* Define:
  * Input / Output
  * Models
    * Describe the different models you tried and why.
  * Loss, Optimizer, other Hyperparameters.

### Training

* Describe the training:
  * How you trained: software and hardware.
  * How did training take.
  * Training curves (loss vs epoch for test/train).
  * How did you decide to stop training.
  * Any difficulties? How did you resolve them?

### Performance Comparison

* Clearly define the key performance metric(s).
* Show/compare results in one table.
* Show one (or few) visualization(s) of results, for example ROC curves.

### Conclusions

* State any conclusions you can infer from your work. Example: LSTM work better than GRU.

### Future Work

* What would be the next thing that you would try.
* What are some other studies that can be done starting from here.

## How to reproduce results

* In this section, provide instructions at least one of the following:
   * Reproduce your results fully, including training.
   * Apply this package to other data. For example, how to use the model you trained.
   * Use this package to perform their own study.
* Also describe what resources to use for this package, if appropirate. For example, point them to Collab and TPUs.

### Overview of files in repository

* Describe the directory structure, if any.
* List all relavent files and describe their role in the package.
* An example:
  * utils.py: various functions that are used in cleaning and visualizing data.
  * preprocess.ipynb: Takes input data in CSV and writes out data frame after cleanup.
  * visualization.ipynb: Creates various visualizations of the data.
  * models.py: Contains functions that build the various models.
  * training-model-1.ipynb: Trains the first model and saves model during training.
  * training-model-2.ipynb: Trains the second model and saves model during training.
  * training-model-3.ipynb: Trains the third model and saves model during training.
  * performance.ipynb: loads multiple trained models and compares results.
  * inference.ipynb: loads a trained model and applies it to test data to create kaggle submission.

* Note that all of these notebooks should contain enough text for someone to understand what is happening.

### Software Setup
* List all of the required packages.
* If not standard, provide or point to instruction for installing the packages.
* Describe how to install your package.

### Data

* Point to where they can download the data.
* Lead them through preprocessing steps, if necessary.

### Training

* Describe how to train the model

#### Performance Evaluation

* Describe how to run the performance evaluation.


## Citations

* Provide any references.






