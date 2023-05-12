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
* Type:
  * 200 anonymized features representing customer behavior/history.
* Training Dataset
  * 200000 rows × 202 columns
  * Includes target variable
* Testing Dataset 
  * 200000 rows × 201 columns
  * Omits target variable
* Size: 1.06 GB for both the training and testing datasets
* Train & Test Split **after Dimension Reduction**:
  * Training Dataset
    * 200000 rows × 177 columns
  * Testing Dataset
    * 200000 rows × 176 columns


#### Preprocessing / Clean up

**Dimension Reduction**
* Prinicipal Component Analysis (PCA)
![image](https://github.com/lemaurK/SantanderBankBinaryClassification/assets/89792487/ac404f92-fb59-4b63-804d-a22cc9776b9c)
![image](https://github.com/lemaurK/SantanderBankBinaryClassification/assets/89792487/0310b22a-1a36-4932-911d-bfd10765f64a)
  * Due to the nature of the dataset explained in the **Data Visualization** section, I couldn't use these results.

* Random Forest Regressor
![image](https://github.com/lemaurK/SantanderBankBinaryClassification/assets/89792487/d501d06b-36fd-4a93-9afd-33409a4c76c8)
  * Due to the nature of the dataset explained in the **Data Visualization** section, I couldn't use these results.

* Variance Inflation Factor 
![image](https://github.com/lemaurK/SantanderBankBinaryClassification/assets/89792487/fa9933c4-d92e-46b3-a08b-feef2fe9fc0b)
  * The features represented in this plot were omitted from model training/testing.

#### Data Visualization

* Dataset Descriptive Statistics
  * Distrtibution of pd.describe() function
    * Training
    
    ![image](https://github.com/lemaurK/SantanderBankBinaryClassification/assets/89792487/7aa7f995-991e-48e7-9903-27ed78b48072)

    * Testing
    
    ![image](https://github.com/lemaurK/SantanderBankBinaryClassification/assets/89792487/0f156ba4-0a8f-4e2f-8ea8-b574cc2da186)
    
  * Distribution of Target Variable
    * Ones - Customer made a purchase (10.049%)
    * Zeros - Customer did not make a purchase (89.951%)
    ![image](https://github.com/lemaurK/SantanderBankBinaryClassification/assets/89792487/75333ccf-e1c1-442e-9871-928ddeb001a3)
  
  * Feature Correlation Heatmap
    * This  
    ![image](https://github.com/lemaurK/SantanderBankBinaryClassification/assets/89792487/151b07b2-b174-43b9-b954-3fa3ce46a884)



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






