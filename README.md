# Multi-Class Classification with Predictive Maintenance Dataset

In this project, we aim to perform multi-class classification using the Predictive Maintenance Dataset. Below are the steps we'll follow along with explanations for each part:

## 1. Data Preprocessing

We'll use various preprocessing techniques to clean and standardize/normalize the dataset. This involves handling missing values, encoding categorical variables, and scaling numerical features.

![Screenshot 2024-03-10 192838](https://github.com/Sarah9515/Multi-Class-Cassification/assets/72395246/1a8b8aec-2dde-4ff8-a477-47fd324ad208)

![Screenshot 2024-03-10 192931](https://github.com/Sarah9515/Multi-Class-Cassification/assets/72395246/e24ce45d-4304-4498-a48c-b5276db0089a)


## 2. Exploratory Data Analysis (EDA)

EDA techniques will be applied to gain insights into the dataset's structure, distribution of features, relationships between variables, and potential patterns.

![Screenshot 2024-03-10 192213](https://github.com/Sarah9515/Multi-Class-Cassification/assets/72395246/9505a37e-7d12-42d2-82eb-7cbe32b6beb2)

![Screenshot 2024-03-10 192349](https://github.com/Sarah9515/Multi-Class-Cassification/assets/72395246/33d377b3-5b28-408d-9a60-006b01a77928)

## 3. Data Augmentation

To address class imbalance, data augmentation techniques such as oversampling, undersampling, or synthetic data generation will be applied to balance the dataset.

![Screenshot 2024-03-10 192442](https://github.com/Sarah9515/Multi-Class-Cassification/assets/72395246/8e42d9e7-8e8d-4259-bc16-0b5fef7b99fc)

So we need to balance the target variable using SMOTE method

## 4. Deep Neural Network Architecture

We will establish a Deep Neural Network (DNN) architecture using PyTorch library to handle the multi-class classification task. The architecture will be designed with appropriate layers, activations, and regularization techniques.

## 5. Hyperparameter Tuning

GridSearch tool from sklearn library will be utilized to choose the best hyperparameters for the DNN model. This includes tuning parameters like learning rate, optimizers, number of epochs, and model architecture.

## 6. Visualization

Loss and accuracy metrics will be plotted against epochs for both training and test datasets. Interpretations will be provided based on the trends observed in the graphs.

## 7. Performance Metrics Calculation

Metrics such as accuracy, sensitivity, F1 score, etc., will be calculated on both training and test datasets to evaluate the model's performance.

## 8. Regularization Techniques

Several regularization techniques such as dropout, L1/L2 regularization, and batch normalization will be applied to the architecture. The results obtained with the regularized model will be compared with the initial model to analyze improvements in performance and generalization.

By following these steps, we aim to develop an efficient deep learning model for multi-class classification on the Predictive Maintenance Dataset.
