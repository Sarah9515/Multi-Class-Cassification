# Multi-Class Classification with Predictive Maintenance Dataset

In this project, we aim to perform multi-class classification using the Predictive Maintenance Dataset. Below are the steps we'll follow along with explanations for each part:

Throughout this lab, I delved into the intricacies of multi-class classification, a technique employed to categorize data into more than two distinct categories. The process entailed meticulous data preprocessing steps, ranging from handling missing values and encoding categorical variables to standardizing or normalizing the data to ensure uniformity across features. With the data suitably prepared, an Exploratory Data Analysis (EDA) ensued, offering insights into the dataset's characteristics through visualization techniques. Transitioning to the Deep Neural Network (DNN) phase, I constructed a Multilayer Perceptron (MLP) model tailored for our classification task. Equipped with a Cross-Entropy loss function and utilizing the Adam optimizer, the MLP model was primed for training. Following model definition, I embarked on the crucial steps of model training and subsequent evaluation to gauge its performance accurately.


## 1. Data Preprocessing

We'll use various preprocessing techniques to clean and standardize/normalize the dataset. This involves handling missing values, encoding categorical variables, and scaling numerical features.

![Screenshot 2024-03-10 192838](https://github.com/Sarah9515/Multi-Class-Cassification/assets/72395246/1a8b8aec-2dde-4ff8-a477-47fd324ad208)

![Screenshot 2024-03-10 192931](https://github.com/Sarah9515/Multi-Class-Cassification/assets/72395246/e24ce45d-4304-4498-a48c-b5276db0089a)


## 2. Exploratory Data Analysis (EDA)

EDA techniques will be applied to gain insights into the dataset's structure, distribution of features, relationships between variables, and potential patterns.

![téléchargé (4)](https://github.com/Sarah9515/Multi-Class-Cassification/assets/72395246/b9e8fc73-7751-4ebe-9854-296f3aade986)

![téléchargé (3)](https://github.com/Sarah9515/Multi-Class-Cassification/assets/72395246/4615bd9a-6e30-4535-b412-7ecb2c277797)



## 3. Data Augmentation

To address class imbalance, data augmentation techniques such as oversampling, undersampling, or synthetic data generation will be applied to balance the dataset.

![téléchargé (2)](https://github.com/Sarah9515/Multi-Class-Cassification/assets/72395246/7e2e95cc-555f-467e-8161-3778dd5c52c9)


So we need to balance the target variable using SMOTE method

## 4. Deep Neural Network Architecture

We will establish a Deep Neural Network (DNN) architecture using PyTorch library to handle the multi-class classification task. The architecture will be designed with appropriate layers, activations, and regularization techniques.

## 5. Hyperparameter Tuning

GridSearch tool from sklearn library will be utilized to choose the best hyperparameters for the DNN model. This includes tuning parameters like learning rate, optimizers, number of epochs, and model architecture.

## 6. Visualization

Loss and accuracy metrics will be plotted against epochs for both training and test datasets. Interpretations will be provided based on the trends observed in the graphs.

![téléchargé](https://github.com/Sarah9515/Multi-Class-Cassification/assets/72395246/8106fd1e-a9e0-45cf-bb72-6425fc7a1e20)

![téléchargé (1)](https://github.com/Sarah9515/Multi-Class-Cassification/assets/72395246/07b37113-7cd3-4253-a8e4-763a1df7f49a)





## 7. Performance Metrics Calculation

Metrics such as accuracy, sensitivity, F1 score, etc., will be calculated on both training and test datasets to evaluate the model's performance.

![Screenshot 2024-03-10 194805](https://github.com/Sarah9515/Multi-Class-Cassification/assets/72395246/aed79f27-cf6d-4f53-a3d8-89d954152a87)


## 8. Regularization Techniques

Regularization techniques are pivotal in mitigating overfitting, a common issue in machine learning models where the algorithm fits too closely to the training data, leading to poor generalization on unseen data. These techniques introduce constraints to the model's optimization process, promoting simpler and more generalized solutions. L1 and L2 regularization, also known as Lasso and Ridge regularization, respectively, impose penalties on the model's coefficients during training, encouraging smaller weights and reducing the model's sensitivity to individual data points. Dropout, another widely used technique, randomly deactivates a fraction of neurons during training, forcing the network to learn redundant representations and enhancing its robustness. Regularization techniques play a crucial role in achieving better model performance, improving generalization, and ensuring models are better equipped to handle diverse datasets and real-world scenarios.

![téléchargé (5)](https://github.com/Sarah9515/Multi-Class-Cassification/assets/72395246/147f433e-5fa1-4bcc-b93d-e24d11bb7ba2)

![téléchargé (6)](https://github.com/Sarah9515/Multi-Class-Cassification/assets/72395246/f5d2f3d6-e404-4cd6-9b4d-50dab14b837f)

![Screenshot 2024-03-10 203817](https://github.com/Sarah9515/Multi-Class-Cassification/assets/72395246/1c71881d-37e9-4300-9ab8-b01853da4a9b)




By following these steps, we aim to develop an efficient deep learning model for multi-class classification on the Predictive Maintenance Dataset.
