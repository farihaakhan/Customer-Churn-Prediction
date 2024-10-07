# Customer-Churn-Prediction

This project focuses on predicting customer churn in the telecommunications sector, leveraging machine learning techniques. Churn refers to the process of customers discontinuing their services with a company. Predicting churn helps telecom providers understand customer behavior and improve retention strategies. 

## Project Overview

The goal of this project is to predict customer churn using two machine learning models: Logistic Regression and Random Forest. The dataset used in this analysis consists of customer data from four major telecom companies.

### Key Steps:
1. **Data Loading and Merging**: Merged two separate datasets into a single DataFrame.
2. **Churn Rate Calculation**: Calculated the churn rate to understand the distribution of the target variable.
3. **Feature Engineering**: Categorical variables were identified and converted using one-hot encoding. Feature scaling was applied to numerical data.
4. **Model Training**: Trained Logistic Regression and Random Forest classifiers on the processed data.
5. **Model Evaluation**: Compared model performance using accuracy, confusion matrix, and classification report.

## Results

The project compares the accuracy of both models to identify which algorithm better predicts customer churn. 

- **Logistic Regression Accuracy**: 73% (rounded)
- **Random Forest Accuracy**: 79% (rounded)
- Model with higher accuracy: `Random Forest` 

## Technologies Used

- **Python**: Language used for the analysis.
- **Pandas**: Data manipulation and cleaning.
- **Scikit-learn**: Machine learning algorithms and data preprocessing.
- **Jupyter Notebook**: For executing and visualizing code.
