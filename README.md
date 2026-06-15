# PRODIGY_ML_01
# House Price Prediction using Linear and Ridge Regression

## Project Overview

This project aims to predict residential house prices using machine learning techniques on the Ames Housing Dataset. The objective is to understand how different property characteristics influence house prices and build predictive models with increasing levels of sophistication.

The project is divided into two stages:

1. Basic House Price Prediction using selected numerical features.
2. Advanced House Price Prediction using feature engineering, one-hot encoding, feature selection, and regularized regression.
## Dataset

The dataset contains various attributes describing residential properties, including:

* Living area
* Number of bedrooms
* Number of bathrooms
* Garage size
* Construction year
* Neighborhood
* Overall quality
* Basement information
* Sale price

Target Variable:

* SalePrice – Property sale price in dollars

# Notebook 1: Basic House Price Prediction

## Features Used

```python
[
    'GrLivArea',
    'BedroomAbvGr',
    'FullBath',
    'OverallQual',
    'GarageArea',
    'YearBuilt'
]
```

## Workflow

* Data Cleaning
* Feature Selection
* Train-Test Split
* Linear Regression Model
* Model Evaluation

## Results

| Metric   | Value |
| -------- | ----- |
| R² Score | ~0.78 |

### Key Observations

* Larger living areas generally increase house prices.
* Overall quality strongly influences property value.
* Garage area and construction year contribute positively to predictions.

# Notebook 2: Advanced House Price Prediction

## Enhancements Performed

### Feature Engineering

New features were created to better represent the property characteristics:

* TotalBath
* HouseAge

### One-Hot Encoding

Categorical variables were converted into numerical format using:

```python
pd.get_dummies(drop_first=True)
```

### Correlation-Based Feature Selection

Top features were selected based on correlation with SalePrice.

### Models Implemented

1. Linear Regression
2. Ridge Regression
3. Hyperparameter Tuning using GridSearchCV

## Results

| Model                                    | R² Score |
| ---------------------------------------- | -------- |
| Linear Regression                        | 0.892    |
| Ridge Regression                         | 0.909    |
| 5-Fold Cross Validation                  | 0.900    |
| Tuned Ridge Regression (Best Alpha = 10) | 0.915    |

### Best Hyperparameter

```python
alpha = 10
```

## Techniques Used

* Data Preprocessing
* Missing Value Handling
* Feature Engineering
* One-Hot Encoding
* Correlation Analysis
* Linear Regression
* Ridge Regression
* Cross Validation
* Hyperparameter Tuning

## Libraries

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn

## Conclusion

The project demonstrates how systematic feature selection, feature engineering, and regularization techniques can significantly improve predictive performance.

Starting from a simple Linear Regression model, the final tuned Ridge Regression model achieved a cross-validated R² score of approximately 0.915, explaining over 91% of the variance in house prices.

The results highlight the importance of data preprocessing and feature engineering in building effective machine learning models.

## Future Improvements

* Lasso Regression
* Elastic Net Regression
* XGBoost Regressor
* Feature Importance Analysis
* Model Deployment using Streamlit
