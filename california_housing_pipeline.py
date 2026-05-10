#!/usr/bin/env python
# coding: utf-8

# # California Housing Machine Learning Pipeline
# This notebook develops a complete machine learning pipeline for regression using the California Housing dataset.
# 

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Set style for plots
sns.set_theme(style="whitegrid")


# ## Question 1 — Load Dataset

# In[3]:


# Load the California Housing dataset
california = fetch_california_housing()

# Print feature names, target name, number of samples and features
print("Feature names:", california.feature_names)
print("Target name:", california.target_names)
print(f"Number of samples: {california.data.shape[0]}")
print(f"Number of features: {california.data.shape[1]}")


# ## Question 2 — Convert to DataFrame

# In[4]:


# Convert data into a Pandas DataFrame and display first 10 rows
df = pd.DataFrame(california.data, columns=california.feature_names)
df['MedHouseVal'] = california.target

display(df.head(10))


# ## Question 3 — Check Missing Values

# In[5]:


# Check for missing values and report findings
missing_values = df.isnull().sum()
print("Missing values per column:\n")
print(missing_values)

print("\nFindings:")
print("The dataset has no missing values initially. However, we will demonstrate the imputation step in the next question as requested.")


# ## Question 4 — Handle Missing Values

# In[6]:


# Impute missing values using: SimpleImputer(strategy="median")
# Even though there are no missing values, we apply the imputer as requested to build the pipeline
X = df.drop(columns=['MedHouseVal'])
y = df['MedHouseVal']

imputer = SimpleImputer(strategy="median")
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

print("Imputation completed. Shape of imputed features:", X_imputed.shape)


# ## Question 5 — Train-Test Split

# In[7]:


# Split dataset into: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

print(f"Training set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")


# ## Question 6 — Feature Scaling

# In[8]:


# Apply StandardScaler
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

display(X_train_scaled.head())

text_scaling = """
**Why scaling is needed:**
Linear Regression and many other machine learning algorithms use gradient descent for optimization or rely on distance metrics. If features have different scales (e.g., 'HouseAge' ranges from 1-50 while 'Population' ranges from 1-35000), the model will be biased towards features with larger values, and optimization will converge much slower. StandardScaler standardizes features by removing the mean and scaling to unit variance.
"""
from IPython.display import Markdown
display(Markdown(text_scaling))


# ## Question 7 — Train Model

# In[9]:


# Train Linear Regression on the processed dataset
model = LinearRegression()
model.fit(X_train_scaled, y_train)

print("Linear Regression model trained successfully.")
print("Model coefficients:", model.coef_)


# ## Question 8 — Model Evaluation

# In[10]:


# Evaluate model using MSE, RMSE, R² Score
y_pred = model.predict(X_test_scaled)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R² Score: {r2:.4f}")

text_interpretation = """
**Interpretation:**
The R² score of ~0.57 indicates that our model explains approximately 57% of the variance in California house prices based on the provided features. The RMSE of ~0.74 means that our predictions are, on average, about $74,000 away from the true median house values (since the target is in units of $100,000s). While this provides a reasonable baseline, there is still significant room for improvement, suggesting that a more complex model or better feature engineering could yield better performance.
"""
display(Markdown(text_interpretation))


# ## Question 9 — Save Predictions

# In[11]:


# Generate predictions on the test set and save them into predictions.csv
predictions_df = pd.DataFrame({'Actual_MedHouseVal': y_test, 'Predicted_MedHouseVal': y_pred})
predictions_df.to_csv('predictions.csv', index=False)

print("Predictions successfully saved to 'predictions.csv'.")
display(predictions_df.head())


# ## Question 10 — Reflection

# In[12]:


text_reflection = """
### Reflection

**Which pipeline step was most useful?**
Feature Scaling was highly useful. It brought features with vastly different scales (like population, income, and house age) into a common range, ensuring the Linear Regression algorithm could weigh them appropriately without being biased by features with larger magnitudes.

**Where could errors occur?**
Errors commonly occur during data scaling and splitting if we accidentally scale the test set using `fit_transform` instead of `transform`, which causes data leakage. Errors could also occur in imputation if a production dataset has missing values in columns not expected during training.

**How could accuracy be improved?**
Accuracy could be improved by:
1. Using more powerful non-linear models like Random Forest, XGBoost, or Neural Networks.
2. Feature engineering (e.g., combining rooms and bedrooms to calculate bedrooms per room).
3. Handling spatial coordinates properly (e.g., clustering based on Latitude and Longitude instead of treating them as regular numerical features).
"""
display(Markdown(text_reflection))

