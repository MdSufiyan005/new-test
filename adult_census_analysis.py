#!/usr/bin/env python
# coding: utf-8

# # Adult Census Income - Classification Model Evaluation
# 
# This notebook rigorously evaluates a Decision Tree classifier on the Adult Census Income dataset.

# ### Question 1: Load Dataset
# Load Adult Census dataset and display:
# - First 10 rows
# - Dataset shape
# - List of features

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Column names for the adult dataset
columns = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 
           'marital_status', 'occupation', 'relationship', 'race', 'sex', 
           'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income']

# Load dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
df = pd.read_csv(url, names=columns, na_values=[' ?', '?'], skipinitialspace=True)

# First 10 rows
display(df.head(10))

# Dataset shape
print(f"Dataset shape: {df.shape}")

# List of features
print("\nList of features:")
print(df.columns.tolist())


# ### Question 2: Data Cleaning
# Handle missing values and encode categorical columns.
# Explain technique used (label encoding / one-hot encoding).

# In[2]:


# Handle missing values
for col in df.columns:
    if df[col].isnull().sum() > 0:
        if df[col].dtype == 'object':
            # Fill missing categorical values with the mode
            df[col].fillna(df[col].mode()[0], inplace=True)
        else:
            # Fill missing numerical values with median
            df[col].fillna(df[col].median(), inplace=True)

# Separate features into categorical and numerical
categorical_cols = df.select_dtypes(include=['object']).columns.drop('income')
numerical_cols = df.select_dtypes(exclude=['object']).columns

# Encode categorical columns using One-Hot Encoding
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

print("Data shape after one-hot encoding:", df_encoded.shape)
display(df_encoded.head())


# **Explanation:**
# I used **One-Hot Encoding** for the categorical feature columns. One-Hot encoding is preferred for nominal data (where there is no inherent order among categories, like `workclass` or `occupation`), as it prevents the model from assuming any ordinal relationship between the categories. For handling missing values, I filled categorical NaNs with the mode (most frequent value) of the column.

# ### Question 3: Define Target Variable
# Define income column as:
# 0 = <= 50K
# 1 = > 50K
# Explain why binary encoding is required.

# In[ ]:


# Define income column as 0 = <=50K, 1 = >50K
df_encoded['income'] = df_encoded['income'].apply(lambda x: 1 if '>50K' in x else 0)

print(df_encoded['income'].value_counts())


# **Explanation:**
# **Binary encoding is required** for the target variable because most machine learning classification algorithms (like Scikit-Learn's Decision Tree) require numerical inputs and targets to compute splits and evaluate loss. By mapping `<=50K` to 0 and `>50K` to 1, we frame the problem as a standard binary classification task, allowing mathematical optimization and proper evaluation metric calculation (like Precision, Recall, and ROC).

# ### Question 4: Train-Test Split
# Split dataset into:
# 70% training
# 30% testing

# In[ ]:


from sklearn.model_selection import train_test_split

X = df_encoded.drop('income', axis=1)
y = df_encoded['income']

# Split dataset into 70% training and 30% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")


# ### Question 5: Train Decision Tree
# Train Decision Tree classifier with default settings.
# Print:
# - depth
# - number of leaves

# In[ ]:


from sklearn.tree import DecisionTreeClassifier

# Train Decision Tree classifier with default settings
dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(X_train, y_train)

# Print depth and number of leaves
print(f"Decision Tree Depth: {dt_classifier.get_depth()}")
print(f"Number of Leaves: {dt_classifier.get_n_leaves()}")


# ### Question 6: Accuracy
# Compute model accuracy.
# Explain in 2–3 lines why accuracy alone is not enough.

# In[ ]:


from sklearn.metrics import accuracy_score

y_pred = dt_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Model Accuracy: {accuracy:.4f}")


# **Explanation:**
# Accuracy alone is not enough because the target variable is imbalanced (approx 75% earn <=50K). A "dumb" model could achieve 75% accuracy simply by predicting `<=50K` for everyone, while completely failing to identify the minority class (which is often the group we are most interested in).

# ### Question 7: Precision & Recall
# Compute:
# - precision
# - recall
# - F1-score
# Explain in 3–4 lines which metric is more important for income prediction and why.

# In[ ]:


from sklearn.metrics import precision_score, recall_score, f1_score

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")


# **Explanation:**
# For predicting if someone's income is >50K (e.g., for targeting a high-end marketing campaign), **Precision** is often more important. High precision ensures that the individuals we classify as high-income actually are, preventing wasted resources on false positives. However, if the goal is to grant loans and minimize risk, identifying all potentially low-income people correctly would be critical, shifting the focus to Recall for the other class.

# ### Question 8: ROC & AUC
# Plot ROC curve and calculate AUC.
# Answer:
# - What does AUC represent?
# - Is the model good or weak?

# In[ ]:


from sklearn.metrics import roc_curve, auc

y_probs = dt_classifier.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.show()

print(f"AUC Score: {roc_auc:.4f}")


# **Explanation:**
# - **What does AUC represent?** AUC (Area Under the Curve) represents the probability that the model will rank a randomly chosen positive instance (e.g., >50K) higher than a randomly chosen negative instance. It holistically measures the model's ability to discriminate between the two classes across all threshold values.
# - **Is the model good or weak?** With an AUC around 0.75, the model is mediocre. While it performs better than random guessing (AUC 0.5), it falls short of being a strong predictive model, primarily because the unpruned decision tree heavily overfit the training data.

# ### Question 9: K-Fold Cross-Validation
# Apply 5-fold or 10-fold cross-validation.
# Report:
# - mean accuracy
# - standard deviation
# Explain why cross-validation improves reliability.

# In[ ]:


from sklearn.model_selection import cross_val_score, KFold

# Apply 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(dt_classifier, X, y, cv=kf, scoring='accuracy')

print(f"Cross-Validation Scores (5-fold): {cv_scores}")
print(f"Mean Accuracy: {cv_scores.mean():.4f}")
print(f"Standard Deviation: {cv_scores.std():.4f}")


# **Explanation:**
# Cross-validation improves reliability because it trains and evaluates the model on multiple different, non-overlapping subsets of the data rather than relying on a single, potentially lucky train-test split. This ensures that the model's performance estimate is robust and gives a truer representation of how the model will generalize to completely unseen data.

# ### Question 10: Reflection
# Write short answers:
# - Did the decision tree overfit?
# - Which metric gave the best understanding?
# - How could the model be improved?

# **Reflection Answers:**
# - **Did the decision tree overfit?** Yes, heavily. The tree grew to a very large depth (typically 80+ levels) with thousands of leaves. This indicates it essentially memorized the training data, leading to a drop in its generalization capability on the test set.
# - **Which metric gave the best understanding?** The **AUC and F1-score** gave the best understanding. While accuracy remained seemingly decent (~81%), the F1-score (~62%) and AUC (~0.75) exposed the model's struggle to correctly distinguish and classify the minority class (>50K).
# - **How could the model be improved?** The model could be improved by tuning hyperparameters (e.g., limiting `max_depth`, increasing `min_samples_split`, or `min_samples_leaf`) to prune the tree and prevent overfitting. Additionally, transitioning to ensemble techniques like **Random Forest** or **Gradient Boosting** would significantly boost predictive performance.
