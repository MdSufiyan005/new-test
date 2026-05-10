#!/usr/bin/env python
# coding: utf-8

# # Titanic Data Analysis
# 

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for plots
sns.set_theme(style="whitegrid")


# ## Question 1: Load and Inspect Dataset

# In[3]:


# 1. Load the file titanic_data.csv using Pandas.
df = pd.read_csv('titanic_data.csv')

# 2. Display:
# o Number of rows and columns (.shape)
print("Shape of the dataset:", df.shape)
print("-" * 40)

# o Column names (.columns)
print("Columns:", df.columns.tolist())
print("-" * 40)

# o Data types (.dtypes)
print("Data types:\n", df.dtypes)
print("-" * 40)

# o First 10 rows (.head())
display(df.head(10))


# ## Question 2: Identify Missing Values

# In[4]:


# 1. Display total missing values per column.
print("Total missing values per column:\n", df.isnull().sum())
print("-" * 40)

# 2. Check percentage of missing values for each column.
missing_percentage = (df.isnull().sum() / len(df)) * 100
print("Percentage of missing values per column:\n", missing_percentage)


# ## Question 3: Clean the Dataset

# In[5]:


# 1. Fill missing Age values using the median age.
median_age = df['Age'].median()
df['Age'] = df['Age'].fillna(median_age)

# 2. Fill missing Embarked values with the most frequent value.
most_frequent_embarked = df['Embarked'].mode()[0]
df['Embarked'] = df['Embarked'].fillna(most_frequent_embarked)

# 3. Drop any rows that still contain large amounts of missing data (only if necessary).
# 'Cabin' has a lot of missing values (~77%), so we can drop the column.
if 'Cabin' in df.columns:
    df = df.drop(columns=['Cabin'])

# Drop rows with any remaining missing values (e.g., in 'Embarked' or other columns)
df = df.dropna()

print("Missing values after cleaning:\n", df.isnull().sum())


# ## Question 4: Encode Categorical Features

# In[6]:


# Convert the following columns into numeric form using Label Encoding / Mapping:
# Sex -> (male = 0, female = 1)
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

# Embarked -> numeric categories
# Using factorization to assign numeric categories
df['Embarked'] = df['Embarked'].astype('category').cat.codes

display(df[['Sex', 'Embarked']].head())


# ## Question 5: Compute Summary Statistics

# In[7]:


# 1. Display Mean, Median, Standard deviation, Minimum and Maximum values
numeric_df = df.select_dtypes(include=[np.number])
summary_stats = numeric_df.describe().T
summary_stats['median'] = numeric_df.median()
display(summary_stats[['mean', '50%', 'std', 'min', 'max']].rename(columns={'50%': 'median'}))

print("-" * 40)

# 2. Compute survival counts:
survived_counts = df['Survived'].value_counts()
print(f"Survived (1): {survived_counts.get(1, 0)}")
print(f"Did not survive (0): {survived_counts.get(0, 0)}")


# ## Question 6: Grouped Survival Analysis

# In[8]:


# Analyze survival patterns:
# 1. Survival by gender
print("Survival by gender:")
display(df.groupby('Sex')['Survived'].mean().to_frame())

# 2. Survival by passenger class(Pclass)
print("\nSurvival by passenger class (Pclass):")
display(df.groupby('Pclass')['Survived'].mean().to_frame())

# 3. Survival by embarkation port
print("\nSurvival by embarkation port:")
display(df.groupby('Embarked')['Survived'].mean().to_frame())


# ## Question 7: Visualization – Survival Count

# In[9]:


# Create a count plot showing total survivors vs non-survivors.
plt.figure(figsize=(6, 4))
sns.countplot(x='Survived', data=df, palette='Set2')
plt.title('Total Survivors vs Non-Survivors')
plt.xlabel('Survived (0 = No, 1 = Yes)')
plt.ylabel('Count')
plt.show()


# ## Question 8: Visualization – Survival by Gender

# In[10]:


# Create a bar chart showing:
# Male survivors vs non-survivors
# Female survivors vs non-survivors
plt.figure(figsize=(6, 4))
sns.countplot(x='Survived', hue='Sex', data=df, palette='Set1')
plt.title('Survival by Gender')
plt.xlabel('Survived (0 = No, 1 = Yes)')
plt.ylabel('Count')
plt.legend(title='Sex (0=Male, 1=Female)')
plt.show()


# ## Question 9: Visualization – Survival vs Passenger Class

# In[11]:


# Create a bar plot showing survival rates across different classes: 1st, 2nd, 3rd class
plt.figure(figsize=(6, 4))
sns.barplot(x='Pclass', y='Survived', data=df, palette='viridis', errorbar=None)
plt.title('Survival Rate by Passenger Class')
plt.xlabel('Passenger Class (1st, 2nd, 3rd)')
plt.ylabel('Survival Rate')
plt.show()


# ## Question 10: Histogram of Age

# In[12]:


# Plot Age distribution of passengers
plt.figure(figsize=(8, 5))
sns.histplot(df['Age'], bins=30, kde=True, color='skyblue')
plt.title('Age Distribution of Passengers')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# Separate distributions for survived vs not survived (optional bonus)
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x='Age', hue='Survived', bins=30, kde=True, palette='coolwarm', alpha=0.6)
plt.title('Age Distribution: Survived vs Not Survived')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

