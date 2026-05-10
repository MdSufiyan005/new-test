#!/usr/bin/env python
# coding: utf-8

# # Data Visualization with Iris and Penguins Datasets
# This notebook performs data analysis and visualization on the Iris and Penguins datasets using Matplotlib and Seaborn.
# 

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for plots
sns.set_theme(style="whitegrid")


# ## Question 1 — Load and Inspect

# In[2]:


# Load the datasets from seaborn
iris = sns.load_dataset('iris')
penguins = sns.load_dataset('penguins')

print("--- Iris Dataset ---")
print("First 5 rows:")
display(iris.head())
print("\nColumn names:", iris.columns.tolist())
print(f"Number of rows and columns: {iris.shape}")

print("\n--- Penguins Dataset ---")
print("First 5 rows:")
display(penguins.head())
print("\nColumn names:", penguins.columns.tolist())
print(f"Number of rows and columns: {penguins.shape}")


# ## Question 2 — Class Distribution

# In[3]:


# Create a bar chart showing the count of each species for both datasets.
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Iris species distribution
sns.countplot(x='species', data=iris, ax=axes[0], palette='Set2')
axes[0].set_title('Iris Species Distribution')
axes[0].set_ylabel('Count')

# Penguins species distribution
sns.countplot(x='species', data=penguins, ax=axes[1], palette='Set1')
axes[1].set_title('Penguins Species Distribution')
axes[1].set_ylabel('Count')

plt.tight_layout()
plt.show()


# ## Question 3 — Feature Distribution

# In[4]:


# Create histograms for at least two numerical features.
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Histogram for Iris Sepal Length
sns.histplot(iris['sepal_length'], kde=True, ax=axes[0], color='skyblue')
axes[0].set_title('Distribution of Sepal Length (Iris)')

# Histogram for Penguins Body Mass
sns.histplot(penguins['body_mass_g'].dropna(), kde=True, ax=axes[1], color='lightcoral')
axes[1].set_title('Distribution of Body Mass (Penguins)')

plt.tight_layout()
plt.show()


# ## Question 4 — Scatter Plot

# In[5]:


# Plot a scatter graph showing sepal_length vs sepal_width. Color points by species.
plt.figure(figsize=(8, 6))
sns.scatterplot(x='sepal_length', y='sepal_width', hue='species', data=iris, palette='deep', s=100)
plt.title('Sepal Length vs Sepal Width (Iris)')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.legend(title='Species')
plt.show()


# ## Question 5 — Pair Plot

# In[6]:


# Create a pairplot to visualize all feature relationships.
# For Iris dataset
print("Pairplot for Iris dataset:")
sns.pairplot(iris, hue='species', palette='husl')
plt.show()

# For Penguins dataset
print("Pairplot for Penguins dataset:")
sns.pairplot(penguins, hue='species', palette='tab10')
plt.show()


# ## Question 6 — Heatmap

# In[7]:


# Create a correlation heatmap and identify which features are strongly related.
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Iris Correlation Heatmap
iris_numeric = iris.select_dtypes(include=[np.number])
sns.heatmap(iris_numeric.corr(), annot=True, cmap='coolwarm', ax=axes[0])
axes[0].set_title('Correlation Heatmap (Iris)')

# Penguins Correlation Heatmap
penguins_numeric = penguins.select_dtypes(include=[np.number])
sns.heatmap(penguins_numeric.corr(), annot=True, cmap='viridis', ax=axes[1])
axes[1].set_title('Correlation Heatmap (Penguins)')

plt.tight_layout()
plt.show()

print("Observations:")
print("- In the Iris dataset, petal length and petal width are strongly positively correlated (0.96).")
print("- In the Penguins dataset, flipper length and body mass are strongly positively correlated (0.87).")


# ## Question 7 — Box Plot

# In[8]:


# Draw boxplots for numerical features to observe spread and outliers.
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Boxplot for Iris dataset
sns.boxplot(data=iris, orient='h', ax=axes[0], palette='Set2')
axes[0].set_title('Boxplot of Iris Features')

# Boxplot for Penguins dataset
# Normalizing or plotting specific features due to scale differences
penguins_subset = penguins[['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm']]
sns.boxplot(data=penguins_subset, orient='h', ax=axes[1], palette='Set3')
axes[1].set_title('Boxplot of Penguin Measurements')

plt.tight_layout()
plt.show()

# Boxplot for Penguins body mass separately due to scale
plt.figure(figsize=(8, 4))
sns.boxplot(x=penguins['body_mass_g'], color='orange')
plt.title('Boxplot of Penguins Body Mass (g)')
plt.show()


# ## Question 8 — Violin Plot

# In[9]:


# Create violin plots comparing feature distribution across species.
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Violin plot for Iris (Petal Length across Species)
sns.violinplot(x='species', y='petal_length', data=iris, ax=axes[0], palette='muted', inner='quartile')
axes[0].set_title('Iris: Petal Length Distribution by Species')

# Violin plot for Penguins (Flipper Length across Species)
sns.violinplot(x='species', y='flipper_length_mm', data=penguins, ax=axes[1], palette='pastel', inner='quartile')
axes[1].set_title('Penguins: Flipper Length Distribution by Species')

plt.tight_layout()
plt.show()


# ## Question 9 — Compare Datasets

# In[10]:


# Visualize and compare one feature from Iris vs Penguins datasets (for example: body mass vs flipper length).
# We will visualize body mass vs flipper length for Penguins as requested by the example.
plt.figure(figsize=(8, 6))
sns.scatterplot(x='flipper_length_mm', y='body_mass_g', hue='species', data=penguins, s=100, palette='viridis')
plt.title('Penguins: Body Mass vs Flipper Length')
plt.xlabel('Flipper Length (mm)')
plt.ylabel('Body Mass (g)')
plt.show()

print("Explanation:")
print("The scatter plot shows a strong positive correlation between flipper length and body mass in penguins. As flipper length increases, body mass also tends to increase. Furthermore, the species cluster nicely: Gentoo penguins are clearly larger and heavier than Adelie and Chinstrap penguins.")


# ## Question 10 — Summary Interpretation

# In[11]:


text_summary = """
### Summary Interpretation

**1. What did you observe?**
- In the Iris dataset, petal length and petal width are excellent features for distinguishing species, specifically separating *setosa* from the others.
- In the Penguins dataset, *Gentoo* penguins are noticeably larger (higher body mass and flipper length) than *Adelie* and *Chinstrap*.
- There are strong positive correlations between physical dimensions in both datasets (e.g., petal length & width in Iris; flipper length & body mass in Penguins).

**2. Which plot explained data best?**
- The **Pair Plot** was highly effective as it provided a comprehensive overview of all numerical feature relationships at once, clearly highlighting how species cluster across different dimensions.
- The **Scatter Plot with hue** was also excellent for visualizing the distinct separation of classes based on two highly correlated variables.

**3. How visualization helped?**
- Visualization instantly revealed patterns that would be difficult to spot by looking at raw numbers alone.
- It helped identify correlations, outliers, and the distribution spread.
- It made classifying and distinguishing species visually intuitive.
"""
from IPython.display import Markdown
display(Markdown(text_summary))

