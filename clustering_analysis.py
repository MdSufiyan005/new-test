#!/usr/bin/env python
# coding: utf-8

# # Clustering Analysis - K-Means & DBSCAN
# 
# This notebook applies unsupervised learning techniques (K-Means and DBSCAN) to the Mall Customer Segmentation dataset and the Iris dataset.

# ### Question 1: Load Dataset
# Load Mall dataset.
# Print:
# - first 10 rows
# - description of variables

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Load Mall Customer Segmentation Dataset
url = "https://raw.githubusercontent.com/tirthajyoti/Machine-Learning-with-Python/master/Datasets/Mall_Customers.csv"
mall_df = pd.read_csv(url)

# Display first 10 rows
display(mall_df.head(10))

# Description of variables
print("\nVariable Descriptions:")
print("CustomerID: Unique ID assigned to the customer")
print("Gender: Gender of the customer")
print("Age: Age of the customer")
print("Annual Income (k$): Annual Income of the customer in thousands of dollars")
print("Spending Score (1-100): Score assigned by the mall based on customer behavior and spending nature")


# ### Question 2: Feature Selection & Scaling
# Choose meaningful variables: Annual Income, Spending Score.
# Scale features using StandardScaler.

# In[ ]:


from sklearn.preprocessing import StandardScaler

# Select Annual Income and Spending Score
X_mall = mall_df[['Annual Income (k$)', 'Spending Score (1-100)']].values

# Scale features
scaler = StandardScaler()
X_mall_scaled = scaler.fit_transform(X_mall)

print(f"Original Data (first 5 rows):\n{X_mall[:5]}\n")
print(f"Scaled Data (first 5 rows):\n{X_mall_scaled[:5]}")


# ### Question 3: Apply K-Means
# Run K-Means with default clusters (k=4).
# Plot scatter with cluster colors.

# In[ ]:


from sklearn.cluster import KMeans

# Run K-Means with k=4
kmeans_4 = KMeans(n_clusters=4, random_state=42, n_init=10)
y_kmeans_4 = kmeans_4.fit_predict(X_mall_scaled)

# Plot scatter
plt.figure(figsize=(8, 5))
plt.scatter(X_mall_scaled[:, 0], X_mall_scaled[:, 1], c=y_kmeans_4, cmap='viridis', s=50, alpha=0.8)
plt.scatter(kmeans_4.cluster_centers_[:, 0], kmeans_4.cluster_centers_[:, 1], s=200, c='red', label='Centroids', marker='X')
plt.title('K-Means Clustering (k=4)')
plt.xlabel('Annual Income (Scaled)')
plt.ylabel('Spending Score (Scaled)')
plt.legend()
plt.grid(alpha=0.3)
plt.show()


# ### Question 4: Determine Optimal K (Elbow Method)
# Plot WCSS vs k.
# Explain why elbow occurs.

# In[ ]:


wcss = []
k_values = range(1, 11)

for i in k_values:
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42, n_init=10)
    kmeans.fit(X_mall_scaled)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(k_values, wcss, marker='o', linestyle='--', color='b')
plt.title('The Elbow Method')
plt.xlabel('Number of clusters (k)')
plt.ylabel('WCSS (Within-Cluster Sum of Squares)')
plt.xticks(k_values)
plt.grid(alpha=0.3)
plt.show()


# **Explanation:**
# The 'elbow' occurs because adding more clusters initially yields a large reduction in WCSS (since points are much closer to their new centroids). However, after a certain point (in this case, k=5), the WCSS drops marginally because the intrinsic groups in the data have already been found, and further clusters are just subdividing meaningful groups arbitrarily. The optimal k is at the hinge of the elbow, which is **k=5** here.

# ### Question 5: Interpret Mall Clusters
# Describe clusters (e.g., high income-high spend, low income-low spend, etc.) using k=5.

# In[ ]:


# Run K-Means with the optimal k=5
kmeans_5 = KMeans(n_clusters=5, random_state=42, n_init=10)
y_kmeans_5 = kmeans_5.fit_predict(X_mall) # Plotting on original scale for interpretation

plt.figure(figsize=(9, 6))
colors = ['red', 'blue', 'green', 'cyan', 'magenta']
labels = ['Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4', 'Cluster 5']

for i in range(5):
    plt.scatter(X_mall[y_kmeans_5 == i, 0], X_mall[y_kmeans_5 == i, 1], s=50, c=colors[i], label=labels[i])

plt.scatter(kmeans_5.cluster_centers_[:, 0], kmeans_5.cluster_centers_[:, 1], s=200, c='black', marker='X', label='Centroids')
plt.title('Mall Customer Segments (k=5)')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(alpha=0.3)
plt.show()


# **Interpretation:**
# Based on the k=5 clustering (on original axes):
# - **High Income - High Spend:** These are the mall's best customers. They earn a lot and spend a lot (prime target for luxury marketing).
# - **High Income - Low Spend:** These customers earn well but are frugal. They might need discounts or high-value propositions to spend.
# - **Low Income - High Spend:** Careless spenders. They have low income but spend heavily (often a target for credit card offers or fast fashion).
# - **Low Income - Low Spend:** Sensible shoppers. They have low income and spend cautiously.
# - **Moderate Income - Moderate Spend:** The average middle-class shopper. This is the largest, most dense cluster in the middle.

# ### Question 6: Apply DBSCAN
# Use DBSCAN parameters (eps, min_samples).
# Discuss noisy / outlier points.

# In[ ]:


from sklearn.cluster import DBSCAN

# Apply DBSCAN on scaled data
dbscan = DBSCAN(eps=0.35, min_samples=5)
y_dbscan = dbscan.fit_predict(X_mall_scaled)

num_clusters = len(set(y_dbscan)) - (1 if -1 in y_dbscan else 0)
num_noise = list(y_dbscan).count(-1)

print(f"Estimated number of clusters: {num_clusters}")
print(f"Estimated number of noise points: {num_noise}")


# **Explanation:**
# DBSCAN requires two parameters: `eps` (the maximum distance between two samples for them to be considered in the same neighborhood) and `min_samples` (the number of samples in a neighborhood for a point to be considered a core point). Points that do not belong to any cluster's neighborhood are labeled as **-1 (noise or outliers)**. These represent customers whose behavior doesn't tightly align with any main group.

# ### Question 7: Visualize DBSCAN Clusters
# Plot clusters and noise separately.
# Observe differences vs K-Means.

# In[ ]:


plt.figure(figsize=(8, 5))

# Plot core samples
core_mask = y_dbscan != -1
plt.scatter(X_mall_scaled[core_mask, 0], X_mall_scaled[core_mask, 1], c=y_dbscan[core_mask], cmap='Set1', s=50, label='Clusters')

# Plot noise samples (-1)
noise_mask = y_dbscan == -1
plt.scatter(X_mall_scaled[noise_mask, 0], X_mall_scaled[noise_mask, 1], c='black', s=20, marker='x', label='Noise (-1)')

plt.title('DBSCAN Clustering (eps=0.35, min_samples=5)')
plt.xlabel('Annual Income (Scaled)')
plt.ylabel('Spending Score (Scaled)')
plt.legend()
plt.grid(alpha=0.3)
plt.show()


# **Explanation:**
# Unlike K-Means, which forces every single point into a cluster (sometimes resulting in unnatural boundaries), DBSCAN identified the dense regions as clusters and marked the sparsely populated areas between them as noise (the black 'x' marks). This prevents outliers from artificially dragging cluster centroids away from the true center of mass.

# ### Question 8: Apply Clustering on Iris Dataset
# Cluster using K-Means and DBSCAN.
# Compare with true labels using cluster purity or confusion matrix.

# In[ ]:


from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix

iris = load_iris()
X_iris = iris.data
y_iris_true = iris.target

# Scale Iris
scaler_iris = StandardScaler()
X_iris_scaled = scaler_iris.fit_transform(X_iris)

# K-Means
kmeans_iris = KMeans(n_clusters=3, random_state=42, n_init=10)
y_iris_kmeans = kmeans_iris.fit_predict(X_iris_scaled)

# DBSCAN
dbscan_iris = DBSCAN(eps=0.7, min_samples=4)
y_iris_dbscan = dbscan_iris.fit_predict(X_iris_scaled)

print("--- K-Means vs True Labels ---")
print(confusion_matrix(y_iris_true, y_iris_kmeans))

print("\n--- DBSCAN vs True Labels ---")
print(confusion_matrix(y_iris_true, y_iris_dbscan))


# **Explanation:**
# - **K-Means Matrix:** K-Means accurately isolates class 0 (Setosa) perfectly, but there is some confusion between class 1 and class 2 (Versicolor and Virginica), which overlap in the feature space.
# - **DBSCAN Matrix:** DBSCAN isolates class 0 perfectly into its own cluster, but it entirely merged class 1 and 2 into a single massive cluster, while labeling boundary points as noise (-1). Because DBSCAN joins all dense points together, it struggles when clusters touch or overlap without a drop in density.

# ### Question 9: Comparison Table
# Compare K-Means and DBSCAN.

# In[ ]:


comp_data = {
    "Feature": ["Shape Assumption", "Parameter Required", "Handling Outliers", "Deterministic?", "Use Case"],
    "K-Means": ["Assumes spherical clusters", "k (number of clusters)", "Sensitive to outliers (forces them into clusters)", "Yes (with fixed random seed)", "Evenly sized, spherical clusters"],
    "DBSCAN": ["Can find arbitrarily shaped clusters", "eps, min_samples", "Robust (labels them as noise)", "Yes", "Clusters of varying shapes with noise"]
}

display(pd.DataFrame(comp_data))


# ### Question 10: Reflection
# Explain in 4–5 lines:
# - Which algorithm worked better and why?
# - When would you prefer DBSCAN over K-Means?

# **Reflection Answers:**
# - **Which algorithm worked better and why?** For the Mall Customers dataset, **K-Means** worked slightly better because the natural clusters in the income/spending space are relatively spherical and evenly sized, aligning perfectly with K-Means' assumptions. For the Iris dataset, K-Means also performed better because it was forced to split the overlapping Versicolor/Virginica groups, whereas DBSCAN just merged them into one giant blob.
# - **When would you prefer DBSCAN over K-Means?** DBSCAN is strongly preferred when your data has **arbitrary, non-spherical shapes** (like moons or rings), when the data contains many **noisy outliers** that you wish to ignore rather than force into a cluster, or when you **do not know how many clusters** exist beforehand.
