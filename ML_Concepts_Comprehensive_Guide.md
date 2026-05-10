# Comprehensive Machine Learning Concepts Guide

This guide compiles and summarizes all the core machine learning concepts explored across your assignments. It serves as a comprehensive study sheet covering data preprocessing, classification algorithms, ensemble methods, dimensionality reduction, and unsupervised clustering.

---

## 1. Classification Evaluation Metrics
*Explored in the Adult Census Income Assignment.*

When evaluating classification models, accuracy alone is often misleading—especially when dealing with imbalanced datasets.

- **Accuracy**: The ratio of correctly predicted observations to total observations. It fails when one class dominates (e.g., predicting everyone earns `<=50K` yields 75% accuracy but is useless).
- **Precision**: Of all positive predictions, how many were actually positive? ($TP / (TP + FP)$). High precision is required when false positives are costly (e.g., targeting high-income earners for luxury goods).
- **Recall (Sensitivity)**: Of all actual positive instances, how many did the model find? ($TP / (TP + FN)$). High recall is required when false negatives are costly (e.g., detecting cancer).
- **F1-Score**: The harmonic mean of Precision and Recall. It provides a balanced metric when classes are uneven.
- **ROC (Receiver Operating Characteristic) & AUC**: ROC plots the True Positive Rate vs False Positive Rate at various thresholds. **AUC (Area Under Curve)** represents the probability that the model ranks a random positive instance higher than a random negative one. An AUC of 0.5 is random guessing; 1.0 is perfect.
- **K-Fold Cross-Validation**: Splitting the dataset into $K$ folds, training on $K-1$ folds, and testing on the remaining fold $K$ times. This ensures the model's evaluation is not dependent on a "lucky" train-test split, providing a more reliable estimate of generalization.

---

## 2. Tree-Based Models & Ensembles
*Explored in the Titanic Dataset Assignment.*

### Decision Trees
A Decision Tree splits data based on feature thresholds to maximize information gain (or minimize Gini impurity).
- **Overfitting**: An unpruned decision tree will grow until every leaf is pure (depth goes very high, e.g., 80+ levels). It essentially memorizes the training data, leading to poor test performance.
- **Pruning (Depth Control)**: Limiting `max_depth` (e.g., to 3 or 4) forces the tree to learn generalized rules rather than hyper-specific exceptions, improving test accuracy.
- **Interpretability**: Decision Trees are highly interpretable white-box models. You can visually trace the logic from root to leaf.

### Random Forest
Random Forest is an **ensemble** method that trains multiple decision trees and averages their predictions (majority voting for classification).
- **Why it reduces overfitting**: It uses **Bagging** (Bootstrap Aggregating) by training each tree on a random subset of data, and considering a random subset of features at each split. This de-correlates the trees, lowering the overall model variance.
- **Feature Importance**: Random Forests can calculate how much each feature contributes to decreasing impurity across all trees (e.g., `sex`, `fare`, and `age` were most critical for Titanic survival).

---

## 3. Support Vector Machines (SVM), Logistic Regression & k-NN
*Explored in the Iris and Wine Datasets Assignment.*

### Support Vector Machines (SVM)
SVM aims to find the optimal hyperplane that maximizes the margin between classes.
- **Support Vectors**: These are the data points closest to the decision boundary. They dictate the position of the hyperplane; removing all other points wouldn't change the boundary.
- **Linear Kernel**: Finds a straight-line (or flat plane) separation. Fast and effective for linearly separable data.
- **RBF Kernel (Radial Basis Function)**: Uses the **Kernel Trick** to project data into a higher-dimensional space implicitly, allowing the SVM to find non-linear decision boundaries. Essential for complex, intertwining classes.
- **When to use**: Great for high-dimensional data (text classification, gene expression).
- **When NOT to use**: Struggles with massive datasets (high training time $O(n^3)$) and highly noisy data with overlapping classes.

### Logistic Regression & k-NN
- **Logistic Regression**: A linear model that outputs probabilities using the sigmoid function. Highly interpretable, computationally cheap, and great as a baseline.
- **k-Nearest Neighbors (k-NN)**: A non-parametric, lazy-learning algorithm that classifies a point based on the majority vote of its $k$ closest neighbors. Highly sensitive to the choice of $k$ (small $k$ = overfitting/noise-sensitive, large $k$ = underfitting).

---

## 4. Dimensionality Reduction (PCA)
*Explored in the Fashion-MNIST Assignment.*

**Principal Component Analysis (PCA)** is an unsupervised technique used to reduce the number of features while retaining as much variance (information) as possible.

- **Normalization**: PCA is heavily affected by scale because it maximizes variance. You must standardize/normalize data (e.g., pixel values to 0-1 range) so features with naturally larger scales don't dominate the components.
- **Vectors required**: PCA calculates the covariance matrix between features, requiring 2D images to be flattened into 1D vectors (e.g., 28x28 -> 784).
- **Explained Variance**: The percentage of information captured by a principal component. Often, you can capture 90-95% of the variance using only ~10-20% of the original features.
- **Benefits**: Drastic reduction in memory footprint and massive speedups in training time for classifiers (like SVM or Logistic Regression) while suffering negligible accuracy drops.
- **Image Reconstruction**: Converting PCA components back to the original space results in a loss of fine, high-frequency details (making images slightly blurry), but the core semantic structure remains.

---

## 5. Unsupervised Clustering
*Explored in the Mall Customer Segmentation Assignment.*

Clustering algorithms group unlabeled data based on feature similarity.

### K-Means Clustering
- **How it works**: Assigns points to the nearest centroid, then updates centroids to the mean of the points, repeating until convergence.
- **Elbow Method**: Used to find the optimal $k$. You plot WCSS (Within-Cluster Sum of Squares) against $k$. The "elbow" point where the WCSS drop slows down significantly represents the optimal number of clusters.
- **Assumptions**: Assumes clusters are spherical and roughly equal in size. It is highly sensitive to outliers because it is forced to assign every single point to a cluster.

### DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
- **How it works**: Groups together points that are closely packed together (many neighbors), marking points in low-density regions as outliers.
- **Parameters**: `eps` (neighborhood radius) and `min_samples` (minimum points to form a dense region).
- **Advantages over K-Means**: 
  1. It does **not** require specifying the number of clusters beforehand.
  2. It can find arbitrarily shaped clusters (e.g., moons, rings).
  3. It explicitly identifies and handles **noise/outliers**, assigning them a label of `-1` rather than dragging centroids outward.

---

### Summary Checklist for Model Selection:
- **Need baseline probabilities?** $\rightarrow$ Logistic Regression
- **Need explainable rules?** $\rightarrow$ Decision Tree
- **Need robust tabular classification?** $\rightarrow$ Random Forest or Gradient Boosting
- **High dimensions, clear margins?** $\rightarrow$ SVM
- **Data is massive, need to reduce features?** $\rightarrow$ PCA
- **Spherical, distinct groups to find?** $\rightarrow$ K-Means
- **Oddly shaped groups with noise?** $\rightarrow$ DBSCAN
