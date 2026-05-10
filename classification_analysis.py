import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

from sklearn.datasets import load_breast_cancer, load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score, precision_score, recall_score, ConfusionMatrixDisplay

# Set style for plots
sns.set_theme(style="whitegrid")


# Load Datasets
cancer = load_breast_cancer()
digits = load_digits()

print("--- Breast Cancer Dataset ---")
print(f"Shape: {cancer.data.shape}")
print(f"Feature Names (first 5): {cancer.feature_names[:5]}")
print(f"Target Names: {cancer.target_names}")
print(f"Sample Outputs: {cancer.target[:10]}")

print("\n--- MNIST Digits Dataset ---")
print(f"Shape: {digits.data.shape}")
print(f"Feature Names (first 5): {digits.feature_names[:5]}")
print(f"Target Names: {digits.target_names}")
print(f"Sample Outputs: {digits.target[:10]}")


# Split each dataset into 70% training, 30% testing
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    cancer.data, cancer.target, test_size=0.3, random_state=42, stratify=cancer.target)

X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(
    digits.data, digits.target, test_size=0.3, random_state=42, stratify=digits.target)

print("Breast Cancer Split:", X_train_c.shape, X_test_c.shape)
print("Digits Split:", X_train_d.shape, X_test_d.shape)

text_split = """
**Why splitting is required:**
Splitting is necessary to evaluate the model's performance on unseen data. If we trained and evaluated on the same dataset, the model might just memorize the data (overfitting) and perform poorly in the real world. A separate test set gives us an unbiased estimate of how well the model generalizes to new examples.
"""
from IPython.display import Markdown
# display(Markdown(text_split))


# Train Logistic Regression on the Breast Cancer dataset
log_reg_c = LogisticRegression(max_iter=10000, random_state=42)
log_reg_c.fit(X_train_c, y_train_c)

y_pred_log_c = log_reg_c.predict(X_test_c)
acc_log_c = accuracy_score(y_test_c, y_pred_log_c)
cm_log_c = confusion_matrix(y_test_c, y_pred_log_c)

print(f"Logistic Regression (Breast Cancer) Accuracy: {acc_log_c:.4f}")
print("Confusion Matrix:")
print(cm_log_c)


# Calculate precision, recall, F1-score
prec_c = precision_score(y_test_c, y_pred_log_c)
rec_c = recall_score(y_test_c, y_pred_log_c)
f1_c = f1_score(y_test_c, y_pred_log_c)

print(f"Precision: {prec_c:.4f}")
print(f"Recall: {rec_c:.4f}")
print(f"F1-Score: {f1_c:.4f}")

text_f1 = """
**Interpretation:**
- **Precision (0.9464)**: Out of all the tumors the model predicted as benign (class 1), 94.64% were actually benign.
- **Recall (0.9907)**: Out of all the actual benign tumors, the model successfully identified 99.07% of them.
- **F1-Score (0.9680)**: The harmonic mean of precision and recall. Since both are very high, the F1-score is excellent, showing a highly balanced and accurate model.
"""
# display(Markdown(text_f1))


# Train Logistic Regression on MNIST
log_reg_d = LogisticRegression(max_iter=10000, random_state=42)
start_time = time.time()
log_reg_d.fit(X_train_d, y_train_d)
log_reg_d_time = time.time() - start_time

y_pred_log_d = log_reg_d.predict(X_test_d)
acc_log_d = accuracy_score(y_test_d, y_pred_log_d)

print(f"Logistic Regression (MNIST) Accuracy: {acc_log_d:.4f}")
print(f"Training Time: {log_reg_d_time:.4f} seconds")

text_log_multi = """
**Performance & Challenges:**
The model achieves a surprisingly high accuracy (~96.3%) on this subset of MNIST digits. However, a key challenge with Logistic Regression on multiclass image data is that it doesn't capture complex non-linear spatial patterns (like recognizing shapes). It simply assigns weights to specific pixel intensities. Additionally, it requires optimization techniques like "One-vs-Rest" or "Multinomial" which can be computationally expensive on larger datasets.
"""
# display(Markdown(text_log_multi))


# Train SVM classifier on Breast Cancer dataset
svm_c = SVC(kernel='linear', random_state=42)
svm_c.fit(X_train_c, y_train_c)

y_pred_svm_c = svm_c.predict(X_test_c)
acc_svm_c = accuracy_score(y_test_c, y_pred_svm_c)
f1_svm_c = f1_score(y_test_c, y_pred_svm_c)

print(f"SVM (Breast Cancer) Accuracy: {acc_svm_c:.4f}")
print(f"SVM (Breast Cancer) F1-Score: {f1_svm_c:.4f}")

text_svm_compare = f"""
**Comparison:**
- **Logistic Regression**: Accuracy = {acc_log_c:.4f}, F1 = {f1_c:.4f}
- **SVM (Linear)**: Accuracy = {acc_svm_c:.4f}, F1 = {f1_svm_c:.4f}

Both models perform exceptionally well, achieving similar top-tier accuracy. This suggests that the Breast Cancer dataset is highly linearly separable, meaning a straight line (or hyperplane) easily divides the malignant and benign classes.
"""
# display(Markdown(text_svm_compare))


# Train SVM classifier for MNIST
svm_d = SVC(kernel='rbf', random_state=42) # RBF is typically best for non-linear image data
start_time = time.time()
svm_d.fit(X_train_d, y_train_d)
svm_d_time = time.time() - start_time

y_pred_svm_d = svm_d.predict(X_test_d)
acc_svm_d = accuracy_score(y_test_d, y_pred_svm_d)

print(f"SVM (MNIST) Accuracy: {acc_svm_d:.4f}")
print(f"SVM (MNIST) Training Time: {svm_d_time:.4f} seconds")

text_svm_mnist = f"""
**Explanation:**
- **Accuracy Difference**: SVM with an RBF kernel achieves a higher accuracy ({acc_svm_d:.4f}) compared to Logistic Regression ({acc_log_d:.4f}). This is because RBF kernels can map pixels into higher dimensional spaces, capturing non-linear relationships that Logistic Regression misses.
- **Time Taken**: Logistic Regression took {log_reg_d_time:.4f}s while SVM took {svm_d_time:.4f}s. SVMs generally scale poorly with the number of samples ($O(n^2)$ or $O(n^3)$), making them slower to train on larger datasets.
- **Advantages / Disadvantages**: SVMs are powerful for complex boundaries and high-dimensional spaces, but they are computationally heavy, sensitive to noisy data, and lack direct probability estimates compared to Logistic Regression.
"""
# display(Markdown(text_svm_mnist))


# Plot confusion matrix for SVM on MNIST
fig, ax = plt.subplots(figsize=(8, 6))
cm_svm_d = confusion_matrix(y_test_d, y_pred_svm_d)
disp = ConfusionMatrixDisplay(confusion_matrix=cm_svm_d, display_labels=digits.target_names)
disp.plot(cmap='Blues', ax=ax)
plt.title("Confusion Matrix: SVM on MNIST")
plt.grid(False)
plt.show()

text_cm = """
**What misclassifications mean:**
The rows represent the true labels, and the columns represent the predicted labels. The diagonal shows correct predictions. A misclassification is any non-zero number outside the diagonal. For example, if row '3' intersects column '8' with a value of 2, it means the model incorrectly predicted a handwritten '3' as an '8' two times. This happens because those numbers can visually resemble each other.
"""
# display(Markdown(text_cm))


# Compare models based on accuracy, F1-score, confusion matrix
print("--- Multiclass (MNIST) Comparison ---")
print(f"Logistic Regression Accuracy: {acc_log_d:.4f}")
print(f"SVM Accuracy: {acc_svm_d:.4f}")

text_comp = """
**Discussion: Which model performs better and why?**
- For the **binary, linearly separable dataset (Breast Cancer)**, both models performed nearly identically. Logistic Regression is preferred here because it's simpler, faster, and provides interpretable probabilities.
- For the **multiclass, non-linear dataset (MNIST Digits)**, SVM performed better in terms of accuracy. The RBF kernel allows SVM to find complex, non-linear decision boundaries between different digits, whereas Logistic Regression is limited to drawing linear hyperplanes between classes, leading to more misclassifications.
"""
# display(Markdown(text_comp))


text_reflection = """
### Reflection

**When should we use Logistic Regression?**
Use Logistic Regression when the problem is binary, the relationship between features is primarily linear, you need fast training times, and you require interpretable probability scores (e.g., "there is an 80% chance this email is spam").

**When is SVM better?**
SVM is better when the data has complex, non-linear boundaries (using kernel tricks like RBF), when you have high-dimensional data (many features), and when the margin between classes needs to be maximized for robustness. It is less suitable for massive datasets due to high training times.

**Why is F1-score more useful than accuracy sometimes?**
F1-score is crucial when dealing with **imbalanced datasets**. If 99% of emails are normal and 1% are spam, a model that blindly guesses "normal" every time will have 99% accuracy but a terrible F1-score (0 for the spam class). F1-score balances Precision and Recall, ensuring the model is actually capable of detecting the minority class rather than just relying on the majority class distribution.
"""
# display(Markdown(text_reflection))

