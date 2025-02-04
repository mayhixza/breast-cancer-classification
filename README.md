# Comparative Analysis of Supervised Machine Learning Models for Breast Cancer Classification

This project focuses on comparing the performance of five different machine learning algorithms—K-Nearest Neighbors (KNN), Logistic Regression, Decision Trees, and Support Vector Machines (SVM), Random Forest—for classifying breast cancer as benign or malignant. The goal is to determine which model provides the most accurate and reliable predictions.

### Dataset:
Source: [Breast Cancer Wisconsin (Diagnostic)](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)  
Description: The dataset includes features such as mean radius, texture, perimeter, area, smoothness, etc.  
Size: 569×30  
**Preprocessing:**  
Feature scaling: Standardization/Normalization applied for models like SVM and KNN  
Label encoding: Converting categorical labels (benign/malignant) to numerical values  

### Technologies Use:
Python, NumPy, Pandas, Scikit-learn, Matplotlib

### Evaluation Metrics
Accuracy: Overall correctness of the model.  
Precision & Recall: For imbalanced datasets, these metrics provide better insights.  
F1 Score: Harmonic mean of precision and recall.  
ROC-AUC: Evaluates the trade-off between true positive and false positive rates.  
Confusion Matrix: Provides detailed insight into classification errors.  

### Results
After evaluating all five models, the results were as follows:

K-Nearest Neighbors (KNN): Accuracy of 97%, F1 Score of 0.97  
Logistic Regression: Accuracy of 95%, F1 Score of 0.95  
Decision Trees: Accuracy of 95%, F1 Score of 0.95  
Support Vector Machines (SVM): Accuracy of 96%, F1 Score of 0.96  
Random Forest: Accuracy of 96%, F1 Score of 0.96  

### Conclusion
The analysis demonstrated that K-Nearest Neighbors outperformed the others in terms of accuracy and F1 score. 

## Run the Notebook on Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mayhixza/breast-cancer-classification/blob/main/Breast_Cancer.ipynb)

### How to Use the Colab Notebook

To run this project in Google Colab, click the "Open in Colab" button above. Follow these simple steps to set it up:

1. **Click the "Open in Colab" button** above to open the notebook in Google Colab.
2. **Run the setup cell**: The first cell installs all necessary libraries and mounts Google Drive if needed. You only need to do this once.
3. **Follow the notebook instructions**: Run each cell in order to execute the code and see the results.

**Note**: You must have a Google account to use Google Colab.
