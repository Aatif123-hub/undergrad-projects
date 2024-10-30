## Task: Predicting Data Quality Indicator Using Support Vector Machine (SVM)
## Objective
Predict the data_quality indicator (0 or 1) based on various cost-of-living and economic factors for cities worldwide, utilizing an SVM model.

## Preprocessing
### Missing Data Handling: 
Describe the imputation strategy chosen for handling missing values across the cost-of-living factors.
### Categorical Encoding: 
Encode categorical columns (city and country) using techniques like one-hot encoding or label encoding.
### Feature Scaling: 
Apply feature scaling to numeric columns (x1 through x55), essential for SVM models to perform well.
## Model Training and Evaluation
### Model Training: 
Train an SVM model on the dataset and experiment with hyperparameters like the kernel type (e.g., linear, RBF) and regularization parameter (C) for optimal results.
### Evaluation Metrics: 
Report accuracy, precision, recall, and F1-score to evaluate the SVM model's predictive performance.
## Visualizations
### Confusion Matrix: 
Generate a confusion matrix plot to visualize model classification performance.
### ROC Curve: 
Include an ROC curve plot and calculate the AUC score to evaluate the model’s ability to distinguish between different data_quality levels.
