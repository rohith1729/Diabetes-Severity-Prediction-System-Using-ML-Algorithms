# Diabetes-Severity-Prediction-System-Using-ML-Algorithms

## Objective

To build a machine learning-based prediction system that detects diabetes presence and severity using survey data from the CDC's BRFSS. The aim is to identify high-risk individuals early and support personalized health interventions.

**Data Source**

Data is Sourced from CDC's BRFSS Survey.

**Technologies Used**

Languages & Libraries: Python, pandas, NumPy, seaborn, matplotlib, scikit-learn, XGBoost, GeoPandas, SHAP, Hyperopt, Folium
ML Algorithms: Decision Tree, Random Forest, Logistic Regression, XGBoost
Hyperparameter Tuning: GridSearchCV, RandomizedSearchCV, Hyperopt (TPE)
Visualization: seaborn, matplotlib, Folium (map-based), geopandas, SHAP
Statistical Testing: ANOVA (for hypothesis testing)

**Modeling Pipeline**

Addressed severe class imbalance (Yes/No ratio) by under-sampling the majority class and SMOTE.
Applied one-hot encoding to all categorical features for model compatibility.

**Model Training**

Implemented and evaluated four models - Logistic Regression, Decision Tree, Random Forest, XGBoost


**Hyperparameter Tuning**

Used GridSearchCV for Decision Tree and Logistic Regression.
Used RandomizedSearchCV for Random Forest.
Used Hyperopt (TPE) for XGBoost (searching max_depth, learning_rate, subsample).

**Evaluation**

Plotted confusion matrices for each model.
Calculated precision, recall, F1-score for all models.

**ROC Curve Analysis**

Compared ROC curves for XGBoost, Random Forest, and Decision Tree.
XGBoost yielded the highest AUC and best performance under varying thresholds.

