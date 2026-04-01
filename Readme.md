# Therapy Predictor using Machine Learning

![Python](https://img.shields.io/badge/Python-3.x-blue)
![scikit-learn](https://img.shields.io/badge/ML-scikit--learn-orange)
![Status](https://img.shields.io/badge/Status-Completed-success)
![Type](https://img.shields.io/badge/Project-Classification-informational)

---

##  Author  
**Ailya Zainab**  
BSDS-2A  

---

## Overview  

This project implements a complete machine learning pipeline to predict whether an individual seeks mental health treatment based on workplace and personal factors.

The focus of this project is not just model accuracy, but building a **clean, reproducible pipeline** that integrates preprocessing, training, and evaluation in a structured way.

---

## Repository Structure  

```
Therapy-Predictor-using-ML/
│
├── survey.csv
├── Mental-Health-Classification.ipynb
├── README.md
```

- `survey.csv` → dataset used for training/testing  
- `Mental-Health-Classification.ipynb` → full implementation (pipeline + models + results)  
- `README.md` → project documentation  

---

## Dataset  

- **Name:** Mental Health in Tech Survey  
- **Source:** OSMI / Kaggle  
- **Samples:** 1259  
- **Features:** 27  
- **Link:** https://www.kaggle.com/datasets/osmi/mental-health-in-tech-survey

### Target Variable  
`treatment`

- `1` → Yes (sought treatment)  
- `0` → No (did not seek treatment)  

### Notes  
- Mix of numerical and categorical features  
- Missing values present (handled in pipeline)  
- Some noisy values (e.g., unrealistic ages)  

---

## Pipeline Design  

The entire workflow is built using **scikit-learn Pipelines** to ensure consistency and avoid data leakage.

### Preprocessing  

**Numerical Data**
- Mean imputation  
- Standard scaling  

**Categorical Data**
- Most frequent imputation  
- One-hot encoding  

Implemented using:
- `Pipeline`
- `ColumnTransformer`

---

## Models Implemented  

- Logistic Regression  
- Decision Tree  
- K-Nearest Neighbors (KNN)  
- Support Vector Machine (SVM)  

Each model is wrapped inside a pipeline along with preprocessing.

---

## Hyperparameter Tuning  

Basic tuning is done using `GridSearchCV`:

- Logistic Regression → `C`  
- Decision Tree → `max_depth`  
- KNN → `n_neighbors`  
- SVM → `kernel`, `C`  

---

## Ensemble Learning  

A **Voting Classifier** is used to combine predictions:

- **Hard Voting** → majority decision  
- **Soft Voting** → probability averaging  

Soft voting is enabled using `probability=True` in SVM.

---

## Evaluation Metrics  

- Accuracy  
- Precision  
- Recall  
- F1 Score  
- ROC-AUC  
- PR-AUC  
- Confusion Matrix  

---

## Results  

- **Best Individual Model:** SVM (~74.6% accuracy)  
- **Voting Classifier:** similar or slightly improved performance  
- **Soft Voting > Hard Voting**  

---

## Insights  

- Different models capture different patterns → combining them improves stability  
- SVM performs well with high-dimensional encoded data  
- Decision Tree tends to overfit if not controlled  
- False negatives are important in this problem (missed treatment cases)  

---

##  How to Run  

```bash
# clone repo
git clone https://github.com/Ailya-Shah/Therapy-Predictor-using-ML.git

# go into folder
cd Therapy-Predictor-using-ML

# install dependencies
pip install pandas numpy scikit-learn matplotlib jupyter joblib

# run notebook
jupyter notebook
```

Open:
```
Mental-Health-Classification.ipynb
```

Run all cells from top to bottom to reproduce the full workflow and generate:
```
best_model.joblib
```

---

## Requirements  

- Python 3.x  
- pandas  
- numpy  
- scikit-learn  
- matplotlib  
- jupyter  
- joblib  

---

## Conclusion  

This project demonstrates a complete ML pipeline for a real-world classification task.

- Pipeline ensures reproducibility  
- SVM gave best standalone results  
- Voting classifier improved overall robustness  

The project highlights the importance of combining models and handling mixed-type data effectively.

---

## Notes  

- All preprocessing is done inside pipelines  
- No data leakage  
- Fully reproducible workflow  
- Developed as part of a Machine Learning lab  

---