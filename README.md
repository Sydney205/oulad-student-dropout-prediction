# Student Dropout Prediction using OULAD Dataset

## Project Overview

This project focuses on developing a machine learning model to predict student dropout risk in online learning environments. Leveraging the Open University Learning Analytics Dataset (OULAD), the initiative aims to identify at-risk students early enough for targeted interventions. The core objective is not only to build an effective predictive model but also to gain insights into the behavioral and demographic factors that correlate with student withdrawal.

### Key Question Addressed:

Can student engagement, academic performance, and demographic information effectively predict dropout risk in online courses, enabling timely intervention?

## Features & Functionality

- **Data Ingestion & Integration:** Processes and merges multiple relational tables from the OULAD dataset, focusing on student demographics, engagement, and assessment data.
- **Feature Engineering:** Aggregates raw event-level data (e.g., VLE clicks, assessment scores) into meaningful student-level features (e.g., total clicks, median score).
- **Missing Value Handling:** Implements strategic imputation techniques tailored to educational data (e.g., 0 for missing clicks/scores, 'Unknown' for IMD band).
- **Categorical Encoding:** Employs one-hot encoding for nominal variables (region, gender) and ordinal encoding for variables with intrinsic order (highest education, age band, IMD band).
- **Binary Classification:** Transforms the `final_result` into a binary `dropout` target variable, simplifying the prediction problem.
- **Logistic Regression Model:** Utilizes a Logistic Regression model with `StandardScaler` for robust prediction, incorporating `class_weight='balanced'` to address dataset imbalance.
- **Performance Evaluation:** Assesses model efficacy using accuracy, classification report (precision, recall, F1-score), and a confusion matrix.
- **Streamlit Web Application:** Provides an interactive user interface for real-time dropout risk prediction based on user-inputted student features.

## Technical Stack

- **Python:** Primary programming language.
- **Pandas & NumPy:** Data manipulation and numerical operations.
- **Matplotlib & Seaborn:** Data visualization and exploratory data analysis.
- **Scikit-learn:** Machine learning model development (preprocessing, modeling, evaluation).
- **Joblib:** Efficient serialization and deserialization of the trained machine learning model.
- **Streamlit:** Framework for building interactive web applications for model deployment.
- **Kagglehub:** For convenient dataset access.

## Getting Started

To run this project locally, follow these steps:

### 1. Prerequisites

Ensure you have Python 3.8+ installed. You can download it from [python.org](https://www.python.org/).

### 2. Installation

Clone the repository and install the required Python packages:

```bash
git clone <repository-url>
cd student-dropout-prediction
pip install -r requirements.txt
```

**Note:** If `requirements.txt` is not available, you can install the necessary libraries manually:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn streamlit joblib kagglehub
```

### 3. Dataset

The project uses the OULAD dataset. The notebook automatically downloads it via `kagglehub`. Ensure you have an internet connection when running the notebook for the first time.

### 4. Running the Notebook

Execute the provided Jupyter/Colab notebook (`student_dropout_prediction.ipynb` or similar) sequentially. This will:

- Load and preprocess the data.
- Train the Logistic Regression model.
- Evaluate the model's performance.
- Save the trained model as `logistic_regression_dropout_model.joblib`.

### 5. Running the Streamlit Application

After successfully running the notebook and saving the model, create a Python file (e.g., `app.py`) with the Streamlit code provided in the notebook. Then, navigate to the directory containing `app.py` in your terminal and run:

```bash
streamlit run app.py
```

This will launch the interactive web application in your browser, allowing you to input student features and get real-time dropout predictions.

## Project Structure

- `student_dropout_prediction.ipynb` (or similar): The main notebook containing all data processing, modeling, and evaluation steps.
- `logistic_regression_dropout_model.joblib`: The saved trained model file.
- `app.py` (example): A Python script for the Streamlit web application (to be created by the user from the notebook output).
- `README.md`: This file.

## Model Details & Evaluation

The project utilizes a `Pipeline` from `scikit-learn` comprising a `StandardScaler` for feature scaling and a `LogisticRegression` classifier. The `class_weight='balanced'` parameter was crucial for handling the imbalanced nature of the target variable (more non-dropouts than dropouts).

### Performance Metrics (Example Output):

```
Accuracy: 0.764

Classification Report:
              precision    recall  f1-score   support

           0       0.84      0.81      0.82      4488  (Non-Dropout)
           1       0.61      0.67      0.64      2031  (Dropout)

    accuracy                           0.76      6519
   macro avg       0.73      0.74      0.73      6519
weighted avg       0.77      0.76      0.77      6519

Confusion Matrix:
[[3619  869]
 [ 669 1362]]
```

**Interpretation:**
- **Accuracy:** The model achieves an overall accuracy of 76.4%.
- **Precision for Dropout (Class 1):** 0.61. This means that when the model predicts a student will drop out, it is correct 61% of the time. A lower precision suggests a notable number of false positives (students predicted to drop out who actually don't).
- **Recall for Dropout (Class 1):** 0.67. This indicates that the model correctly identifies 67% of all actual dropout students. A higher recall is often desirable in dropout prediction to maximize the identification of at-risk students for intervention.

This balance highlights a trade-off. While the model is good at catching a significant portion of dropouts, it also flags some students unnecessarily. Further refinement could focus on optimizing this balance based on the specific costs of false positives versus false negatives in a real-world intervention program.

## Contributing

Contributions are welcome! If you have suggestions for improvements, feature enhancements, or bug fixes, please open an issue or submit a pull request.
