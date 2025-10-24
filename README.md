# Load Default Prediction

This project aims to build a machine learning model to predict loan defaults based on customer behavior and other key features. The goal is to identify high-risk individuals to assist banks and financial institutions in better decision-making when granting loans.

## Overview

The project analyzes a dataset of customer information, including personal and financial details, to predict the likelihood of loan default. The analysis is performed through feature engineering, exploratory data analysis (EDA), and machine learning techniques. The final model is evaluated using various metrics to assess performance and balance between false positives and false negatives.

## Dataset

The dataset used in this project is publicly available on Kaggle:

**Loan Prediction Based on Customer Behavior Dataset**  
[Link to Dataset on Kaggle](https://www.kaggle.com/datasets/subhamjain/loan-prediction-based-on-customer-behavior)

It includes the following features:
- **Income**: Annual income of the customer.
- **Age**: Age of the customer.
- **Experience**: Years of work experience.
- **Current Job Years**: Years in the current job.
- **Current House Years**: Years at the current residence.
- **Marital Status**: Whether the customer is married or single.
- **Car Ownership**: Whether the customer owns a car.
- **House Ownership**: Whether the customer owns a house.
- **Other demographic details** like city, state, etc.

## Features and Methodology

### Data Preprocessing:
- Handled missing values and data inconsistencies.
- Categorical features were encoded using techniques like label encoding and response encoding.
- Features were scaled using Min-Max normalization.

### Exploratory Data Analysis (EDA):
- Visualized the distribution of key features like income, age, and work experience.
- Analyzed relationships between features and the target variable, `loan_default`.
- Checked for data imbalance and handled it using oversampling techniques like SMOTEENN.

### Feature Engineering
- Handled missing values.
- Encoded categorical features.
- Created new features: Age grouping, interaction features between income and age, experience and current_job_years, and car_ownership and house_ownership were also engineered to capture combined effects between these variables.
- Scaled numerical features using Min-Max Scaling.
- Feature Selection: The initial dataset had a large number of features, but after performing feature selection techniques like correlation analysis and analyzing feature importance from decision trees, the most relevant features were selected for the final model. Features with low correlation to the target or high correlation with other features were dropped to reduce noise and prevent overfitting.

### Machine Learning Models:
- **Logistic Regression**: Used as a baseline model.
- **Decision Tree**: Evaluated for better performance.
- **Random Forest**: Used for more robust classification.
- **XGBoost**: The final model, showing the best performance.

### Model Evaluation:
- Evaluated models based on metrics like accuracy, precision, recall, F1-score, and ROC-AUC.
- **XGBoost** achieved the highest performance with an AUC of 0.92 and accuracy of 84%.

### Model Tuning:
- Fine-tuned models using techniques like GridSearch and RandomizedSearch to optimize hyperparameters.

## Files Included
- **LoanDefaultPrediction.ipynb**: Jupyter notebook containing the full analysis, including data preprocessing, feature engineering, and model building.
- **report.pdf**: The final report outlining the methodology, results, and recommendations of the project.

## Usage

To run the analysis:

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/LoanDefaultPrediction.git
    cd LoanDefaultPrediction
    ```

2. Install required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the Jupyter notebook:
    ```bash
    jupyter notebook LoanDefaultPrediction.ipynb
    ```

## Results

The final model, **XGBoost**, was able to predict loan defaults with significant accuracy, outperforming other models in terms of both recall and precision. The performance metrics are as follows:

### Logistic Regression:
- **Training Accuracy**: 68%
- **Test Accuracy**: 60%
- **Precision (Class 1)**: 0.60
- **Recall (Class 1)**: 0.54
- **F1-Score (Class 1)**: 0.57

### Decision Tree:
- **Training Accuracy**: 79%
- **Test Accuracy**: 75%
- **Precision (Class 1)**: 0.74
- **Recall (Class 1)**: 0.79
- **F1-Score (Class 1)**: 0.76
- **AUC**: 0.75

### Random Forest:
- **Training Accuracy**: 96%
- **Test Accuracy**: 83%
- **Precision (Class 1)**: 0.85
- **Recall (Class 1)**: 0.75
- **F1-Score (Class 1)**: 0.80
- **AUC**: 0.88

### XGBoost:
- **Training Accuracy**: 92%
- **Test Accuracy**: 84%
- **Precision (Class 1)**: 0.86
- **Recall (Class 1)**: 0.77
- **F1-Score (Class 1)**: 0.81
- **AUC**: 0.89

## Conclusion
- **XGBoost** is the top-performing model, achieving a balanced and robust classification with an accuracy of 84%, AUC of 0.89, and good precision and recall.
- **Random Forest** follows closely with a high test accuracy of 83%, but it shows slight overfitting compared to XGBoost.
- **Decision Tree** and **Logistic Regression** lag behind in terms of overall performance.

These results suggest that **XGBoost** is the most reliable model for predicting loan defaults in this dataset.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- **Kaggle** for providing the dataset.
- Various machine learning libraries used throughout the project, including **scikit-learn**, **pandas**, and **XGBoost**.
