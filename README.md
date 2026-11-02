# Medical Insurance Cost Prediction

## Project Overview
This project uses Machine Learning regression algorithms to predict medical insurance charges based on customer details.

## Dataset
The dataset contains information about:
- Age
- Sex
- BMI
- Number of children
- Smoking status
- Region
- Insurance charges

## Data Preprocessing
- Checked and handled duplicate records.
- Checked missing values.
- Applied encoding to categorical variables.
- Split data into training and testing sets.
- Applied StandardScaler for feature scaling.

## Machine Learning Models
The following regression models were trained:
1. Linear Regression
2. Ridge Regression
3. Lasso Regression
4. ElasticNet Regression

## Best Model
Linear Regression performed best among the tested models, with an R² score of approximately 0.807 on the test dataset.

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Matplotlib
- Seaborn

## How to Run the Project

Install the required libraries:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

## Application
The Streamlit application allows users to enter personal details and obtain an estimated medical insurance cost.