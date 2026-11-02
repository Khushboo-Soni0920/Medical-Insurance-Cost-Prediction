import pandas as pd
# Import joblib to save machine learning models
import joblib

# ==============================
# DATA INSPECTION
# ==============================

# Load the medical insurance dataset
df = pd.read_csv("data/insurance.csv")

# Display the first 5 rows
print(df.head())

# Display the number of rows and columns
print("Dataset Shape:", df.shape)

# 3. Display information about columns and data types
print(df.info())

# 4. Check missing values in each column
print("\nMissing Values:")
print(df.isnull().sum())

# 5. Check duplicate rows
print("\nNumber of Duplicate Rows:")
print(df.duplicated().sum())

# 6. Display statistical summary of numerical columns
print("\nStatistical Summary:")
print(df.describe())

# 8. Display column names
print("\nColumn Names:")
print(df.columns)

# ==================================
# DATA CLEANING
# ==================================

# Remove duplicate rows
df = df.drop_duplicates()

# Check duplicate rows after removal
print("\nDuplicates after cleaning:")
print(df.duplicated().sum())

# Display updated dataset shape
print("\nDataset Shape after cleaning:")
print(df.shape)

# Check missing values after data cleaning
print("\nMissing Values after cleaning:")
print(df.isnull().sum())

# Check minimum and maximum values of numerical columns
print("\nMinimum Values:")
print(df.select_dtypes(include='number').min())
# Check maximum values
print("\nMaximum Values:")
print(df.select_dtypes(include='number').max())

# Check unique values in categorical columns
print("\nUnique Values in Categorical Columns:")
print("Sex:", df['sex'].unique())
print("Smoker:", df['smoker'].unique())
print("Region:", df['region'].unique())

# ==================================
# OUTLIER DETECTION USING IQR METHOD
# ==================================

# Select numerical columns
numerical_columns = ['age', 'bmi', 'children', 'charges']

# Calculate outliers for each numerical column
for col in numerical_columns:

    # Calculate Q1 and Q3
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)

    # Calculate IQR
    IQR = Q3 - Q1

    # Calculate lower and upper limits
    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR

    # Count outliers
    outliers = df[(df[col] < lower_limit) | (df[col] > upper_limit)]

    print(f"\nOutliers in {col}: {len(outliers)}")
    print("Lower Limit:", lower_limit)
    print("Upper Limit:", upper_limit)

    # ==================================
# INSPECT OUTLIER VALUES
# ==================================

# Display BMI values above the upper limit
print("\nBMI Outlier Values:")
print(df[df['bmi'] > 47.315]['bmi'])

# Display insurance charges above the upper limit
print("\nCharges Outlier Values:")
print(df[df['charges'] > 34524.777625]['charges'])

# ==================================
# INVESTIGATE CHARGES OUTLIERS
# ==================================

# Filter records where charges exceed the IQR upper limit
charges_outliers = df[df['charges'] > 34524.777625]

# Check smoker and non-smoker counts among outliers
print("\nSmoker Status Among Charges Outliers:")
print(charges_outliers['smoker'].value_counts())

# Display average charges for smokers and non-smokers
print("\nAverage Charges by Smoking Status:")
print(df.groupby('smoker')['charges'].mean())

# ==================================
# INVESTIGATE BMI OUTLIERS
# ==================================

# Filter records where BMI is above the IQR upper limit
bmi_outliers = df[df['bmi'] > 47.315]

# Display BMI, smoking status and insurance charges
print("\nBMI Outlier Investigation:")
print(bmi_outliers[['bmi', 'smoker', 'charges']])

# Count smokers and non-smokers among BMI outliers
print("\nSmoking Status Among BMI Outliers:")
print(bmi_outliers['smoker'].value_counts())

# Calculate average charges among BMI outliers
print("\nAverage Charges Among BMI Outliers:")
print(bmi_outliers['charges'].mean())

# ==================================
# FEATURE ENGINEERING - ENCODING
# ==================================

# Binary encoding for sex column
df['sex'] = df['sex'].map({'female': 0, 'male': 1})

# Binary encoding for smoker column
df['smoker'] = df['smoker'].map({'no': 0, 'yes': 1})

# One-Hot Encoding for region column
df = pd.get_dummies(df, columns=['region'], dtype=int)

# Display the first 5 rows after encoding
print("\nDataset after Encoding:")
print(df.head())

# Display updated column names
print("\nColumns after Encoding:")
print(df.columns)

# Check dataset information after encoding
print("\nDataset Information After Encoding:")
df.info()

# Check updated dataset shape
print("\nDataset Shape After Encoding:")
print(df.shape)

# Check for missing values after encoding
print("\nMissing Values After Encoding:")
print(df.isnull().sum())

# ==================================
# RESET DATASET INDEX
# ==================================

# Reset index after removing duplicate records
df = df.reset_index(drop=True)

# Verify updated index
print("\nDataset after resetting index:")
print(df.index)

# Separate input features (X) and target variable (y)

X = df.drop('charges', axis=1)

y = df['charges']

# Display the shapes
print("X Shape:", X.shape)
print("y Shape:", y.shape)

# Import train_test_split
from sklearn.model_selection import train_test_split

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Display the shapes
print("X_train Shape:", X_train.shape)
print("X_test Shape:", X_test.shape)
print("y_train Shape:", y_train.shape)
print("y_test Shape:", y_test.shape)

# Import StandardScaler for feature scaling
from sklearn.preprocessing import StandardScaler

# Create an object of StandardScaler
scaler = StandardScaler()

# Fit the scaler on training data and standardize it
X_train_scaled = scaler.fit_transform(X_train)

# Display the scaled training data
print("Scaled Training Data:")
print(X_train_scaled[:5])

# Transform testing data using the scaler fitted on training data
X_test_scaled = scaler.transform(X_test)

# Display the first five rows of scaled testing data
print("\nScaled Testing Data:")
print(X_test_scaled[:5])

# Import Linear Regression algorithm
from sklearn.linear_model import LinearRegression

# Create an object of Linear Regression
model = LinearRegression()

# Train the Linear Regression model
model.fit(X_train_scaled, y_train)

print("Linear Regression model trained successfully!")

# Predict insurance charges using testing data
y_pred = model.predict(X_test_scaled)

# Display the first five predicted values
print("\nPredicted Insurance Charges:")
print(y_pred[:5])

# Compare actual and predicted insurance charges
comparison = pd.DataFrame({
    'Actual Charges': y_test.values[:5],
    'Predicted Charges': y_pred[:5]
})

print("\nActual vs Predicted Charges:")
print(comparison)

# Import evaluation metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Calculate Mean Absolute Error
mae = mean_absolute_error(y_test, y_pred)

# Display MAE
print("\nMean Absolute Error:", mae)

# Calculate Mean Squared Error
mse = mean_squared_error(y_test, y_pred)

# Display MSE
print("Mean Squared Error:", mse)

# Calculate Root Mean Squared Error
import numpy as np
rmse = np.sqrt(mse)

# Display RMSE
print("Root Mean Squared Error:", rmse)

# Calculate R2 Score
r2 = r2_score(y_test, y_pred)

# Display R2 Score
print("R2 Score:", r2)

# Display Linear Regression coefficients
coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
})

print("\nLinear Regression Coefficients:")
print(coefficients)

# Display intercept
print("\nIntercept:", model.intercept_)

# Import Ridge Regression
from sklearn.linear_model import Ridge

# Create Ridge Regression model
ridge_model = Ridge(alpha=1.0)

# Train the Ridge Regression model
ridge_model.fit(X_train_scaled, y_train)

print("\nRidge Regression model trained successfully!")

# Predict insurance charges using Ridge Regression
ridge_pred = ridge_model.predict(X_test_scaled)

# Display first five predictions
print("\nRidge Regression Predictions:")
print(ridge_pred[:5])

# Evaluate Ridge Regression model

ridge_mae = mean_absolute_error(y_test, ridge_pred)
ridge_mse = mean_squared_error(y_test, ridge_pred)
ridge_rmse = np.sqrt(ridge_mse)
ridge_r2 = r2_score(y_test, ridge_pred)

# Display evaluation metrics
print("\nRidge Regression Evaluation:")
print("MAE:", ridge_mae)
print("MSE:", ridge_mse)
print("RMSE:", ridge_rmse)
print("R2 Score:", ridge_r2)

# Import Lasso Regression
from sklearn.linear_model import Lasso

# Create Lasso Regression model
lasso_model = Lasso(alpha=1.0)

# Train the Lasso Regression model
lasso_model.fit(X_train_scaled, y_train)

print("\nLasso Regression model trained successfully!")

# Generate predictions using Lasso Regression
lasso_pred = lasso_model.predict(X_test_scaled)

# Calculate evaluation metrics
lasso_mae = mean_absolute_error(y_test, lasso_pred)
lasso_mse = mean_squared_error(y_test, lasso_pred)
lasso_rmse = np.sqrt(lasso_mse)
lasso_r2 = r2_score(y_test, lasso_pred)

# Display evaluation results
print("\nLasso Regression Evaluation:")
print("MAE:", lasso_mae)
print("MSE:", lasso_mse)
print("RMSE:", lasso_rmse)
print("R2 Score:", lasso_r2)

# Import ElasticNet Regression
from sklearn.linear_model import ElasticNet

# Create ElasticNet Regression model
elastic_model = ElasticNet(alpha=1.0, l1_ratio=0.5)

# Train the ElasticNet model
elastic_model.fit(X_train_scaled, y_train)

print("\nElasticNet model trained successfully!")

# Generate predictions using ElasticNet
elastic_pred = elastic_model.predict(X_test_scaled)

# Calculate evaluation metrics
elastic_mae = mean_absolute_error(y_test, elastic_pred)
elastic_mse = mean_squared_error(y_test, elastic_pred)
elastic_rmse = np.sqrt(elastic_mse)
elastic_r2 = r2_score(y_test, elastic_pred)

# Display evaluation results
print("\nElasticNet Regression Evaluation:")
print("MAE:", elastic_mae)
print("MSE:", elastic_mse)
print("RMSE:", elastic_rmse)
print("R2 Score:", elastic_r2)

# Save the best performing Linear Regression model
joblib.dump(model, "models/linear_regression_model.pkl")

# Save the StandardScaler
joblib.dump(scaler, "models/scaler.pkl")

print("\nModel and scaler saved successfully!")