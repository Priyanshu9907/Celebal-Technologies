# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib

# Load the dataset
data_path = 'path/to/your/train.csv'
df = pd.read_csv(data_path)

# Display the first few rows
print(df.head())

# Basic info and statistics
print(df.info())
print(df.describe())

# Check for missing values
print(df.isnull().sum())

# Fill missing values for numerical columns with mean
numerical_features = df.select_dtypes(include=['float64', 'int64']).columns
df[numerical_features] = df[numerical_features].fillna(df[numerical_features].mean())

# Fill missing values for categorical columns with mode
categorical_features = df.select_dtypes(include=['object']).columns
df[categorical_features] = df[categorical_features].fillna(df[categorical_features].mode().iloc[0])

# Example: Extracting year from dates
df['YearBuilt'] = pd.to_datetime(df['YearBuilt'], format='%Y').dt.year
df['YearRemodAdd'] = pd.to_datetime(df['YearRemodAdd'], format='%Y').dt.year

# Create new feature: TotalBsmtSF to indicate basement area
df['TotalBsmtSF'] = df['TotalBsmtSF'].apply(lambda x: np.log(x+1) if x > 0 else 0)

# One-hot encoding for categorical features
categorical_features = ['Neighborhood', 'BldgType', 'OverallCond', 'ExterCond', 'ExterQual']
df = pd.get_dummies(df, columns=categorical_features, drop_first=True)

# Define features and target
X = df.drop('SalePrice', axis=1)
y = df['SalePrice']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define preprocessing for numerical and categorical data
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Define the model pipeline
model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))])

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f'Root Mean Squared Error: {rmse}')

# Save the trained model
joblib.dump(model, 'house_price_model.pkl')

# Load the model
loaded_model = joblib.load('house_price_model.pkl')

# Predict new data
new_data = pd.DataFrame({...})  # Replace with actual new data
predicted_price = loaded_model.predict(new_data)
print(f'Predicted Sale Price: {predicted_price}')