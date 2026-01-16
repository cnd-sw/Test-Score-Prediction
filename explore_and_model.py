import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Load data
print("Loading data...")
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
sample_submission = pd.read_csv('sample_submission.csv')

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")
print("\nTrain columns:", train.columns.tolist())
print("\nFirst few rows:")
print(train.head())

print("\nData types:")
print(train.dtypes)

print("\nMissing values:")
print(train.isnull().sum())

print("\nBasic statistics:")
print(train.describe())

print("\nTarget variable statistics:")
print(f"Mean: {train['exam_score'].mean():.2f}")
print(f"Std: {train['exam_score'].std():.2f}")
print(f"Min: {train['exam_score'].min():.2f}")
print(f"Max: {train['exam_score'].max():.2f}")

print("\nCategorical features:")
categorical_cols = train.select_dtypes(include=['object']).columns.tolist()
print(categorical_cols)

for col in categorical_cols:
    print(f"\n{col}: {train[col].unique()}")
    print(f"Value counts:\n{train[col].value_counts()}")

print("\nNumerical features correlation with target:")
numerical_cols = train.select_dtypes(include=[np.number]).columns.tolist()
numerical_cols.remove('id')
numerical_cols.remove('exam_score')

correlations = train[numerical_cols + ['exam_score']].corr()['exam_score'].sort_values(ascending=False)
print(correlations)
