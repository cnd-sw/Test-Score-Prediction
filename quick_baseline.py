import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

print("Loading data...")
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
sample_submission = pd.read_csv('sample_submission.csv')

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")

def quick_feature_engineering(df, is_train=True):
    """Quick but effective feature engineering"""
    df = df.copy()
    
    # Top interaction features
    df['study_attendance'] = df['study_hours'] * df['class_attendance']
    df['study_squared'] = df['study_hours'] ** 2
    df['study_cubed'] = df['study_hours'] ** 3
    df['attendance_squared'] = df['class_attendance'] ** 2
    
    # Target encoding for exam_difficulty (most important categorical)
    if is_train:
        global diff_encoding
        if 'exam_score' in df.columns:
            diff_encoding = df.groupby('exam_difficulty')['exam_score'].mean().to_dict()
            df['diff_target_enc'] = df['exam_difficulty'].map(diff_encoding)
    else:
        df['diff_target_enc'] = df['exam_difficulty'].map(diff_encoding)
    
    return df

print("\nApplying feature engineering...")
train = quick_feature_engineering(train, is_train=True)
test = quick_feature_engineering(test, is_train=False)

# Encode categorical variables
categorical_cols = ['gender', 'course', 'internet_access', 'sleep_quality', 'study_method', 
                   'facility_rating', 'exam_difficulty']

for col in categorical_cols:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col])
    test[col] = le.transform(test[col])

# Prepare features
feature_cols = [col for col in train.columns if col not in ['id', 'exam_score']]
X = train[feature_cols]
y = train['exam_score']
X_test = test[feature_cols]

print(f"\nNumber of features: {len(feature_cols)}")

# Quick 3-fold CV
n_folds = 3
kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

print("\n" + "="*50)
print("Training Quick LightGBM...")
print("="*50)

lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': 127,
    'learning_rate': 0.03,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'max_depth': 10,
    'min_child_samples': 20,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'verbose': -1,
    'random_state': 42,
    'n_jobs': -1
}

oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f"\nFold {fold + 1}/{n_folds}")
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)
    
    model = lgb.train(
        lgb_params,
        lgb_train,
        num_boost_round=3000,
        valid_sets=[lgb_val],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=100)
        ]
    )
    
    oof_preds[val_idx] = model.predict(X_val)
    test_preds += model.predict(X_test) / n_folds
    
    fold_rmse = np.sqrt(mean_squared_error(y_val, oof_preds[val_idx]))
    print(f"Fold {fold + 1} RMSE: {fold_rmse:.6f}")

cv_rmse = np.sqrt(mean_squared_error(y, oof_preds))
print(f"\nOverall CV RMSE: {cv_rmse:.6f}")

# Create submission
submission = sample_submission.copy()
submission['exam_score'] = test_preds

submission.to_csv('submission_quick.csv', index=False)
print("\nSubmission file created: submission_quick.csv")
print(f"\nPrediction statistics:")
print(f"Mean: {submission['exam_score'].mean():.2f}")
print(f"Std: {submission['exam_score'].std():.2f}")
print(f"Min: {submission['exam_score'].min():.2f}")
print(f"Max: {submission['exam_score'].max():.2f}")
