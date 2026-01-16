import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
import warnings
warnings.filterwarnings('ignore')

print("Loading data...")
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
sample_submission = pd.read_csv('sample_submission.csv')

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")

def advanced_feature_engineering(df, is_train=True):
    """Ultra advanced feature engineering"""
    df = df.copy()
    
    # Core interaction features
    df['study_attendance_interaction'] = df['study_hours'] * df['class_attendance']
    df['study_sleep_interaction'] = df['study_hours'] * df['sleep_hours']
    df['attendance_sleep_interaction'] = df['class_attendance'] * df['sleep_hours']
    
    # Three-way interactions
    df['study_attendance_sleep'] = df['study_hours'] * df['class_attendance'] * df['sleep_hours']
    
    # Polynomial features
    df['study_hours_squared'] = df['study_hours'] ** 2
    df['study_hours_cubed'] = df['study_hours'] ** 3
    df['class_attendance_squared'] = df['class_attendance'] ** 2
    df['class_attendance_cubed'] = df['class_attendance'] ** 3
    df['sleep_hours_squared'] = df['sleep_hours'] ** 2
    
    # Ratio features
    df['study_per_age'] = df['study_hours'] / (df['age'] + 1)
    df['sleep_per_age'] = df['sleep_hours'] / (df['age'] + 1)
    df['study_sleep_ratio'] = df['study_hours'] / (df['sleep_hours'] + 0.1)
    df['attendance_sleep_ratio'] = df['class_attendance'] / (df['sleep_hours'] + 0.1)
    
    # Log transformations
    df['log_study_hours'] = np.log1p(df['study_hours'])
    df['log_attendance'] = np.log1p(df['class_attendance'])
    df['log_sleep'] = np.log1p(df['sleep_hours'])
    
    # Sqrt transformations
    df['sqrt_study_hours'] = np.sqrt(df['study_hours'])
    df['sqrt_attendance'] = np.sqrt(df['class_attendance'])
    
    # Binning features
    df['age_bin'] = pd.cut(df['age'], bins=[0, 19, 21, 23, 100], labels=[0, 1, 2, 3])
    df['study_hours_bin'] = pd.cut(df['study_hours'], bins=[-1, 2, 4, 6, 10], labels=[0, 1, 2, 3])
    df['attendance_bin'] = pd.cut(df['class_attendance'], bins=[-1, 50, 75, 90, 100], labels=[0, 1, 2, 3])
    df['sleep_bin'] = pd.cut(df['sleep_hours'], bins=[-1, 5, 7, 9, 15], labels=[0, 1, 2, 3])
    
    # Target encoding (mean encoding) for categorical variables
    if is_train:
        global target_encodings
        target_encodings = {}
        
        # We'll use the target for encoding in training
        if 'exam_score' in df.columns:
            for cat_col in ['gender', 'course', 'study_method', 'exam_difficulty', 'facility_rating', 'sleep_quality', 'internet_access']:
                target_encodings[cat_col] = df.groupby(cat_col)['exam_score'].mean().to_dict()
                df[f'{cat_col}_target_enc'] = df[cat_col].map(target_encodings[cat_col])
    else:
        for cat_col in ['gender', 'course', 'study_method', 'exam_difficulty', 'facility_rating', 'sleep_quality', 'internet_access']:
            df[f'{cat_col}_target_enc'] = df[cat_col].map(target_encodings[cat_col])
    
    # Aggregate features
    if is_train:
        global agg_features
        agg_features = {}
        
        for cat_col in ['gender', 'course', 'study_method', 'exam_difficulty', 'facility_rating', 'sleep_quality']:
            agg_features[f'{cat_col}_mean_study'] = df.groupby(cat_col)['study_hours'].mean().to_dict()
            agg_features[f'{cat_col}_mean_attendance'] = df.groupby(cat_col)['class_attendance'].mean().to_dict()
            agg_features[f'{cat_col}_mean_sleep'] = df.groupby(cat_col)['sleep_hours'].mean().to_dict()
            agg_features[f'{cat_col}_std_study'] = df.groupby(cat_col)['study_hours'].std().to_dict()
            
            df[f'{cat_col}_mean_study'] = df[cat_col].map(agg_features[f'{cat_col}_mean_study'])
            df[f'{cat_col}_mean_attendance'] = df[cat_col].map(agg_features[f'{cat_col}_mean_attendance'])
            df[f'{cat_col}_mean_sleep'] = df[cat_col].map(agg_features[f'{cat_col}_mean_sleep'])
            df[f'{cat_col}_std_study'] = df[cat_col].map(agg_features[f'{cat_col}_std_study'])
    else:
        for cat_col in ['gender', 'course', 'study_method', 'exam_difficulty', 'facility_rating', 'sleep_quality']:
            df[f'{cat_col}_mean_study'] = df[cat_col].map(agg_features[f'{cat_col}_mean_study'])
            df[f'{cat_col}_mean_attendance'] = df[cat_col].map(agg_features[f'{cat_col}_mean_attendance'])
            df[f'{cat_col}_mean_sleep'] = df[cat_col].map(agg_features[f'{cat_col}_mean_sleep'])
            df[f'{cat_col}_std_study'] = df[cat_col].map(agg_features[f'{cat_col}_std_study'])
    
    # Deviation from group mean
    for cat_col in ['gender', 'course', 'study_method', 'exam_difficulty']:
        df[f'{cat_col}_study_deviation'] = df['study_hours'] - df[f'{cat_col}_mean_study']
        df[f'{cat_col}_attendance_deviation'] = df['class_attendance'] - df[f'{cat_col}_mean_attendance']
    
    return df

print("\nApplying advanced feature engineering...")
train = advanced_feature_engineering(train, is_train=True)
test = advanced_feature_engineering(test, is_train=False)

# Encode categorical variables
categorical_cols = ['gender', 'course', 'internet_access', 'sleep_quality', 'study_method', 
                   'facility_rating', 'exam_difficulty', 'age_bin', 'study_hours_bin', 'attendance_bin', 'sleep_bin']

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col].astype(str))
    test[col] = le.transform(test[col].astype(str))
    label_encoders[col] = le

# Prepare features
feature_cols = [col for col in train.columns if col not in ['id', 'exam_score']]
X = train[feature_cols]
y = train['exam_score']
X_test = test[feature_cols]

print(f"\nNumber of features: {len(feature_cols)}")

# Cross-validation setup
n_folds = 5  # Reduced folds for faster training
kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

# Optimized LightGBM parameters
print("\n" + "="*50)
print("Training Optimized LightGBM Ensemble...")
print("="*50)

lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': 255,
    'learning_rate': 0.02,
    'feature_fraction': 0.85,
    'bagging_fraction': 0.85,
    'bagging_freq': 5,
    'max_depth': 12,
    'min_child_samples': 20,
    'reg_alpha': 0.05,
    'reg_lambda': 0.05,
    'verbose': -1,
    'random_state': 42,
    'n_jobs': -1
}

lgb_oof = np.zeros(len(X))
lgb_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f"\nFold {fold + 1}/{n_folds}")
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)
    
    model = lgb.train(
        lgb_params,
        lgb_train,
        num_boost_round=5000,
        valid_sets=[lgb_train, lgb_val],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=200)
        ]
    )
    
    lgb_oof[val_idx] = model.predict(X_val)
    lgb_preds += model.predict(X_test) / n_folds
    
    fold_rmse = np.sqrt(mean_squared_error(y_val, lgb_oof[val_idx]))
    print(f"Fold {fold + 1} RMSE: {fold_rmse:.6f}")

lgb_cv_rmse = np.sqrt(mean_squared_error(y, lgb_oof))
print(f"\nLightGBM CV RMSE: {lgb_cv_rmse:.6f}")

# Optimized XGBoost
print("\n" + "="*50)
print("Training Optimized XGBoost...")
print("="*50)

xgb_params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'max_depth': 10,
    'learning_rate': 0.02,
    'subsample': 0.85,
    'colsample_bytree': 0.85,
    'min_child_weight': 1,
    'reg_alpha': 0.05,
    'reg_lambda': 0.05,
    'random_state': 42,
    'tree_method': 'hist',
    'n_jobs': -1
}

xgb_oof = np.zeros(len(X))
xgb_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f"\nFold {fold + 1}/{n_folds}")
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    model = xgb.train(
        xgb_params,
        dtrain,
        num_boost_round=5000,
        evals=[(dtrain, 'train'), (dval, 'val')],
        early_stopping_rounds=50,
        verbose_eval=200
    )
    
    xgb_oof[val_idx] = model.predict(dval)
    xgb_preds += model.predict(xgb.DMatrix(X_test)) / n_folds
    
    fold_rmse = np.sqrt(mean_squared_error(y_val, xgb_oof[val_idx]))
    print(f"Fold {fold + 1} RMSE: {fold_rmse:.6f}")

xgb_cv_rmse = np.sqrt(mean_squared_error(y, xgb_oof))
print(f"\nXGBoost CV RMSE: {xgb_cv_rmse:.6f}")

# Optimized CatBoost
print("\n" + "="*50)
print("Training Optimized CatBoost...")
print("="*50)

cat_params = {
    'iterations': 5000,
    'learning_rate': 0.02,
    'depth': 10,
    'l2_leaf_reg': 1,
    'random_seed': 42,
    'verbose': 200,
    'early_stopping_rounds': 50,
    'task_type': 'CPU',
    'thread_count': -1
}

cat_oof = np.zeros(len(X))
cat_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f"\nFold {fold + 1}/{n_folds}")
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    model = CatBoostRegressor(**cat_params)
    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        use_best_model=True
    )
    
    cat_oof[val_idx] = model.predict(X_val)
    cat_preds += model.predict(X_test) / n_folds
    
    fold_rmse = np.sqrt(mean_squared_error(y_val, cat_oof[val_idx]))
    print(f"Fold {fold + 1} RMSE: {fold_rmse:.6f}")

cat_cv_rmse = np.sqrt(mean_squared_error(y, cat_oof))
print(f"\nCatBoost CV RMSE: {cat_cv_rmse:.6f}")

# Optimized ensemble with stacking
print("\n" + "="*50)
print("Creating optimized ensemble...")
print("="*50)

# Find optimal weights using simple grid search
best_rmse = float('inf')
best_weights = None

for w1 in np.arange(0.2, 0.5, 0.05):
    for w2 in np.arange(0.2, 0.5, 0.05):
        w3 = 1 - w1 - w2
        if w3 >= 0.2 and w3 <= 0.5:
            ensemble_oof = w1 * lgb_oof + w2 * xgb_oof + w3 * cat_oof
            rmse = np.sqrt(mean_squared_error(y, ensemble_oof))
            if rmse < best_rmse:
                best_rmse = rmse
                best_weights = [w1, w2, w3]

print(f"Best weights: LGB={best_weights[0]:.3f}, XGB={best_weights[1]:.3f}, CAT={best_weights[2]:.3f}")
print(f"Best ensemble CV RMSE: {best_rmse:.6f}")

ensemble_preds = best_weights[0] * lgb_preds + best_weights[1] * xgb_preds + best_weights[2] * cat_preds

print("\n" + "="*50)
print("Summary of Results:")
print("="*50)
print(f"LightGBM CV RMSE:  {lgb_cv_rmse:.6f}")
print(f"XGBoost CV RMSE:   {xgb_cv_rmse:.6f}")
print(f"CatBoost CV RMSE:  {cat_cv_rmse:.6f}")
print(f"Ensemble CV RMSE:  {best_rmse:.6f}")

# Create submission
submission = sample_submission.copy()
submission['exam_score'] = ensemble_preds

submission.to_csv('submission_optimized.csv', index=False)
print("\nSubmission file created: submission_optimized.csv")
print(f"Submission shape: {submission.shape}")
print(f"\nFirst few predictions:")
print(submission.head(10))
print(f"\nPrediction statistics:")
print(f"Mean: {submission['exam_score'].mean():.2f}")
print(f"Std: {submission['exam_score'].std():.2f}")
print(f"Min: {submission['exam_score'].min():.2f}")
print(f"Max: {submission['exam_score'].max():.2f}")
