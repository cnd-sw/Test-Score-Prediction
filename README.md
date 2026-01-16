# Predicting Student Test Scores - Kaggle Competition

## Competition Overview
This is a solution for the Kaggle Playground Series S6E1 competition focused on predicting student test scores based on various features like study hours, class attendance, sleep patterns, and demographic information.

## Dataset
- **Train**: 630,000 samples
- **Test**: 270,000 samples
- **Target**: exam_score (continuous variable, range: 19.6 - 100.0)

### Features
1. **Numerical Features**:
   - age
   - study_hours
   - class_attendance
   - sleep_hours

2. **Categorical Features**:
   - gender (male, female, other)
   - course (b.sc, diploma, bca, b.com, ba, bba, b.tech)
   - internet_access (yes, no)
   - sleep_quality (poor, good, average)
   - study_method (online videos, self-study, coaching, group study, mixed)
   - facility_rating (low, medium, high)
   - exam_difficulty (easy, moderate, hard)

## Key Insights
- **study_hours** has the strongest correlation with exam_score (0.762)
- **class_attendance** is the second most important feature (0.361)
- **sleep_hours** has moderate correlation (0.167)
- Age has minimal correlation (0.010)

## Approach

### 1. Feature Engineering
We created extensive features including:

#### Interaction Features
- study_hours × class_attendance
- study_hours × sleep_hours
- class_attendance × sleep_hours
- Three-way interactions

#### Polynomial Features
- Squared and cubed transformations of key features
- Log and sqrt transformations

#### Ratio Features
- study_per_age
- sleep_per_age
- study_sleep_ratio
- attendance_sleep_ratio

#### Categorical Encodings
- Label encoding for basic categories
- Target encoding (mean encoding) for categorical variables
- Group aggregations (mean, std) by categorical features

#### Binning
- Age bins
- Study hours bins
- Attendance bins
- Sleep hours bins

#### Deviation Features
- Deviation from group means for various categorical groups

### 2. Models Used

#### LightGBM
- num_leaves: 255
- learning_rate: 0.02
- max_depth: 12
- Early stopping with 50 rounds

#### XGBoost
- max_depth: 10
- learning_rate: 0.02
- tree_method: hist
- Early stopping with 50 rounds

#### CatBoost
- depth: 10
- learning_rate: 0.02
- Early stopping with 50 rounds

### 3. Ensemble Strategy
- 5-fold cross-validation for robust predictions
- Weighted average ensemble of all three models
- Optimal weights found through grid search
- Out-of-fold predictions to prevent overfitting

### 4. Validation Strategy
- KFold cross-validation with 5 folds
- Stratified split with shuffle
- RMSE as evaluation metric

## Files

### Scripts
1. **explore_and_model.py**: Initial data exploration
2. **advanced_model.py**: Full ensemble with 10-fold CV
3. **ultra_optimized_model.py**: Optimized version with 5-fold CV and advanced features

### Outputs
- **submission.csv**: Predictions from advanced_model.py
- **submission_optimized.csv**: Predictions from ultra_optimized_model.py

## How to Run

### Install Dependencies
```bash
pip install pandas numpy scikit-learn lightgbm xgboost catboost matplotlib seaborn
```

### Run Exploration
```bash
python explore_and_model.py
```

### Train and Generate Predictions
```bash
# For full 10-fold ensemble (slower but more robust)
python advanced_model.py

# For optimized 5-fold ensemble (faster)
python ultra_optimized_model.py
```

## Expected Performance
- Individual model CV RMSE: ~8.7-8.8
- Ensemble CV RMSE: ~8.7 (target to beat top leaderboard scores)

## Strategy to Beat Top Score
1. **Extensive Feature Engineering**: Created 70+ features from 12 original features
2. **Multiple Strong Models**: Ensemble of LightGBM, XGBoost, and CatBoost
3. **Hyperparameter Tuning**: Optimized parameters for each model
4. **Robust Validation**: K-fold cross-validation to ensure generalization
5. **Optimal Ensemble Weights**: Grid search to find best model combination
6. **Target Encoding**: Leveraging target information for categorical features
7. **Group Statistics**: Capturing patterns within categorical groups

## Next Steps for Improvement
1. Add more complex interaction features
2. Try neural network models
3. Implement stacking with meta-learner
4. Feature selection to remove noise
5. Pseudo-labeling on test set
6. Try different ensemble techniques (blending, stacking)
