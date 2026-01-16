Overview
Welcome to the 2026 Kaggle Playground Series! We plan to continue in the spirit of previous playgrounds, providing interesting and approachable datasets for our community to practice their machine learning skills, and anticipate a competition each month.

Your Goal: Predict students' test scores.

Evaluation
link
keyboard_arrow_up
Submissions are evaluated using the Root Mean Squared Error between the predicted and the observed target.

Submission File

For each ID in the test set, you must predict a probability for the exam_score variable. The file should contain a header and have the following format:

id,exam_score
630000,97.5
630001,89.2
630002,85.5
etc.


Dataset Description
The dataset for this competition (both train and test) was generated from a deep learning model trained on the Exam score prediction dataset. Feature distributions are close to, but not exactly the same, as the original. Feel free to use the original dataset as part of this competition, both to explore differences as well as to see whether incorporating the original in training improves model performance.

Files

train.csv - the training set
test.csv - the test set
sample_submission.csv - a sample submission file in the correct format



The noteboom which gave the highest score is 

https://www.kaggle.com/code/ahmedabdulhamid/7adid-elsafina-ml-eda-1?scriptVersionId=292201190&cellId=1