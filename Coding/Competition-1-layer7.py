# %%
import pandas as pd
import numpy as np          # For mathematical calculations
import matplotlib.pyplot as plt  # For plotting graphs
# import seaborn as sns
# %matplotlib inline

train_csv_file_path = "D:\\ACA semester 7\\CS4622 - Machine Learning\\ML-Project\Data\\train.csv"
valid_csv_file_path = "D:\\ACA semester 7\\CS4622 - Machine Learning\\ML-Project\Data\\valid.csv"
test_csv_file_path = "D:\\ACA semester 7\\CS4622 - Machine Learning\\ML-Project\Data\\test.csv"

# %% [markdown]
# # Train dataset

# %%
train_df = pd.read_csv(train_csv_file_path)
train_df.head()

# %%
# get type of each column
train_df.dtypes

# %% [markdown]
# ## Renaming Labels
# 
# *   Since the labels are in 'label_1', 'label_2' ... format, I will be renaming them to 'speaker_ID', 'speaker_age', ... format
# 

# %%
train_df.rename(columns={'label_1': 'speaker_ID', 'label_2': 'speaker_age', 'label_3': 'speaker_gender', 'label_4': 'speaker_accent'}, inplace=True)

# %%
train_df.describe()

# %% [markdown]
# ## Check for null/NaN values in all columns
# 
# 

# %%
train_df.isna().any()

# Based on below output we can see that there are missing values in the speaker_age column of the dataset.
# Let's now check whether that is the only column with missing values.

# %%
train_df.isnull().sum()

# %%
train_df.isnull().sum().sum()

# Based on the above and below outputs, we can see that there are 480 missing values 'only' in the speaker_age column. No missing values in other columns.

# %%
print("train dataset shape:", train_df.shape)
print("null values row count: ", train_df.isnull().sum().sum())
print("null values row count percentage: ", (train_df.isnull().sum().sum() / train_df.shape[0]) * 100)

# %%
# Let's now check the distribution of the speaker_age column.
train_df.speaker_age.value_counts()

# %% [markdown]
# ## Handling Null values (Replace with Mean)

# %%
# Let's get the mean of the speaker_age column.
speaker_age_mean = train_df.speaker_age.mean()
print("mean: ", speaker_age_mean)

# round it to nearest int
speaker_age_mean = round(speaker_age_mean)

# %%
# Let's now fill the missing values with the mean value.
train_df.speaker_age.fillna(speaker_age_mean, inplace=True)

# %% [markdown]
# ## Checking each Label distribution

# %% [markdown]
# ### Speaker age

# %%
# Let's now check the distribution of the speaker_age column.
train_df.speaker_age.value_counts()

# %%
# Number os unique values in speaker age column
train_df.speaker_age.nunique()

# %%
train_df.speaker_age.value_counts().plot.bar()

# %%
# There is a slight class imbalance issue based on above outputs. As a solution we can use RandomForrestClassifier with class_weight='balanced' parameter.

# Note; hpt: forest_params = [{'max_depth': list(range(10, 15)), 'max_features': list(range(0,14))}]

# use averaged F1 score 

# %% [markdown]
# ### Speaker ID

# %%
# Let's now check the distribution of the speaker_ID column.
train_df.speaker_ID.value_counts()

# %%
# Let's get num of unique values in speaker_ID column.
train_df.speaker_ID.nunique()

# %%
train_df.speaker_ID.value_counts().plot.bar()

# There is no significant class imbalance issue based on above outputs for the Speaker_ID column

# %% [markdown]
# ### Speaker gender

# %%
# Let's now check the distribution of the speaker_age column.
train_df.speaker_gender.value_counts()

# %%
train_df.speaker_gender.nunique()

# %%
train_df.speaker_gender.value_counts().plot.bar()

# There is a significant class imbalance issue in speaker_gender column. As a solution we can use RandomForrestClassifier with class_weight='balanced' parameter.
# Also when splitting the dataset, I will use the stratisfied sampling technique.

# Note; hpt: forest_params = [{'max_depth': list(range(10, 15)), 'max_features': list(range(0,14))}]

# NOte; print classification report | use averaged/macro F1

# # Define the steps in your pipeline
# steps = [
#     ('scaler', StandardScaler()),  # Standardize the features
#     ('oversampler', SMOTE(random_state=42)),  # Apply SMOTE for oversampling
#     ('pca', PCA(n_components=0.95)),  # Apply PCA for dimensionality reduction
#     ('xgb', XGBClassifier(scale_pos_weight=np.sqrt(np.sum(y == 0) / np.sum(y == 1))))  # XGBoost Classifier
# ]

# The parameter scale_pos_weight is used in XGBoost to address class imbalance. It's an important hyperparameter to consider when working with imbalanced datasets. The specific value provided (np.sqrt(np.sum(y == 0) / np.sum(y == 1))) is a common heuristic used for setting scale_pos_weight, but it should be chosen based on the characteristics of your dataset.

# %% [markdown]
# ### Speaker accent

# %%
# Let's now check the distribution of the speaker_age column.
train_df.speaker_accent.value_counts()

# %%
train_df.speaker_accent.nunique()

# %%
train_df.speaker_accent.value_counts().plot.bar()   

# There is a significant class imbalance issue in speaker_accents column. As a solution we can use RandomForrestClassifier with class_weight='balanced' parameter.
# Also when splitting the dataset, I will use the stratisfied sampling technique.

# Note; hpt: forest_params = [{'max_depth': list(range(10, 15)), 'max_features': list(range(0,14))}]

# use avraged F1/macro F1

# <!-- # Make predictions on the test set -->
# y_pred = classifier_pipeline.predict(X_test)

# <!-- # Generate a classification report -->
# report = classification_report(y_test, y_pred)

# <!-- # Print the classification report -->
# print("Classification Report:\n", report)

# %% [markdown]
# # Validation dataset

# %%
valid_df = pd.read_csv(valid_csv_file_path)
valid_df.head()

# %%
# get type of each column
valid_df.dtypes

# %% [markdown]
# ## Renaming Labels
# 
# *   Since the labels are in 'label_1', 'label_2' ... format, I will be renaming them to 'speaker_ID', 'speaker_age', ... format
# 

# %%
valid_df.rename(columns={'label_1': 'speaker_ID', 'label_2': 'speaker_age', 'label_3': 'speaker_gender', 'label_4': 'speaker_accent'}, inplace=True)

# %%
valid_df.describe()

# %% [markdown]
# ## Check for null/NaN values in all columns
# 
# 

# %%
valid_df.isna().any()

# Based on below output we can see that there are missing values in the speaker_age column of the dataset.
# Let's now check whether that is the only column with missing values.

# %%
valid_df.isnull().sum()

# %%
valid_df.isnull().sum().sum()

# Based on the above and below outputs, we can see that there are 480 missing values 'only' in the speaker_age column. No missing values in other columns.

# %%
print("validation dataset shape:", valid_df.shape)
print("null values row count: ", valid_df.isnull().sum().sum())
print("null values row count percentage: ", (valid_df.isnull().sum().sum() / valid_df.shape[0]) * 100)

# %%
# Let's now check the distribution of the speaker_age column.
valid_df.speaker_age.value_counts()

# %% [markdown]
# ## Handling Null values (Replace with Mean)

# %%
# Let's get the mean of the speaker_age column.
speaker_age_mean = valid_df.speaker_age.mean()

# round it to nearest int
speaker_age_mean = round(speaker_age_mean)
print("mean: ", speaker_age_mean)

# %%
# Let's now fill the missing values with the mean value.
valid_df.speaker_age.fillna(speaker_age_mean, inplace=True)

# %%
valid_df.speaker_age.value_counts()

# %%
# Preparing training and validation datasets
from sklearn.model_selection import train_test_split

train_X = train_df.drop(['speaker_ID', 'speaker_age', 'speaker_gender', 'speaker_accent'], axis=1)
train_speaker_IDs = train_df['speaker_ID']
train_speaker_ages = train_df['speaker_age']
train_speaker_genders = train_df['speaker_gender']
train_speaker_accents = train_df['speaker_accent']

valid_X = valid_df.drop(['speaker_ID', 'speaker_age', 'speaker_gender', 'speaker_accent'], axis=1)
valid_speaker_IDs = valid_df['speaker_ID']
valid_speaker_ages = valid_df['speaker_age']
valid_speaker_genders = valid_df['speaker_gender']
valid_speaker_accents = valid_df['speaker_accent']


# %% [markdown]
# # Test dataset

# %%
test_df = pd.read_csv(test_csv_file_path)
test_df.head()

# %%
# get type of each column
test_df.dtypes

# %%
test_df.isna().any()

# %%
test_df.isnull().sum().sum()

# based on above output we can see that there are no missing values in the test dataset

# %%
test_X = test_df.drop(["ID"], axis=1)
print("test dataset shape:", test_X.shape)
test_X.head()

# %% [markdown]
# # Developing the Pipelines

# %%
# Let's now develop pipelines to predict the speaker ID, speaker age, speaker gender, and speaker accent. 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFECV
from sklearn.svm import LinearSVC
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import GridSearchCV

# %% [markdown]
# ## For Speaker_IDs

# %%
#  StandardScaler => For ensuring that all features have the same scale, which is often crucial for the proper functioning of many machine learning algorithms.
#  PCA => For dimensionality reduction
#  SVC => Support Vector Classifier

# Pipeline for speaker ID prediction without feature engineering (to check raw accuracy)
speaker_ID_pipe_svc = Pipeline([ 
    ('clf', SVC())
    ])


# Pipleline for speaker ID prediction with PCA for feature reduction
speaker_ID_pipe_scaler_pca_svc = Pipeline([
    # ('scaler', StandardScaler()),
    ('pca', PCA(n_components=0.95)),
    ('clf', SVC())
    ])

# Pipleline for speaker ID prediction with Model-based feature reduction
speaker_ID_pipe_scaler_sfmlr_svc = Pipeline([
    # ('scaler', StandardScaler()),
    ('SFM_LR', SelectFromModel(LogisticRegression(C=0.01, penalty='l1', solver='liblinear', class_weight='balanced'))),
    ('clf', SVC())  
])

# %%
temp = Pipeline([
        ('pca', PCA(n_components=0.95)),
        ('clf', SVC())
    ])

# %%
temp.fit(train_X, train_speaker_IDs)


# %%
temp.score(valid_X, valid_speaker_IDs)

# %%


# %%


# %%
##visualize Pipeline
from sklearn import set_config
set_config(display='diagram')

# %%
train_X.shape, train_speaker_IDs.shape, train_speaker_genders.shape, train_speaker_accents.shape

# %% [markdown]
# ### Training

# %%
# Pipeine without feature engineering
speaker_ID_pipe_svc.fit(train_X, train_speaker_IDs)

# %%
# Pipeine with PCA for feature reduction
speaker_ID_pipe_scaler_pca_svc.fit(train_X, train_speaker_IDs)

# %%
# Pipeine with Model-based feature reduction (SelectFromModel - LogisticRegression)
speaker_ID_pipe_scaler_sfmlr_svc.fit(train_X, train_speaker_IDs)

# %%
# Let's now check the accuracies.
print("Without feature engineering: ", speaker_ID_pipe_svc.score(valid_X, valid_speaker_IDs)*100, "%")
print("With PCA for feature reduction: ", speaker_ID_pipe_scaler_pca_svc.score(valid_X, valid_speaker_IDs)*100, "%")
print("With Model-based feature reduction: ", speaker_ID_pipe_scaler_sfmlr_svc.score(valid_X, valid_speaker_IDs)*100, "%")

# %%
num_features_before_reduction = speaker_ID_pipe_svc.named_steps['clf'].n_features_in_
num_features_after_pca = speaker_ID_pipe_scaler_pca_svc.named_steps['pca'].n_components_
num_features_after_sfmlr = sum(speaker_ID_pipe_scaler_sfmlr_svc.named_steps['SFM_LR'].get_support())

print("Number of features before reduction: ", num_features_before_reduction)
print("Number of features after PCA: ", num_features_after_pca)
print("Number of features after model-based feature reduction: ", num_features_after_sfmlr)

# Based on the above output accuracies and the reduced num of features, we can see that the model with model-based feature reduction has the highest accuracy (96%). Also it reduced the num of features to 693 from 768
# Also, the PCA model reduced the accuracy slightly (by ~0.2%), but it has reduced the um of features to 321 from 768

# %% [markdown]
# ### Hyperparameter tuning

# %%
# Let's now try to improve the accuracy of the model with PCA for feature reduction by tuning the hyperparameters of the model.

param_grid = dict(PCA__n_components=[0.95, 0.96, 0.97],
                  clf__C=np.logspace(-2, 1, 4),
                  clf__kernel=['rbf','linear',  'poly'])

grid = GridSearchCV(speaker_ID_pipe_scaler_pca_svc, param_grid=param_grid, cv=3, n_jobs=1, verbose=2, scoring= 'accuracy')

# %%
grid.fit(train_X, train_speaker_IDs)
print(grid.best_score_)
print(grid.cv_results_)

# %%
# Print the best hyperparameters and corresponding accuracy
print("Best Hyperparameters: ", grid.best_params_)
print("Best Accuracy: {:f}%".format(grid.best_score_ * 100))

# Therefore the best hyperparameters based on gridSearchCV : 
# - 'PCA__n_components': 0.97, 
# - 'clf__C': 10.0, 
# - 'clf__kernel': 'rbf'

# %%
best_PCA__n_components = 0.97
best_clf__C = 10.0
best_clf__kernel = 'rbf'

# Pipleline with best params
best_speaker_ID_pipe_scaler_pca_svc = Pipeline([
    # ('scaler', StandardScaler()),
    ('pca', PCA(n_components=best_PCA__n_components)),
    ('clf', SVC(C=best_clf__C, kernel=best_clf__kernel))
    ])

best_speaker_ID_pipe_scaler_pca_svc.fit(train_X, train_speaker_IDs)
print("PCA for feature reduction (Hyperparameters tuned): ", best_speaker_ID_pipe_scaler_pca_svc.score(valid_X, valid_speaker_IDs)*100, "%")

# best_svm = grid.best_estimator_
# pred_valid_speaker_IDs = best_svm.predict(valid_X)  
# validation_accuracy = accuracy_score(valid_speaker_IDs, pred_valid_speaker_IDs)
# print("Validation Accuracy:", validation_accuracy)

# %%
# Let's now try to improve the accuracy of the model with model-based for feature reduction by tuning the hyperparameters of the model.
param_grid = dict(SFM_LR__estimator__C=np.logspace(-2, 1, 4),
                    clf__C=np.logspace(-2, 1, 4),
                    clf__kernel=['rbf'])

grid = GridSearchCV(speaker_ID_pipe_scaler_sfmlr_svc, param_grid=param_grid, cv=3, n_jobs=1, verbose=2, scoring= 'accuracy')

# %%
grid.fit(train_X, train_speaker_IDs)
print(grid.best_score_)
print(grid.cv_results_) 

# %%
# Print the best hyperparameters and corresponding accuracy
print("Best Hyperparameters: ", grid.best_params_)
print("Best Accuracy: {:f}%".format(grid.best_score_ * 100))

# Therefore the best hyperparameters based on gridSearchCV : 
# - 'SFM_LR__estimator__C': 0.01, 
# - 'clf__C': 10.0, 
# - 'clf__kernel': 'rbf'

# %%
best_SFM_LR__estimator__C = 0.01
best_clf__C = 10.0
best_clf__kernel = 'rbf'

# Pipleline with best params
best_speaker_ID_pipe_scaler_sfmlr_svc = Pipeline([
    # ('scaler', StandardScaler()),
    ('SFM_LR', SelectFromModel(LogisticRegression(C=best_SFM_LR__estimator__C, penalty='l1', solver='liblinear', class_weight='balanced'))),
    ('clf', SVC(kernel=best_clf__kernel, C=best_clf__C))  
])

best_speaker_ID_pipe_scaler_sfmlr_svc.fit(train_X, train_speaker_IDs)
print("Model-based feature reduction (Hyperparameters tuned): ", best_speaker_ID_pipe_scaler_sfmlr_svc.score(valid_X, valid_speaker_IDs)*100, "%")


# best_svm = grid.best_estimator_
# pred_valid_speaker_IDs = best_svm.predict(valid_X)  
# validation_accuracy = accuracy_score(valid_speaker_IDs, pred_valid_speaker_IDs)
# print("Validation Accuracy:", validation_accuracy)

# %%
print("PCA for feature reduction (Hyperparameters tuned): ", best_speaker_ID_pipe_scaler_pca_svc.score(valid_X, valid_speaker_IDs))
print("Model-based feature reduction (Hyperparameters tuned): ", best_speaker_ID_pipe_scaler_sfmlr_svc.score(valid_X, valid_speaker_IDs))

# %% [markdown]
# ### Prediction on test data 

# %%
# Let's use best performing pipeline to make predictions for the test data
pred_test = best_speaker_ID_pipe_scaler_pca_svc.predict(test_X)
print(pred_test.shape)

# %%
pred_test = pd.DataFrame(pred_test, columns=['label_1'])
pred_test.head()

# %%
if "ID" not in pred_test.columns:
    pred_test.insert(0, "ID", test_df['ID'])
else:
    print(f"Column : ID already exists")

# %%
pred_test.head()

# %% [markdown]
# ## For Speaker_age

# %%
# from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE  

# %%
#  Note: 
# There is a slight class imbalance issue in speaker_age values. As a solution we can use class_weight='balanced' parameter.

# Note; hpt: forest_params = [{'max_depth': list(range(10, 15)), 'max_features': list(range(0,14))}]

# use averaged F1 score 

# Pipeline for speaker ID prediction without feature engineering (to check raw accuracy)

#Let's choose a good classifer

speaker_age_pipe_svc = Pipeline([ 
    ('clf', SVC(class_weight='balanced'))
    ])

speaker_age_pipe_dtc = Pipeline([ 
    ('clf', DecisionTreeClassifier(class_weight='balanced'))
    ])

classes = np.unique(train_speaker_ages)
cw = class_weight.compute_class_weight(class_weight='balanced', classes=classes, y=train_speaker_ages)
weights = dict(zip(classes,cw))
class_weighted_model = DecisionTreeClassifier(class_weight=weights)

speaker_age_pipe_dtc2 = Pipeline([ 
    ('clf', class_weighted_model)
    ])

speaker_age_pipe_rfc = Pipeline([ 
    ('clf', RandomForestClassifier(class_weight='balanced'))
    ])

# %%
##visualize Pipeline
from sklearn import set_config
set_config(display='diagram')

# %% [markdown]
# ### Training

# %%
# Pipeine without feature engineering
speaker_age_pipe_svc.fit(train_X, train_speaker_ages)

# %%
# Pipeine with PCA for feature reduction
speaker_age_pipe_dtc.fit(train_X, train_speaker_ages)

# %%
# Pipeine with PCA for feature reduction
speaker_age_pipe_dtc2.fit(train_X, train_speaker_ages)

# %%
# Pipeine with PCA for feature reduction
speaker_age_pipe_rfc.fit(train_X, train_speaker_ages)

# %%
# Now let's check the accuracies
print("With SVC classifier:", speaker_age_pipe_svc.score(valid_X, valid_speaker_ages)) 
print("With Decision Tree classifier (class_weight='balanced'):", speaker_age_pipe_dtc.score(valid_X, valid_speaker_ages))
print("With Decision Tree classifier (class_weights manuelly calculated):", speaker_age_pipe_dtc2.score(valid_X, valid_speaker_ages)) 
print("With Random Forest classifier (class_weight='balanced'):", speaker_age_pipe_rfc.score(valid_X, valid_speaker_ages))

# %%
# Based on the above outputs SVC with class_weight='balanced' parameter performs better
# Now let's check what feature engineering technique would be better

# Pipleline for speaker age prediction with PCA for feature reduction
speaker_age_pipe_scaler_pca_svc = Pipeline([
    # ('scaler', StandardScaler()),
    ('pca', PCA(n_components=0.95)),
    ('clf', SVC(class_weight='balanced'))
    ])

# Pipleline for speaker age prediction with Model-based feature reduction
speaker_age_pipe_scaler_sfmlr_svc = Pipeline([
    # ('scaler', StandardScaler()),
    ('SFM_LR', SelectFromModel(LogisticRegression(C=0.01, penalty='l1', solver='liblinear', class_weight='balanced'))),
    ('clf', SVC(class_weight='balanced'))  
])

# %%
# Pipeine with PCA for feature reduction
speaker_age_pipe_scaler_pca_svc.fit(train_X, train_speaker_ages)

# %%
# Pipeine with PCA for feature reduction
speaker_age_pipe_scaler_sfmlr_svc.fit(train_X, train_speaker_ages)

# %%
# Let's now check the accuracies.
print("Withouth feature engineering: ", speaker_age_pipe_svc.score(valid_X, valid_speaker_ages))
print("With PCA for feature reduction: ", speaker_age_pipe_scaler_pca_svc.score(valid_X, valid_speaker_ages))
print("With Model-based feature reduction: ", speaker_age_pipe_scaler_sfmlr_svc.score(valid_X, valid_speaker_ages))

# %%
num_features_before_reduction = speaker_age_pipe_svc.named_steps['clf'].n_features_in_
num_features_after_pca = speaker_age_pipe_scaler_pca_svc.named_steps['pca'].n_components_
num_features_after_sfmlr = sum(speaker_age_pipe_scaler_sfmlr_svc.named_steps['SFM_LR'].get_support())

print("Number of features before reduction: ", num_features_before_reduction)
print("Number of features after PCA: ", num_features_after_pca)
print("Number of features after model-based feature reduction: ", num_features_after_sfmlr)

# %% [markdown]
# ### Hyperparameter tuning

# %%
# Let's now try to improve the accuracy of the model with PCA for feature reduction by tuning the hyperparameters of the model.

# NOTE: When 'cv' parameter is used in the GridSearchCV ==> it does Stratified Croos Validation which is good when class imbalance is there 
param_grid = dict(pca__n_components=[0.95, 0.96, 0.97],
                  clf__C=np.logspace(-2, 1, 4),
                  clf__kernel=['rbf'])

grid = GridSearchCV(speaker_age_pipe_scaler_pca_svc, param_grid=param_grid, cv=3, n_jobs=1, verbose=2, scoring= 'balanced_accuracy') ####CHANGED DIDNOTRUN

# %%
grid.fit(train_X, train_speaker_ages)
print(grid.best_score_)
print(grid.cv_results_)

# %%
# Print the best hyperparameters and corresponding accuracy
print("Best Hyperparameters: ", grid.best_params_)
print("Best Accuracy: {:f}%".format(grid.best_score_))

# Therefore the best hyperparameters based on gridSearchCV : 
# - 'pca__n_components': , 
# - 'clf__C': , 
# - 'clf__kernel': 'rbf'

# %%
best_pca__n_components = 0.97
best_clf__C = 10.0
best_clf__kernel = 'rbf'

# Pipleline with best params
best_speaker_age_pipe_scaler_pca_svc = Pipeline([
    # ('scaler', StandardScaler()),
    ('pca', PCA(n_components=best_pca__n_components)),
    ('clf', SVC(C=best_clf__C, kernel=best_clf__kernel))
    ])

best_speaker_age_pipe_scaler_pca_svc.fit(train_X, train_speaker_ages)
print("PCA for feature reduction (Hyperparameters tuned): ", best_speaker_age_pipe_scaler_pca_svc.score(valid_X, valid_speaker_ages))

# %%
# Let's now try to improve the accuracy of the model with model-based for feature reduction by tuning the hyperparameters of the model.
param_grid = dict(SFM_LR__estimator__C=np.logspace(-2, 1, 4),
                    clf__C=[10.0],
                    clf__kernel=['rbf'])

grid = GridSearchCV(speaker_age_pipe_scaler_sfmlr_svc, param_grid=param_grid, cv=3, n_jobs=1, verbose=2, scoring= 'balanced_accuracy')
### CHANGED DIDNOT RUN

# %%
grid.fit(train_X, train_speaker_ages)
print(grid.best_score_)
print(grid.cv_results_) 

# %%
# Print the best hyperparameters and corresponding accuracy
print("Best Hyperparameters: ", grid.best_params_)
print("Best Accuracy: {:f}%".format(grid.best_score_ * 100))

# Therefore the best hyperparameters based on gridSearchCV : 
# - 'SFM_LR__estimator__C': , 
# - 'clf__C': , 
# - 'clf__kernel': 'rbf'

# %%
best_SFM_LR__estimator__C = 1
best_clf__C = 10.0
best_clf__kernel = 'rbf'

# Pipleline with best params
best_speaker_age_pipe_scaler_sfmlr_svc = Pipeline([
    ('scaler', StandardScaler()),
    ('SFM_LR', SelectFromModel(LogisticRegression(C=best_SFM_LR__estimator__C, penalty='l1', solver='liblinear', class_weight='balanced'))),
    ('clf', SVC(class_weight='balanced', C=best_clf__C, kernel=best_clf__kernel))  
])

best_speaker_age_pipe_scaler_sfmlr_svc.fit(train_X, train_speaker_ages)
print("Model-based feature reduction (Hyperparameters tuned): ", best_speaker_age_pipe_scaler_sfmlr_svc.score(valid_X, valid_speaker_ages))


# best_svm = grid.best_estimator_
# pred_valid_speaker_IDs = best_svm.predict(valid_X)  
# validation_accuracy = accuracy_score(valid_speaker_IDs, pred_valid_speaker_IDs)
# print("Validation Accuracy:", validation_accuracy)

# %% [markdown]
# ### Prediction on test data

# %%
test_X.head()

# %%
# Let's use best performing pipeline to make predictions for the test data
pred_speaker_ages_test = best_speaker_age_pipe_scaler_pca_svc.predict(test_X)
print(pred_speaker_ages_test.shape)

# %%
pred_speaker_age_test = pd.DataFrame(pred_speaker_ages_test, columns=['label_2'])
pred_speaker_age_test.head()

# %%
pred_test.head() # pred_test was already created (when doing prediction of speaker_ID)

# %%
if "label_2" not in pred_test.columns:
    pred_test.insert(2, "label_2", pred_speaker_age_test['label_2'])
else:
    print(f"Column : label_2 already exists")

# %%
pred_test.head()

# %% [markdown]
# ## For speaker_gender

# %%
train_df.speaker_gender.value_counts().plot.bar()

# %%
#  Note: 
# There is a class imbalance issue in speaker_gender values. As a solution we can use class_weight='balanced' parameter.

# use averaged F1 score 

#Let's choose a good classifer

speaker_gender_pipe_svc = Pipeline([ 
    ('clf', SVC(class_weight='balanced'))
    ])

speaker_gender_pipe_dtc = Pipeline([ 
    ('clf', DecisionTreeClassifier(class_weight='balanced'))
    ])

classes = np.unique(train_speaker_genders)
cw = class_weight.compute_class_weight(class_weight='balanced', classes=classes, y=train_speaker_genders)
weights = dict(zip(classes,cw))
class_weighted_model = DecisionTreeClassifier(class_weight=weights)

speaker_gender_pipe_dtc2 = Pipeline([ 
    ('clf', class_weighted_model)
    ])

speaker_gender_pipe_rfc = Pipeline([ 
    ('clf', RandomForestClassifier(class_weight='balanced'))
])

# %%
##visualize Pipeline
from sklearn import set_config
set_config(display='diagram')

# %% [markdown]
# ### Training

# %%
# Pipeine without feature engineering
speaker_gender_pipe_svc.fit(train_X, train_speaker_genders)

# %%
# Pipeine with PCA for feature reduction
speaker_gender_pipe_dtc.fit(train_X, train_speaker_genders)

# %%
# Pipeine with PCA for feature reduction
speaker_gender_pipe_dtc2.fit(train_X, train_speaker_genders)

# %%
# Pipeine with PCA for feature reduction
speaker_gender_pipe_rfc.fit(train_X, train_speaker_genders)

# %%
# now let's check the results
print("With SVC classifier:") 
print(classification_report(valid_speaker_genders, speaker_gender_pipe_svc.predict(valid_X)))
print("With Decision Tree classifier (class_weight='balanced'):") 
print(classification_report(valid_speaker_genders, speaker_gender_pipe_dtc.predict(valid_X)))
print("With Decision Tree classifier (class_weights manuelly calculated):") 
print(classification_report(valid_speaker_genders, speaker_gender_pipe_dtc2.predict(valid_X)))
print("With Random Forest classifier (class_weight='balanced'):") 
print(classification_report(valid_speaker_genders, speaker_gender_pipe_rfc.predict(valid_X)))


# %% [markdown]
# ### Prediction on test data

# %%
# Pipeline with SVC (class weights balanced) version gave perferct results (1.00 for all metrics in classification report) ==> Therefore let's use that for the prediction of test data

best_speaker_gender_pipe = speaker_gender_pipe_svc
pred_speaker_genders_test = best_speaker_gender_pipe.predict(test_X)
print(pred_speaker_genders_test.shape)

# %%
pred_speaker_genders_test = pd.DataFrame(pred_speaker_genders_test, columns=['label_3'])
pred_speaker_genders_test.head()

# %%
pred_test.head() # pred_test was already created (when doing prediction of speaker_ID)

# %%
if "label_3" not in pred_test.columns:
    pred_test.insert(3, "label_3", pred_speaker_genders_test['label_3'])
else:
    print(f"Column : label_3 already exists")

# %%
pred_test.head()

# %% [markdown]
# ## For speaker_accent

# %%
train_df.speaker_accent.value_counts().plot.bar()

# %%
# Note: 
# There is a class imbalance issue in speaker_accent values. As a solution we can use class_weight='balanced' parameter.

# use averaged F1 score 

#Let's choose a good classifer

speaker_accent_pipe_svc = Pipeline([ 
    ('clf', SVC(class_weight='balanced'))
    ])

speaker_accent_pipe_dtc = Pipeline([ 
    ('clf', DecisionTreeClassifier(class_weight='balanced'))
    ])

classes = np.unique(train_speaker_accents)
cw = class_weight.compute_class_weight(class_weight='balanced', classes=classes, y=train_speaker_accents)
weights = dict(zip(classes,cw))
class_weighted_model = DecisionTreeClassifier(class_weight=weights)

speaker_accent_pipe_dtc2 = Pipeline([ 
    ('clf', class_weighted_model)
    ])

speaker_accent_pipe_rfc = Pipeline([ 
    ('clf', RandomForestClassifier(class_weight='balanced'))
])

# %%
##visualize Pipeline
from sklearn import set_config
set_config(display='diagram')

# %% [markdown]
# ### Training

# %%
speaker_accent_pipe_svc.fit(train_X, train_speaker_accents)

# %%
speaker_accent_pipe_dtc.fit(train_X, train_speaker_accents)

# %%
speaker_accent_pipe_dtc2.fit(train_X, train_speaker_accents)

# %%
speaker_accent_pipe_rfc.fit(train_X, train_speaker_accents)

# %%
# now let's check the results =>Let's consider weighted avg metric when making desicions because it is a good metric when class imbalance is present 
print("With SVC classifier:") 
print(classification_report(valid_speaker_accents, speaker_accent_pipe_svc.predict(valid_X)))
print("With Decision Tree classifier (class_weight='balanced'):") 
print(classification_report(valid_speaker_accents, speaker_accent_pipe_dtc.predict(valid_X)))
print("With Decision Tree classifier (class_weights manuelly calculated):") 
print(classification_report(valid_speaker_accents, speaker_accent_pipe_dtc2.predict(valid_X)))
print("With Random Forest classifier (class_weight='balanced'):") 
print(classification_report(valid_speaker_accents, speaker_accent_pipe_rfc.predict(valid_X)))


# %%
# Based on above results SVC with class_weight='balanced' performs well.
# Let's now try some feature engineering techniques to check whether it improves weighted avg metric

# Pipleline for speaker accent prediction with PCA for feature reduction
speaker_accent_pipe_scaler_pca_svc = Pipeline([
    # ('scaler', StandardScaler()),
    ('pca', PCA(n_components=0.95)),
    ('clf', SVC(class_weight='balanced'))
    ])

# Pipleline for speaker accent prediction with Model-based feature reduction
speaker_accent_pipe_scaler_sfmlr_svc = Pipeline([
    # ('scaler', StandardScaler()),
    ('SFM_LR', SelectFromModel(LogisticRegression(C=0.01, penalty='l1', solver='liblinear', class_weight='balanced'))),
    ('clf', SVC(class_weight='balanced'))  
])


# %%
speaker_accent_pipe_scaler_pca_svc.fit(train_X, train_speaker_accents)

# %%
speaker_accent_pipe_scaler_sfmlr_svc.fit(train_X, train_speaker_accents)

# %%
# Let's now check the accuracies.
print("Withouth feature engineering: ")
print(speaker_accent_pipe_svc.score(valid_X, valid_speaker_accents))
print("With PCA for feature reduction: ")
print(speaker_accent_pipe_scaler_pca_svc.score(valid_X, valid_speaker_accents))
print("With Model-based feature reduction: ")
print(speaker_accent_pipe_scaler_sfmlr_svc.score(valid_X, valid_speaker_accents))

# %%
num_features_before_reduction = speaker_accent_pipe_svc.named_steps['clf'].n_features_in_
num_features_after_pca = speaker_accent_pipe_scaler_pca_svc.named_steps['pca'].n_components_
num_features_after_sfmlr = sum(speaker_accent_pipe_scaler_sfmlr_svc.named_steps['SFM_LR'].get_support())

print("Number of features before reduction: ", num_features_before_reduction)
print("Number of features after PCA: ", num_features_after_pca)
print("Number of features after model-based feature reduction: ", num_features_after_sfmlr)

# %%
best_pca__n_components = 0.97
best_clf__C = 10.0
best_clf__kernel = 'rbf'

# Pipleline with best params
best_accent_pipe_scaler_pca_svc = Pipeline([
    # ('scaler', StandardScaler()),
    ('pca', PCA(n_components=best_pca__n_components)),
    ('clf', SVC(C=best_clf__C, kernel=best_clf__kernel))
    ])

best_accent_pipe_scaler_pca_svc.fit(train_X, train_speaker_accents)
print("PCA for feature reduction (Hyperparameters tuned): ", best_accent_pipe_scaler_pca_svc.score(valid_X, valid_speaker_accents))

# %%
classification_report(valid_speaker_accents, best_accent_pipe_scaler_pca_svc.predict(valid_X))

# %% [markdown]
# ### Hyperparameter tuning

# %%
# Let's now try to improve the accuracy of the model with PCA for feature reduction by tuning the hyperparameters of the model.

# NOTE: When 'cv' parameter is used in the GridSearchCV ==> it does Stratified Croos Validation which is good when class imbalance is there 

# The "balanced accuracy" in binary and multiclass classification problems to deal with imbalanced datasets. It is defined as the average of recall obtained on each class.
param_grid = dict(pca__n_components=[0.95, 0.96, 0.97],
                  clf__C=np.logspace(-2, 1, 4),
                  clf__kernel=['rbf'])

grid = GridSearchCV(speaker_accent_pipe_scaler_pca_svc, param_grid=param_grid, cv=3, n_jobs=1, verbose=2, scoring= 'balanced_accuracy')

# %% [markdown]
# ### Prediction test data

# %%
test_X.head()

# %%
pred_speaker_accents_test = best_accent_pipe_scaler_pca_svc.predict(test_X)
print(pred_speaker_accents_test.shape)

# %%
pred_speaker_accents_test = pd.DataFrame(pred_speaker_accents_test, columns=['label_4'])
pred_speaker_accents_test.head()

# %%
pred_test.head() # pred_test was already created (when doing prediction of speaker_ID)


# %%
if "label_4" not in pred_test.columns:
    pred_test.insert(4, "label_4", pred_speaker_accents_test['label_4'])
else:
    print(f"Column : label_4 already exists")

# %%
pred_test_layer7 = pred_test
pred_test_layer7.head() 

# %%
pred_test_layer7.to_csv('190290U_pred_test_layer7.csv', index=False)

# %%
# #################### RUN BELOW FOR THE BEST MODEL ####################

# # After grid search is complete, save the best estimator
# best_svm = grid_search.best_estimator_

# # You can also save the best hyperparameters
# best_params = grid_search.best_params_

# # Now, you can use the best estimator for making predictions or further operations
# y_pred = best_svm.predict(X_validation)  # Example: Using it for predictions on validation data

# # If you want to create a new instance of the model with the best hyperparameters
# svm_with_best_params = SVC(**best_params)
# svm_with_best_params.fit(X_train, y_train)  # Train the new model


# %%
# all other labels (age, gender, accent) have class imbalance problem. So account for that -> I think cv parameter in grid search cv can be used for that
# read about it ==> https://scikit-learn.org/stable/modules/cross_validation.html (Cross validation)
# This is vital because, we cannot track the model performance using a single metric
# if cv is given an integer value, it will use stratified k fold cross validation




# %%
# TSNE : feature rection technique

# from sklearn.model_selection import RandomizedSearchCV
# from scipy.stats import uniform

# # Define a range of hyperparameters and their distributions for random search
# param_dist = {
#     'clf__C': uniform(loc=0.1, scale=10),  # Regularization parameter (uniform distribution)
#     'clf__kernel': ['linear', 'rbf', 'poly'],  # Kernel type
# }

# # Create a random search object
# random_search = RandomizedSearchCV(estimator=speaker_ID_pipe_scaler_svc, param_distributions=param_dist, n_iter=10, cv=5, scoring='accuracy', random_state=42)

# # Fit the random search to the data
# random_search.fit(X_train, y_train)

# # Get the best hyperparameters
# best_params_random = random_search.best_params_
# best_accuracy_random = random_search.best_score_

# print("Best Hyperparameters (Random Search):", best_params_random)
# print("Best Accuracy (Random Search):", best_accuracy_random)


# %%
# code to compare different pipelines

# from sklearn.datasets import load_iris
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA
# from sklearn.manifold import TSNE
# from sklearn.svm import SVC
# from sklearn.pipeline import Pipeline
# from sklearn.metrics import accuracy_score

# # Load a sample dataset (Iris dataset in this example)
# data = load_iris()
# X, y = data.data, data.target

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Create a common feature scaling pipeline
# common_preprocessing = Pipeline([
#     ('scaler', StandardScaler())
# ])

# # Define pipelines with different feature reduction techniques on top of common preprocessing
# # You can add more as needed
# pca_pipeline = Pipeline([
#     ('preprocessing', common_preprocessing),
#     ('pca', PCA(n_components=2)),  # Adjust the number of components as needed
#     ('clf', SVC()),
# ])

# tsne_pipeline = Pipeline([
#     ('preprocessing', common_preprocessing),
#     ('tsne', TSNE(n_components=2)),  # Adjust the number of components as needed
#     ('clf', SVC()),
# ])

# selectkbest_pipe = Pipeline([
#     ('preprocessing', common_preprocessing),
#     ('SelectKBest', SelectKBest(chi2, k=10)),
#     ('clf', SVC())
# ])

# # Create a list of feature reduction pipelines for easy iteration
# feature_reduction_pipelines = [
#     ("PCA", pca_pipeline),
#     ("t-SNE", tsne_pipeline),
# ]

# # Fit and compare the performance of different feature reduction pipelines
# for name, pipeline in feature_reduction_pipelines:
#     # Fit the pipeline on the training data
#     pipeline.fit(X_train, y_train)
    
#     # Make predictions on the testing data
#     y_pred = pipeline.predict(X_test)
    
#     # Calculate and print the accuracy
#     accuracy = accuracy_score(y_test, y_pred)
#     print(f"Accuracy with {name}: {accuracy:.2f}")

########################################################################################
###########So we effectively fit the scaler pipeline 2 times right? is it the case?################################
########################################################################################
########################################################################################

# from sklearn.datasets import load_iris
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA
# from sklearn.manifold import TSNE
# from sklearn.svm import SVC
# from sklearn.pipeline import Pipeline
# from sklearn.metrics import accuracy_score

# # Load a sample dataset (Iris dataset in this example)
# data = load_iris()
# X, y = data.data, data.target

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Create a common feature scaling pipeline and fit it once
# common_preprocessing = Pipeline([
#     ('scaler', StandardScaler())
# ])

# # Fit the common preprocessing pipeline on the training data
# common_preprocessing.fit(X_train)

# # Define pipelines with different feature reduction techniques using the fitted common preprocessing
# # You can add more as needed
# pca_pipeline = Pipeline([
#     ('preprocessing', common_preprocessing),
#     ('pca', PCA(n_components=2)),  # Adjust the number of components as needed
#     ('clf', SVC()),
# ])

# tsne_pipeline = Pipeline([
#     ('preprocessing', common_preprocessing),
#     ('tsne', TSNE(n_components=2)),  # Adjust the number of components as needed
#     ('clf', SVC()),
# ])

# # Create a list of feature reduction pipelines for easy iteration
# feature_reduction_pipelines = [
#     ("PCA", pca_pipeline),
#     ("t-SNE", tsne_pipeline),
# ]

# # Fit and compare the performance of different feature reduction pipelines
# for name, pipeline in feature_reduction_pipelines:
#     # Make predictions on the testing data using the fitted common preprocessing
#     X_test_preprocessed = pipeline.named_steps['preprocessing'].transform(X_test)
#     y_pred = pipeline.named_steps['clf'].predict(X_test_preprocessed)
    
#     # Calculate and print the accuracy
#     accuracy = accuracy_score(y_test, y_pred)
#     print(f"Accuracy with {name}: {accuracy:.2f}")




