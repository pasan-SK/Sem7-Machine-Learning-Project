# %%
import pandas as pd
import numpy as np          # For mathematical calculations
import matplotlib.pyplot as plt  # For plotting graphs
# import seaborn as sns
# %matplotlib inline

train_csv_file_path = "D:\\ACA semester 7\\CS4622 - Machine Learning\\ML-Project\\Data\\layer12\\train.csv"
valid_csv_file_path = "D:\\ACA semester 7\\CS4622 - Machine Learning\\ML-Project\\Data\\layer12\\valid.csv"
test_csv_file_path = "D:\\ACA semester 7\\CS4622 - Machine Learning\\ML-Project\\Data\\layer12\\test.csv"

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
print("mean (rounded):", speaker_age_mean)

# %%
# Let's now fill the missing values with the mean value.
train_df.speaker_age.fillna(speaker_age_mean, inplace=True)

# %%
train_df.speaker_age.isna().sum().sum()

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
print("mean: ", speaker_age_mean)
# round it to nearest int
speaker_age_mean = round(speaker_age_mean)
print("mean (rounded): ", speaker_age_mean)

# %%
# Let's now fill the missing values with the mean value.
valid_df.speaker_age.fillna(speaker_age_mean, inplace=True)

# %%
valid_df.speaker_age.value_counts()

# %%
valid_df.speaker_age.isna().sum().sum()

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


# %%
X = train_df.drop(['speaker_ID', 'speaker_age', 'speaker_gender', 'speaker_accent'], axis=1) 
y = train_df[['speaker_ID', 'speaker_age', 'speaker_gender', 'speaker_accent']] 

# %%
splitted_train_X, splitted_test_X, splitted_train_Y, splitted_test_Y = train_test_split(X, y, test_size=0.2, random_state=42)


# %%
splitted_train_speaker_IDs = splitted_train_Y['speaker_ID']
splitted_train_speaker_ages = splitted_train_Y['speaker_age']
splitted_train_speaker_genders = splitted_train_Y['speaker_gender']
splitted_train_speaker_accents = splitted_train_Y['speaker_accent']

splitted_test_speaker_IDs = splitted_test_Y['speaker_ID']
splitted_test_speaker_ages = splitted_test_Y['speaker_age']
splitted_test_speaker_genders = splitted_test_Y['speaker_gender']
splitted_test_speaker_accents = splitted_test_Y['speaker_accent']

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
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFECV
from sklearn.svm import LinearSVC
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier

# %% [markdown]
# ## For Speaker_IDs

# %%
# #  StandardScaler => For ensuring that all features have the same scale, which is often crucial for the proper functioning of many machine learning algorithms.
# #  PCA => For dimensionality reduction
# #  SVC => Support Vector Classifier

# # Pipeline for speaker ID prediction without feature engineering (to check raw accuracy)
# speaker_ID_pipe_svc = Pipeline([ 
#     ('clf', SVC())
#     ])


# # Pipleline for speaker ID prediction with PCA for feature reduction
# speaker_ID_pipe_scaler_pca_svc = Pipeline([
#     ('scaler', StandardScaler()),
#     ('pca', PCA(n_components=0.95)),
#     ('clf', SVC())
#     ])

# # Pipleline for speaker ID prediction with Model-based feature reduction
# speaker_ID_pipe_scaler_sfmlr_svc = Pipeline([
#     ('scaler', StandardScaler()),
#     ('SFM_LR', SelectFromModel(LogisticRegression(C=0.01, penalty='l1', solver='liblinear', class_weight='balanced'))),
#     ('clf', SVC())  
# ])

# %%
p1 = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression())
])

p2 = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', SVC())
])

p3 = Pipeline([
    ('classifier', RandomForestClassifier())
])

p5 = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', KNeighborsClassifier())
])

# %%
##visualize Pipeline
from sklearn import set_config
set_config(display='diagram')

# %%
train_X.shape, train_speaker_IDs.shape, train_speaker_genders.shape, train_speaker_accents.shape

# %% [markdown]
# ### Training

# %%
print("Logistic Regression:")
p1.fit(train_X, train_speaker_IDs)
print(p1.score(valid_X, valid_speaker_IDs))

print("SVC:")
p2.fit(train_X, train_speaker_IDs)
print(p2.score(valid_X, valid_speaker_IDs))

print("RandomForestClassifier:")
p3.fit(train_X, train_speaker_IDs)
print(p3.score(valid_X, valid_speaker_IDs))

print("KNeighborsClassifier:")
p5.fit(train_X, train_speaker_IDs)
print(p5.score(valid_X, valid_speaker_IDs))

# %%
## FROM ABOVE EXPERIMENTES p5 (KNeighborsClassifier is better)

# %%
## let's check without feature scaling
speaker_ID_pipe_knn = Pipeline([
    # ('scaler', StandardScaler()),
    ('classifier', KNeighborsClassifier())
])

speaker_ID_pipe_knn.fit(train_X, train_speaker_IDs)
print(speaker_ID_pipe_knn.score(valid_X, valid_speaker_IDs))

# %%
# Accuracy improved ==> Therefore scaling will not be kept 
# ==> Best acc right now: 0.248 (speaker_ID_pipe_knn)
# let's now try some feature eng techniques 

# %%
speaker_ID_pipe_pca_knn = Pipeline([
    ("pca", PCA(n_components=0.95)),
    ('classifier', KNeighborsClassifier())
])

speaker_ID_pipe_pca_knn.fit(train_X, train_speaker_IDs)
print(speaker_ID_pipe_pca_knn.score(valid_X, valid_speaker_IDs))

# Conclusion after running 
# => acc dropped 
# ==> Best acc right now: 0.248 (speaker_ID_pipe_knn)

# %%
speaker_ID_pipe_scaler_pca_knn = Pipeline([
    ('scaler', StandardScaler()),
    ("pca", PCA(n_components=0.95)),
    ('classifier', KNeighborsClassifier())
])

speaker_ID_pipe_scaler_pca_knn.fit(train_X, train_speaker_IDs)
print(speaker_ID_pipe_scaler_pca_knn.score(valid_X, valid_speaker_IDs))

# Conclusion after running 
# => acc dropped 
# ==> Best acc right now: 0.248 (speaker_ID_pipe_knn)

# %%
speaker_ID_pipe_scaler_sfm_lr = Pipeline([
    ('scaler', StandardScaler()),
    ('SFM_LR', SelectFromModel(LogisticRegression(C=0.01, penalty='l1', solver='liblinear', class_weight='balanced'))),
    ('classifier', KNeighborsClassifier())
])

speaker_ID_pipe_scaler_sfm_lr.fit(train_X, train_speaker_IDs)
print(speaker_ID_pipe_scaler_sfm_lr.score(valid_X, valid_speaker_IDs))

# Conclusion after running 
# => acc dropped
# ==> Best acc right now: 0.248 (speaker_ID_pipe_knn)

# %%
speaker_ID_sfmlr_knn = Pipeline([
    # ('scaler', StandardScaler()),
    ('SFM_LR', SelectFromModel(LogisticRegression(C=0.01, penalty='l1', solver='liblinear', class_weight='balanced'))),
    ('classifier', KNeighborsClassifier())
])

speaker_ID_sfmlr_knn.fit(train_X, train_speaker_IDs)
print(speaker_ID_sfmlr_knn.score(valid_X, valid_speaker_IDs))

# Conclusion after running 
# => acc dropped
# ==> Best acc right now: 0.248 (speaker_ID_pipe_knn)

# %% [markdown]
# ### Hyperparameter tuning

# %%
param_grid = {
    'classifier__n_neighbors': np.arange(1, 21),  # Number of neighbors to consider
    'classifier__weights': ['uniform', 'distance'],  # Weighting method
    'classifier__p': [1, 2]  # Minkowski distance parameter (1 for Manhattan, 2 for Euclidean)
}

random_search = RandomizedSearchCV(estimator=p5, param_distributions=param_grid, n_iter=20, scoring='accuracy', cv=3, n_jobs=-1, verbose=3)

# %%
random_search.fit(train_X, train_speaker_IDs)

# %%
print("Best Hyperparameters: ", random_search.best_params_)

# %%
best_classifier__weights = 'distance' 
best_classifier__p = 2 
best_classifier__n_neighbors = 7

best_speaker_ID_knn = Pipeline([
    ('classifier', KNeighborsClassifier(weights=best_classifier__weights, p=best_classifier__p, n_neighbors=best_classifier__n_neighbors))
])

best_speaker_ID_knn.fit(train_X, train_speaker_IDs)
print(best_speaker_ID_knn.score(valid_X, valid_speaker_IDs))

# Acc increased
# ==> Best acc right now: 0.25066666666666665 (best_speaker_ID_knn)

# %% [markdown]
# ### Prediction on test data 

# %%
# Let's use best performing pipeline to make predictions for the test data
pred_test = best_speaker_ID_knn.predict(test_X)
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

p1 = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(class_weight="balanced"))
])

p2 = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', SVC(class_weight="balanced", kernel='linear'))
])

p3 = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', SVC(class_weight="balanced", kernel='rbf'))
])

p4 = Pipeline([
    ('classifier', RandomForestClassifier(class_weight="balanced"))
])

p5 = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', KNeighborsClassifier())
])

# %%
##visualize Pipeline
from sklearn import set_config
set_config(display='diagram')

# %% [markdown]
# ### Training

# %%
train_speaker_ages.value_counts().plot.bar()

# Let's use class_weight="balanced" to tackle the class imbalance issue 
# Also let's use weighted_avg metric (and F1) since it is a reliable metric when 
# class imbalance issue is there

# %%
print("Classifier = LogisticRegression")
p1.fit(train_X, train_speaker_ages)
print(classification_report(valid_speaker_ages, p1.predict(valid_X)))

print("Classifier = SVC (linear)")
p2.fit(train_X, train_speaker_ages)
print(classification_report(valid_speaker_ages, p2.predict(valid_X)))

print("Classifier = SVC (rbf)")
p3.fit(train_X, train_speaker_ages)
print(classification_report(valid_speaker_ages, p3.predict(valid_X)))

print("Classifier = RandomForestClassifier")
p4.fit(train_X, train_speaker_ages)
print(classification_report(valid_speaker_ages, p4.predict(valid_X)))

print("Classifier = KNeighborsClassifier")
p5.fit(train_X, train_speaker_ages)
print(classification_report(valid_speaker_ages, p5.predict(valid_X)))

# Conslusion after running:
# best algo gives weighted avg = 0.24 (knn)

# %%
# now let's check if we do not do feature scaling would
# it improve the performance

speaker_age_pipe_knn = Pipeline([
    # ('scaler', StandardScaler()),
    ('classifier', KNeighborsClassifier())
])
speaker_age_pipe_knn.fit(train_X, train_speaker_ages)
print(classification_report(valid_speaker_ages, speaker_age_pipe_knn.predict(valid_X)))

# Conslusion after running 
# Weighted avg IMPROVED ==> Therefore let's use without scaling
# Current best weighte avg = 0.30 (speaker_age_pipe_knn)

# %% [markdown]
# #### Hyperparameter tuning

# %%
# Based on the above outputs KNeighborsClassifier() performs better
# Now let's check what hyperparameters performs better

param_dist = {
    'classifier__n_neighbors': np.arange(1, 21),  
    'classifier__weights': ['uniform', 'distance'],  
    'classifier__p': [1, 2]  # Options for the Minkowski distance metric (1 for Manhattan, 2 for Euclidean)
}

# Perform randomized search with cross-validation
random_search = RandomizedSearchCV(speaker_age_pipe_knn, param_distributions=param_dist, n_iter=50, cv=3, scoring='balanced_accuracy', random_state=42, verbose=3)

# %%
random_search.fit(train_X, train_speaker_ages)

# %%
best_params = random_search.best_params_
print("Best Hyperparameters:")
for param, value in best_params.items():
    print(f"{param}: {value}")

# Output
# Best Hyperparameters:
# classifier__weights: uniform
# classifier__p: 1
# classifier__n_neighbors: 1

# %%
# Let's run a pipeline with the best hyperparas as well.
best_classifier__weights = "uniform"
best_classifier__p = 1
best_classifier__n_neighbors = 1

best_speaker_age_knn = Pipeline([
    ('classifier', KNeighborsClassifier(p=best_classifier__p, n_neighbors=best_classifier__n_neighbors, weights=best_classifier__weights))
])

best_speaker_age_knn.fit(train_X, train_speaker_ages)
print(classification_report(valid_speaker_ages, best_speaker_age_knn.predict(valid_X)))

# Conclusion after runing:
# weighted avg DROPPED
# Current best weighte avg = 0.30 (speaker_age_knn)

# %%
# Now let's check what feature engineering technique would be better

speaker_age_pipe_pca_knn = Pipeline([
    ('pca', PCA(n_components=0.95)),
    ('classifier', KNeighborsClassifier(weights="uniform", p=1, n_neighbors=1))
    ])

print("just pca:")
speaker_age_pipe_pca_knn.fit(train_X, train_speaker_ages)
print(classification_report(valid_speaker_ages, speaker_age_pipe_pca_knn.predict(valid_X)))

print("scaling + pca:")
speaker_age_pipe_scaler_pca_knn = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=0.95)),
    ('classifier', KNeighborsClassifier(weights="uniform", p=1, n_neighbors=1))
    ])

speaker_age_pipe_scaler_pca_knn.fit(train_X, train_speaker_ages)
print(classification_report(valid_speaker_ages, speaker_age_pipe_scaler_pca_knn.predict(valid_X)))

# Conclusion: Both DROPPED the weighted avg score

# %%
# Pipleline for speaker age prediction with Model-based feature reduction
speaker_age_pipe_sfmlr_knn = Pipeline([
    ('SFM_LR', SelectFromModel(LogisticRegression(C=0.01, penalty='l1', solver='liblinear', class_weight='balanced'))),
    ('classifier', KNeighborsClassifier(weights="uniform", p=1, n_neighbors=1))
])

print("just model-based feature reduction:")
speaker_age_pipe_sfmlr_knn.fit(train_X, train_speaker_ages)
print(classification_report(valid_speaker_ages, speaker_age_pipe_sfmlr_knn.predict(valid_X)))

speaker_age_pipe_scaler_sfmlr_knn = Pipeline([
    ('scaler', StandardScaler()),
    ('SFM_LR', SelectFromModel(LogisticRegression(C=0.01, penalty='l1', solver='liblinear', class_weight='balanced'))),
    ('classifier', KNeighborsClassifier(weights="uniform", p=1, n_neighbors=1))
])

print("scaling + model-based feature reduction:")
speaker_age_pipe_scaler_sfmlr_knn.fit(train_X, train_speaker_ages)
print(classification_report(valid_speaker_ages, speaker_age_pipe_scaler_sfmlr_knn.predict(valid_X)))

# Conslusion
# Both did not improve the weighted avg score 

# %%
# Just for the checking purpose let's check the same pipelines but with SVC classifier as well (It should give lower weighted_avg than KNN classifier counterparts)

speaker_age_pipe_scaler_pca_svc = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=0.95)),
    ('classifier', SVC(class_weight='balanced'))
    ])

print("scaling + PCA + SVC")
speaker_age_pipe_scaler_pca_svc.fit(train_X, train_speaker_ages)
print(classification_report(valid_speaker_ages, speaker_age_pipe_scaler_pca_svc.predict(valid_X)))

speaker_age_pipe_scaler_sfmlr_svc = Pipeline([
    ('scaler', StandardScaler()),
    ('SFM_LR', SelectFromModel(LogisticRegression(C=0.01, penalty='l1', solver='liblinear', class_weight='balanced'))),
    ('classifier', SVC(class_weight='balanced'))
])

print("scaling + model-based feature reduction + SVC")
speaker_age_pipe_scaler_sfmlr_svc.fit(train_X, train_speaker_ages)
print(classification_report(valid_speaker_ages, speaker_age_pipe_scaler_sfmlr_svc.predict(valid_X)))

# Conclusion
# Both did not imrpve weighted avg as expected

# %% [markdown]
# ### Prediction on test data

# %%
test_X.head()

# %%
# Let's use best performing pipeline to make predictions for the test data
pred_speaker_ages_test = speaker_age_pipe_knn.predict(test_X)
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

# %% [markdown]
# ### Training

# %%
#  Note: 
# There is a class imbalance issue in speaker_gender values. As a solution we can use class_weight='balanced' parameter.
# And will  use weighted avg metric when taking desicions 

#Let's choose a good classifer

speaker_gender_scaler_lr = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(class_weight="balanced"))
])

speaker_gender_scaler_svc_linear = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', SVC(class_weight="balanced", kernel='linear'))
])

speaker_gender_scaler_svc_rbf = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', SVC(class_weight="balanced", kernel='rbf'))
])

speaker_gender_rfc = Pipeline([
    ('classifier', RandomForestClassifier(class_weight="balanced"))
])

speaker_gender_scaler_knn = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', KNeighborsClassifier())
])

# %%
print("Classifier = LogisticRegression")
speaker_gender_scaler_lr.fit(train_X, train_speaker_genders)
print(classification_report(valid_speaker_genders, speaker_gender_scaler_lr.predict(valid_X)))

print("Classifier = SVC (linear)")
speaker_gender_scaler_svc_linear.fit(train_X, train_speaker_genders)
print(classification_report(valid_speaker_genders, speaker_gender_scaler_svc_linear.predict(valid_X)))

print("Classifier = SVC (rbf)")
speaker_gender_scaler_svc_rbf.fit(train_X, train_speaker_genders)
print(classification_report(valid_speaker_genders, speaker_gender_scaler_svc_rbf.predict(valid_X)))

print("Classifier = RandomForestClassifier")
speaker_gender_rfc.fit(train_X, train_speaker_genders)
print(classification_report(valid_speaker_genders, speaker_gender_rfc.predict(valid_X)))

print("Classifier = KNeighborsClassifier")
speaker_gender_scaler_knn.fit(train_X, train_speaker_genders)
print(classification_report(valid_speaker_genders, speaker_gender_scaler_knn.predict(valid_X)))

# Conslusion after running:
# weighted avg of Logistic Regression (pipename=speaker_gender_scaler_lr)  was the highest (0.82)

# %%
# now let's check if we do not do feature scaling would
# it improve the performance

speaker_gender_pipe_lr = Pipeline([
    # ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(class_weight="balanced"))
])
speaker_gender_pipe_lr.fit(train_X, train_speaker_genders)
print(classification_report(valid_speaker_genders, speaker_gender_pipe_lr.predict(valid_X)))

# Conslusion after running 
# Weighted avg IMPROVED to 0.88 ==> therefore let's not keep the feature scaling

# %% [markdown]
# #### Hyperparameter tuning

# %%
param_dist = {
    'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100],  # Regularization parameter
    'classifier__penalty': ['l1', 'l2'],  # Regularization type (L1 or L2)
    'classifier__solver': ['liblinear', 'saga'],  # Solver for logistic regression
}

random_search = RandomizedSearchCV(
    estimator=speaker_gender_pipe_lr,  
    param_distributions=param_dist,  
    n_iter=30, 
    scoring='balanced_accuracy',  
    cv=3,  
    n_jobs=-1,
    verbose=3  
)

# %%

random_search.fit(train_X, train_speaker_genders)

# %%
print("Best Hyperparameters:")
print(random_search.best_params_)

# %%
best_classifier__solver = 'liblinear'
best_classifier__penalty = 'l2'
best_classifier__C = 10

best_speaker_gender_pipe_lr = Pipeline([
    ('classifier', LogisticRegression(class_weight="balanced", solver=best_classifier__solver, penalty=best_classifier__penalty, C=best_classifier__C))
])

best_speaker_gender_pipe_lr.fit(train_X, train_speaker_genders)
print(classification_report(valid_speaker_genders, best_speaker_gender_pipe_lr.predict(valid_X)))

# Weighted avg DROPPED
# Current best Weighted avg is 0.88 (speaker_gender_pipe_lr)

# %%
# Now let's check what feature engineering technique would be better

speaker_gender_pipe_pca_lr = Pipeline([
    ('pca', PCA(n_components=0.95)),
    ('classifier', LogisticRegression(class_weight="balanced", solver=best_classifier__solver, penalty=best_classifier__penalty, C=best_classifier__C))
    ])

print("just pca:")
speaker_gender_pipe_pca_lr.fit(train_X, train_speaker_genders)
print(classification_report(valid_speaker_genders, speaker_gender_pipe_pca_lr.predict(valid_X)))

print("scaling + pca:")
speaker_gender_pipe_scaler_pca_lr = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=0.95)),
    ('classifier', LogisticRegression(class_weight="balanced", solver=best_classifier__solver, penalty=best_classifier__penalty, C=best_classifier__C))
    ])

speaker_gender_pipe_scaler_pca_lr.fit(train_X, train_speaker_genders)
print(classification_report(valid_speaker_genders, speaker_gender_pipe_scaler_pca_lr.predict(valid_X)))

# Conclusion: jsut PCA INCREASED to 0.89 

# %%
# Pipleline for speaker age prediction with Model-based feature reduction
speaker_gender_pipe_sfmlr_knn = Pipeline([
    ('SFM_LR', SelectFromModel(LogisticRegression(C=0.01, penalty='l1', solver='liblinear', class_weight='balanced'))),
    ('classifier', KNeighborsClassifier(weights="uniform", p=1, n_neighbors=1))
])

print("just model-based feature reduction:")
speaker_gender_pipe_sfmlr_knn.fit(train_X, train_speaker_genders)
print(classification_report(valid_speaker_genders, speaker_gender_pipe_sfmlr_knn.predict(valid_X)))

speaker_gender_pipe_scaler_sfmlr_knn = Pipeline([
    ('scaler', StandardScaler()),
    ('SFM_LR', SelectFromModel(LogisticRegression(C=0.01, penalty='l1', solver='liblinear', class_weight='balanced'))),
    ('classifier', KNeighborsClassifier(weights="uniform", p=1, n_neighbors=1))
])

print("scaling + model-based feature reduction:")
speaker_gender_pipe_scaler_sfmlr_knn.fit(train_X, train_speaker_genders)
print(classification_report(valid_speaker_genders, speaker_gender_pipe_scaler_sfmlr_knn.predict(valid_X)))

# Conslusion
# Both did not improve the weighted avg score 

# %% [markdown]
# #### Hyperparameter tuning

# %%
# Current best : speaker_gender_pipe_pca_lr (weighted avg = 0.89)
param_grid = {
    'pca__n_components': np.arange(0.9, 1, 0.01),  # Variance retained by PCA
}

grid_search = GridSearchCV(estimator=speaker_gender_pipe_pca_lr, param_grid=param_grid, scoring='balanced_accuracy', cv=3, n_jobs=-1, verbose=3)


# %%
grid_search.fit(train_X, train_speaker_genders)

# %%
print("Best Hyperparameters: ", grid_search.best_params_)

# %%
best_pca__n_components = 0.945

best_speaker_gender_pipe_pca_lr = Pipeline([
    ('pca', PCA(n_components=best_pca__n_components)),
    ('classifier', LogisticRegression(class_weight="balanced", solver=best_classifier__solver, penalty=best_classifier__penalty, C=best_classifier__C))
    ])

best_speaker_gender_pipe_pca_lr.fit(train_X, train_speaker_genders)
print(classification_report(valid_speaker_genders, best_speaker_gender_pipe_pca_lr.predict(valid_X)))

# Still the best: speaker_gender_pipe_pca_lr (weighted avg = 0.89)

# %% [markdown]
# ### Prediction on test data

# %%
best_speaker_gender_pipe = speaker_gender_pipe_pca_lr
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

# %% [markdown]
# ### Training

# %%
#  Note: 
# There is a class imbalance issue in speaker_accent values. As a solution we can use class_weight='balanced' parameter.
# And will  use weighted avg metric when taking desicions 

#Let's choose a good classifer

speaker_accent_scaler_lr = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(class_weight="balanced"))
])

speaker_accent_scaler_svc_linear = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', SVC(class_weight="balanced", kernel='linear'))
])

speaker_accent_scaler_svc_rbf = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', SVC(class_weight="balanced", kernel='rbf'))
])

speaker_accent_rfc = Pipeline([
    ('classifier', RandomForestClassifier(class_weight="balanced"))
])

speaker_accent_scaler_knn = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', KNeighborsClassifier())
])

# %%
print("Classifier = LogisticRegression")
speaker_accent_scaler_lr.fit(train_X, train_speaker_accents)
print(classification_report(valid_speaker_accents, speaker_accent_scaler_lr.predict(valid_X)))

print("Classifier = SVC (linear)")
speaker_accent_scaler_svc_linear.fit(train_X, train_speaker_accents)
print(classification_report(valid_speaker_accents, speaker_accent_scaler_svc_linear.predict(valid_X)))

print("Classifier = SVC (rbf)")
speaker_accent_scaler_svc_rbf.fit(train_X, train_speaker_accents)
print(classification_report(valid_speaker_accents, speaker_accent_scaler_svc_rbf.predict(valid_X)))

print("Classifier = RandomForestClassifier")
speaker_accent_rfc.fit(train_X, train_speaker_accents)
print(classification_report(valid_speaker_accents, speaker_accent_rfc.predict(valid_X)))

print("Classifier = KNeighborsClassifier")
speaker_accent_scaler_knn.fit(train_X, train_speaker_accents)
print(classification_report(valid_speaker_accents, speaker_accent_scaler_knn.predict(valid_X)))

# Conslusion after running:
# weighted avg of KNN was the best: 0.62       

# %%
# Let's now check whether the feature sclaing helps the weighted svg score or not

speaker_accent_knn = Pipeline([
    # ('scaler', StandardScaler()),
    ('classifier', KNeighborsClassifier())
])

print("Classifier = SVC (linear) without feature scaling")
speaker_accent_knn.fit(train_X, train_speaker_accents)
print(classification_report(valid_speaker_accents, speaker_accent_knn.predict(valid_X)))

# Conclusion
# weighted avg IMPROVED : 0.64 (from 0.62)

# %% [markdown]
# #### Hyperparameter tuning

# %%
param_dist = {
    'classifier__n_neighbors': range(1, 21),  # Number of neighbors to consider
    'classifier__weights': ['uniform', 'distance'],  # Weighting method
    'classifier__p': [1, 2],  # Minkowski distance parameter (1 for Manhattan, 2 for Euclidean)
}

random_search = RandomizedSearchCV(
    estimator=speaker_accent_knn,  # Your pipeline
    param_distributions=param_dist,  # The hyperparameters to search over
    n_iter=10,  # Number of parameter settings that are sampled
    scoring='balanced_accuracy',  # You can use other scoring metrics if needed
    cv=3,  # Number of cross-validation folds
    n_jobs=-1,  # Use all available CPU cores
    verbose=3
)

# %%
random_search.fit(train_X, train_speaker_accents)

# %%
print("Best Hyperparameters:")
print(random_search.best_params_)

# %%
# the best weighted avg as of now: 0.64 (speaker_accent_knn)

best_classifier__weights = 'uniform'
best_classifier__p = 3
best_classifier__n_neighbors = 8

best_speaker_accent_knn = Pipeline([
    ('classifier', KNeighborsClassifier(weights=best_classifier__weights, p=best_classifier__p, n_neighbors=best_classifier__n_neighbors))
])

best_speaker_accent_knn.fit(train_X, train_speaker_accents)
print(classification_report(valid_speaker_accents, best_speaker_accent_knn.predict(valid_X)))

# %%
# Now let's check what feature engineering technique would be better

speaker_accent_pipe_pca_knn = Pipeline([
    ('pca', PCA(n_components=0.95)),
    ('classifier', KNeighborsClassifier(p=best_classifier__p, n_neighbors=best_classifier__n_neighbors, weights=best_classifier__weights))
    ])

print("just pca:")
speaker_accent_pipe_pca_knn.fit(train_X, train_speaker_accents)
print(classification_report(valid_speaker_accents, speaker_accent_pipe_pca_knn.predict(valid_X)))

print("scaling + pca:")
speaker_accent_pipe_scaler_pca_knn = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=0.95)),
    ('classifier', KNeighborsClassifier(p=best_classifier__p, n_neighbors=best_classifier__n_neighbors, weights=best_classifier__weights))
    ])

speaker_accent_pipe_scaler_pca_knn.fit(train_X, train_speaker_accents)
print(classification_report(valid_speaker_accents, speaker_accent_pipe_scaler_pca_knn.predict(valid_X)))

# Conclusion: jsut PCA gave the same weighted avg ==> Let's try to hyperparameter tune it

# %%
# Pipleline for speaker age prediction with Model-based feature reduction
speaker_accent_pipe_sfmlr_knn = Pipeline([
    ('SFM_LR', SelectFromModel(LogisticRegression(C=0.01, penalty='l1', solver='liblinear', class_weight='balanced'))),
    ('classifier', KNeighborsClassifier(p=best_classifier__p, n_neighbors=best_classifier__n_neighbors, weights=best_classifier__weights))
])

print("just model-based feature reduction:")
speaker_accent_pipe_sfmlr_knn.fit(train_X, train_speaker_accents)
print(classification_report(valid_speaker_accents, speaker_accent_pipe_sfmlr_knn.predict(valid_X)))

speaker_accent_pipe_scaler_sfmlr_knn = Pipeline([
    ('scaler', StandardScaler()),
    ('SFM_LR', SelectFromModel(LogisticRegression(C=0.01, penalty='l1', solver='liblinear', class_weight='balanced'))),
    ('classifier', KNeighborsClassifier(p=best_classifier__p, n_neighbors=best_classifier__n_neighbors, weights=best_classifier__weights))
])

print("scaling + model-based feature reduction:")
speaker_accent_pipe_scaler_sfmlr_knn.fit(train_X, train_speaker_accents)
print(classification_report(valid_speaker_accents, speaker_accent_pipe_scaler_sfmlr_knn.predict(valid_X)))

# Conslusion
# Both did not improve the weighted avg score 

# %% [markdown]
# ### Hyperparameter tuning

# %%
# current best weighted avg: 0.65 (best_speaker_accent_knn)
best_speaker_accent_pipe_pca_knn = Pipeline([
    ('pca', PCA(n_components=0.99)),
    ('classifier', KNeighborsClassifier(p=best_classifier__p, n_neighbors=best_classifier__n_neighbors, weights=best_classifier__weights))
    ])

best_speaker_accent_pipe_pca_knn.fit(train_X, train_speaker_accents)
print(classification_report(valid_speaker_accents, best_speaker_accent_pipe_pca_knn.predict(valid_X)))


# %% [markdown]
# ### Prediction test data

# %%
test_X.head()

# %%
pred_speaker_accents_test = best_speaker_accent_pipe_pca_knn.predict(test_X)
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
pred_test.head()

# %%
pred_test_layer7 = pred_test
pred_test_layer7.head() 

# %%
pred_test_layer7.to_csv('3 - 190290U_pred_test_layer12.csv', index=False)

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




