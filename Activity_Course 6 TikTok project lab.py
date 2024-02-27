#!/usr/bin/env python
# coding: utf-8

# # **TikTok Project**
# **Course 6 - The Nuts and bolts of machine learning**

# Recall that you are a data professional at TikTok. Your supervisor was impressed with the work you have done and has requested that you build a machine learning model that can be used to determine whether a video contains a claim or whether it offers an opinion. With a successful prediction model, TikTok can reduce the backlog of user reports and prioritize them more efficiently.
# 
# A notebook was structured and prepared to help you in this project. A notebook was structured and prepared to help you in this project. Please complete the following questions.

# # **Course 6 End-of-course project: Classifying videos using machine learning**
# 
# In this activity, you will practice using machine learning techniques to predict on a binary outcome variable.
# <br/>
# 
# **The purpose** of this model is to increase response time and system efficiency by automating the initial stages of the claims process.
# 
# **The goal** of this model is to predict whether a TikTok video presents a "claim" or presents an "opinion".
# <br/>
# 
# *This activity has three parts:*
# 
# **Part 1:** Ethical considerations
# * Consider the ethical implications of the request
# 
# * Should the objective of the model be adjusted?
# 
# **Part 2:** Feature engineering
# 
# * Perform feature selection, extraction, and transformation to prepare the data for modeling
# 
# **Part 3:** Modeling
# 
# * Build the models, evaluate them, and advise on next steps
# 
# Follow the instructions and answer the questions below to complete the activity. Then, you will complete an Executive Summary using the questions listed on the PACE Strategy Document.
# 
# Be sure to complete this activity before moving on. The next course item will provide you with a completed exemplar to compare to your own work.
# 
# 

# # **Classify videos using machine learning**

# <img src="images/Pace.png" width="100" height="100" align=left>
# 
# # **PACE stages**
# 

# Throughout these project notebooks, you'll see references to the problem-solving framework PACE. The following notebook components are labeled with the respective PACE stage: Plan, Analyze, Construct, and Execute.

# <img src="images/Plan.png" width="100" height="100" align=left>
# 
# 
# ## **PACE: Plan**
# 
# Consider the questions in your PACE Strategy Document to reflect on the Plan stage.
# 
# In this stage, consider the following questions:
# 
# 
# 1.   **What are you being asked to do? What metric should I use to evaluate success of my business/organizational objective?**
# 
# 2.   **What are the ethical implications of the model? What are the consequences of your model making errors?**
#   *   What is the likely effect of the model when it predicts a false negative (i.e., when the model says a video does not contain a claim and it actually does)?
# 
#   *   What is the likely effect of the model when it predicts a false positive (i.e., when the model says a video does contain a claim and it actually does not)?
# 
# 3.   **How would you proceed?**
# 

# ==> ENTER YOUR RESPONSES HERE

# ### **Task 1. Imports and data loading**
# 
# Start by importing packages needed to build machine learning models to achieve the goal of this project.

# In[17]:


# Import packages for data manipulation
import pandas as pd
import numpy as np

# Import packages for data visualization
import matplotlib.pyplot as plt
import seaborn as sns


# Import packages for data preprocessing
from sklearn.feature_extraction.text import CountVectorizer


# Import packages for data modeling
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from xgboost import plot_importance
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay


# Now load the data from the provided csv file into a dataframe.
# 
# **Note:** As shown in this cell, the dataset has been automatically loaded in for you. You do not need to download the .csv file, or provide more code, in order to access the dataset and proceed with this lab. Please continue with this activity by completing the following instructions.

# In[2]:


# Load dataset into dataframe
data = pd.read_csv("tiktok_dataset.csv")


# <img src="images/Analyze.png" width="100" height="100" align=left>
# 
# ## **PACE: Analyze**
# 
# Consider the questions in your PACE Strategy Document to reflect on the Analyze stage.

# ### **Task 2: Examine data, summary info, and descriptive stats**

# Inspect the first five rows of the dataframe.

# In[3]:


# Display first few rows
data.head(10)


# Get the number of rows and columns in the dataset.

# In[4]:


# Get number of rows and columns
data.shape


# Get the data types of the columns.

# In[5]:


# Get data types of columns
data.info()


# Get basic information about the dataset.

# In[6]:


# Get basic information
data.info()


# Generate basic descriptive statistics about the dataset.

# In[7]:


# Generate basic descriptive stats
data.describe()


# Check for and handle missing values.

# In[8]:


# Check for missing values
data.isna().sum()


# In[9]:


# Drop rows with missing values
data = data.dropna(axis = 0)

data.isna().sum()


# In[10]:


# Display first few rows after handling missing values
data.head()


# Check for and handle duplicates.

# In[11]:


# Check for duplicates
data.duplicated().sum()


# Check for and handle outliers.

# In[19]:


### YOUR CODE HERE ###

Accoring to exemplar response:
    
Exemplar response: Tree-based models are robust to outliers, 
so there is no need to impute or drop any values based on where they fall in their distribution.

This makes sense due to the nature of tree model building learned in this course. 
To my understanding Tree-Building involves more of blanket binary decisions thus not implimenting things such as 
complex math on the data being used. Thus outliars are not an issue for this section. 


# Check class balance.

# In[12]:


# Check class balance
data['claim_status'].value_counts(normalize=True)


# <img src="images/Construct.png" width="100" height="100" align=left>
# 
# ## **PACE: Construct**
# Consider the questions in your PACE Strategy Document to reflect on the Construct stage.

# ### **Task 3: Feature engineering**

# Extract the length of each `video_transcription_text` and add this as a column to the dataframe, so that it can be used as a potential feature in the model.

# In[20]:


# Extract the length of each `video_transcription_text` and add this as a column to the dataframe
data["text_length"] = data["video_transcription_text"].str.len()
data.head()


# Calculate the average text_length for claims and opinions.

# In[21]:


# Calculate the average text_length for claims and opinions
data[['claim_status', 'text_length']].groupby('claim_status').mean()


# Visualize the distribution of `text_length` for claims and opinions.

# In[31]:


# Visualize the distribution of `text_length` for claims and opinions
# Create two histograms in one plot
sns.histplot(data=data, stat='count', multiple='dodge', x='text_length', kde=False, palette='pastel',
           hue='claim_status', element='bars', legend=True)
plt.title('Distrobution of video_transcription_text (# of Chars) split by Claim Status')
plt.xlabel('video_transcription_text length (number of characters)')
plt.ylabel('Count')



# **Feature selection and transformation**

# Encode target and catgorical variables.

# In[85]:


# Create a copy of the X data
X = data.copy()
# Drop unnecessary columns
X = X.drop(['#', 'video_id'], axis=1)

# Encode target variable
X['claim_status'] = X['claim_status'].replace({'opinion':0, 'claim':1})

# Dummy encode remaining categorical values
X = pd.get_dummies(X, columns=['verified_status', 'author_ban_status'], drop_first=True)

X.head()


# ### **Task 4: Split the data**

# Assign target variable.

# In[33]:


# Isolate target variable
y = X['claim_status']


# Isolate the features.

# In[34]:


# Isolate features
X = X.drop(['claim_status'], axis=1)

# Display first few rows of features dataframe
X.head()


# #### **Task 5: Create train/validate/test sets**

# Split data into training and testing sets, 80/20.

# In[75]:


# Split the data into training and testing sets
X_tr, X_test, y_tr, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# Split the training set into training and validation sets, 75/25, to result in a final ratio of 60/20/20 for train/validate/test sets.

# In[76]:


# Split the training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_tr, y_tr, test_size=0.25, random_state=0)


# Confirm that the dimensions of the training, validation, and testing sets are in alignment.

# In[77]:


# Get shape of each training, validation, and testing set
X_train.shape, X_val.shape, X_test.shape, y_train.shape, y_val.shape, y_test.shape


# ### **Task 6. Build models**
# 

# ### **Build a random forest model**

# Fit a random forest model to the training set. Use cross-validation to tune the hyperparameters and select the model that performs best on recall.

# In[98]:


# Set up a `CountVectorizer` object, which converts a collection of text to a matrix of token counts
count_vec = CountVectorizer(ngram_range=(2, 3),
                            max_features=15,
                            stop_words='english')
count_vec


# Extract numerical features from `video_transcription_text` in the training set
count_data = count_vec.fit_transform(X_train['video_transcription_text']).toarray()
count_data


# Place the numerical representation of `video_transcription_text` from training set into a dataframe
count_df = pd.DataFrame(data=count_data, columns=count_vec.get_feature_names_out())


# Concatenate `X_train` and `count_df` to form the final dataframe for training data (`X_train_final`)
# Note: Using `.reset_index(drop=True)` to reset the index in X_train after dropping `video_transcription_text`,
# so that the indices align with those in `X_train` and `count_df`
X_train_final = pd.concat([X_train.drop(columns=['video_transcription_text']).reset_index(drop=True), count_df], axis=1)

# Extract numerical features from `video_transcription_text` in the testing set
validation_count_data = count_vec.transform(X_val['video_transcription_text']).toarray()
validation_count_data

# Place the numerical representation of `video_transcription_text` from validation set into a dataframe
validation_count_df = pd.DataFrame(data=validation_count_data, columns=count_vec.get_feature_names_out())

# Concatenate `X_val` and `validation_count_df` to form the final dataframe for training data (`X_val_final`)
# Note: Using `.reset_index(drop=True)` to reset the index in X_val after dropping `video_transcription_text`,
# so that the indices align with those in `validation_count_df`
X_val_final = pd.concat([X_val.drop(columns=['video_transcription_text']).reset_index(drop=True), validation_count_df], axis=1)


# Extract numerical features from `video_transcription_text` in the testing set
test_count_data = count_vec.transform(X_test['video_transcription_text']).toarray()

# Place the numerical representation of `video_transcription_text` from test set into a dataframe
test_count_df = pd.DataFrame(data=test_count_data, columns=count_vec.get_feature_names_out())

# Concatenate `X_val` and `validation_count_df` to form the final dataframe for training data (`X_val_final`)
X_test_final = pd.concat([X_test.drop(columns=['video_transcription_text']
                                      ).reset_index(drop=True), test_count_df], axis=1)




# Instantiate the random forest classifier
rf = RandomForestClassifier(random_state=0)

# Create a dictionary of hyperparameters to tune
cv_params = {'max_depth': [5, 7, None],
             'max_features': [0.3, 0.6],
            #  'max_features': 'auto'
             'max_samples': [0.7],
             'min_samples_leaf': [1,2],
             'min_samples_split': [2,3],
             'n_estimators': [75,100,200],
             }

# Define a dictionary of scoring metrics to capture
scoring = {'accuracy', 'precision', 'recall', 'f1'}

# Instantiate the GridSearchCV object
rf_cv = GridSearchCV(rf, cv_params, scoring=scoring, cv=5, refit='recall')

rf_cv.fit(X_train_final, y_train)


# In[105]:


# Examine best recall score

rf_cv.best_score_


# In[104]:


def make_results(model_name, model_object):
    '''
    Accepts as arguments a model name (your choice - string) and
    a fit GridSearchCV model object.

    Returns a pandas df with the F1, recall, precision, and accuracy scores
    for the model with the best mean F1 score across all validation folds.
    '''

    # Get all the results from the CV and put them in a df
    cv_results = pd.DataFrame(model_object.cv_results_)

    # Isolate the row of the df with the max(mean precision score)
    best_estimator_results = cv_results.iloc[cv_results['mean_test_precision'].idxmax(), :]

    # Extract accuracy, precision, recall, and f1 score from that row
    f1 = best_estimator_results.mean_test_f1
    recall = best_estimator_results.mean_test_recall
    precision = best_estimator_results.mean_test_precision
    accuracy = best_estimator_results.mean_test_accuracy

    # Create table of results
    table = pd.DataFrame({'Model': [model_name],
                          'F1': [f1],
                          'Recall': [recall],
                          'Precision': [precision],
                          'Accuracy': [accuracy]
                         }
                        )

    return table

rf_cv_results = make_results('Random Forest CV', rf_cv)
rf_cv_results


# In[100]:


# Examine best parameters
rf_cv.best_params_


# **Question:** How well is your model performing? Consider average recall score and precision score.

# ### **Build an XGBoost model**

# In[102]:


# Instantiate the XGBoost classifier
xgb = XGBClassifier(objective='binary:logistic', random_state=0)

# Create a dictionary of hyperparameters to tune
cv_params = {'max_depth': [4,8,12],
             'min_child_weight': [3, 5],
             'learning_rate': [0.01, 0.1],
             'n_estimators': [300, 500]
             }

# Define a dictionary of scoring metrics to capture
scoring = {'accuracy', 'precision', 'recall', 'f1'}

# Instantiate the GridSearchCV object
xgb_cv = GridSearchCV(xgb, cv_params, scoring=scoring, cv=5, refit='recall')

xgb_cv.fit(X_train_final, y_train)


# In[106]:


def make_results(model_name, model_object):
    '''
    Accepts as arguments a model name (your choice - string) and
    a fit GridSearchCV model object.

    Returns a pandas df with the F1, recall, precision, and accuracy scores
    for the model with the best mean F1 score across all validation folds.
    '''

    # Get all the results from the CV and put them in a df
    cv_results = pd.DataFrame(model_object.cv_results_)

    # Isolate the row of the df with the max(mean precision score)
    best_estimator_results = cv_results.iloc[cv_results['mean_test_precision'].idxmax(), :]

    # Extract accuracy, precision, recall, and f1 score from that row
    f1 = best_estimator_results.mean_test_f1
    recall = best_estimator_results.mean_test_recall
    precision = best_estimator_results.mean_test_precision
    accuracy = best_estimator_results.mean_test_accuracy

    # Create table of results
    table = pd.DataFrame({'Model': [model_name],
                          'F1': [f1],
                          'Recall': [recall],
                          'Precision': [precision],
                          'Accuracy': [accuracy]
                         }
                        )

    return table

xgb_cv_results = make_results('XGBoost model CV', xgb_cv)
xgb_cv_results


# **Question:** How well does your model perform? Consider recall score and precision score.

# <img src="images/Execute.png" width="100" height="100" align=left>
# 
# ## **PACE: Execute**
# Consider the questions in your PACE Strategy Document to reflect on the Execute stage.

# ### **Task 7. Evaluate model**
# 
# Evaluate models against validation criteria.

# #### **Random forest**

# In[107]:


# Use the random forest "best estimator" model to get predictions on the encoded testing set
y_pred = rf_cv.best_estimator_.predict(X_val_final)


# Display the predictions on the encoded testing set.

# In[108]:


# Display the predictions on the encoded testing set
y_pred


# Display the true labels of the testing set.

# In[109]:


# Display the true labels of the testing set
y_val


# Create a confusion matrix to visualize the results of the classification model.

# In[118]:


# Create a confusion matrix to visualize the results of the classification model

# Compute values for confusion matrix
rf_cm = confusion_matrix(y_val,y_pred)

# Create display of confusion matrix
rf_disp = ConfusionMatrixDisplay(confusion_matrix=rf_cm, display_labels=None)

# Plot confusion matrix
rf_disp.plot()

# Display plot
plt.title('Random Forest - validation set');
plt.show()


# Create a classification report that includes precision, recall, f1-score, and accuracy metrics to evaluate the performance of the model.

# In[111]:


# Create a classification report
# Create classification report for random forest model
target_labels = ['opinion', 'claim']
print(classification_report(y_val, y_pred, target_names=target_labels))


# **Question:** What does your classification report show? What does the confusion matrix indicate?
# 
# The classification report above shows that the random forest model scores were nearly perfect. The confusion matrix indicates that there were 10 misclassifications—five false postives and five false negatives.

# #### **XGBoost**

# In[113]:


#Evaluate XGBoost model
y_pred = xgb_cv.best_estimator_.predict(X_val_final)


# In[116]:


# Compute values for confusion matrix
xgb_cm = confusion_matrix(y_val,y_pred)

# Create display of confusion matrix
xgb_disp = ConfusionMatrixDisplay(confusion_matrix=xgb_cm, display_labels=None)

# Plot confusion matrix
xgb_disp.plot()

# Display plot
plt.title('XGBoost - validation set');
plt.show()


# In[117]:


# Create a classification report
target_labels = ['opinion', 'claim']
print(classification_report(y_val, y_pred, target_names=target_labels))


# **Question:** Describe your XGBoost model results. How does your XGBoost model compare to your random forest model?
# 
# The results of the XGBoost model were also nearly perfect. However, its errors tended to be false negatives. Identifying claims was the priority, so it's important that the model be good at capturing all actual claim videos. The random forest model has a better recall score, and is therefore the champion model.

# ### **Use champion model to predict on test data**

# In[119]:


### YOUR CODE HERE ###
y_pred = rf_cv.best_estimator_.predict(X_test_final)


# In[120]:


# Compute values for confusion matrix
log_cm = confusion_matrix(y_test, y_pred)

# Create display of confusion matrix
log_disp = ConfusionMatrixDisplay(confusion_matrix=log_cm, display_labels=None)

# Plot confusion matrix
log_disp.plot()

# Display plot
plt.title('Random forest - test set');
plt.show()


# #### **Feature importances of champion model**
# 

# In[121]:


importances = rf_cv.best_estimator_.feature_importances_
rf_importances = pd.Series(importances, index=X_test_final.columns)

fig, ax = plt.subplots()
rf_importances.plot.bar(ax=ax)
ax.set_title('Feature importances')
ax.set_ylabel('Mean decrease in impurity')
fig.tight_layout()


# **Question:** Describe your most predictive features. Were your results surprising?
# 
# The most predictive features all were related to engagement levels generated by the video. This is not unexpected, as analysis from prior EDA pointed to this conclusion.

# ### **Task 8. Conclusion**
# 
# In this step use the results of the models above to formulate a conclusion. Consider the following questions:
# 
# 1. **Would you recommend using this model? Why or why not?**
# 
# 2. **What was your model doing? Can you explain how it was making predictions?**
# 
# 3. **Are there new features that you can engineer that might improve model performance?**
# 
# 4. **What features would you want to have that would likely improve the performance of your model?**
# 
# Remember, sometimes your data simply will not be predictive of your chosen target. This is common. Machine learning is a powerful tool, but it is not magic. If your data does not contain predictive signal, even the most complex algorithm will not be able to deliver consistent and accurate predictions. Do not be afraid to draw this conclusion.
# 

# 1) Would you recommend using this model? Why or why not? Yes, one can recommend this model because it performed well on both the validation and test holdout data. Furthermore, both precision and F1 scores were consistently high. The model very successfully classified claims and opinions.
# 
# 2) What was your model doing? Can you explain how it was making predictions? The model's most predictive features were all related to the user engagement levels associated with each video. It was classifying videos based on how many views, likes, shares, and downloads they received.
# 
# 3) Are there new features that you can engineer that might improve model performance? Because the model currently performs nearly perfectly, there is no need to engineer any new features.
# 
# 4) What features would you want to have that would likely improve the performance of your model? The current version of the model does not need any new features. However, it would be helpful to have the number of times the video was reported. It would also be useful to have the total number of user reports for all videos posted by each author.

# **Congratulations!** You've completed this lab. However, you may not notice a green check mark next to this item on Coursera's platform. Please continue your progress regardless of the check mark. Just click on the "save" icon at the top of this notebook to ensure your work has been logged.
