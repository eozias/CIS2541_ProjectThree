# Emma Ozias and Peyton Skwarczynski

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve

warnings.simplefilter(action='ignore', category=FutureWarning)

# Step 1: Data Collection (load csv)
data = pd.read_csv('titanic.csv')

# Step 2: Data Cleaning and Preparation
# Identify missing values and decide how to handle them
print("Missing values in each column:")
print(data.isnull().sum())
data['Age'].fillna(data['Age'].median(), inplace=True)
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)

# Drop irrelevant features (name, ticket, cabin, passengerId)
data.drop(['Name', 'Ticket', 'Cabin', 'PassengerId'], axis=1, inplace=True)

# Encode categorical variables (e.g., Sex, Embarked) using one-hot encoding
data = pd.get_dummies(data, columns=['Sex', 'Embarked'], drop_first=True)
print("columns after encoding:")
print(data.columns)

# Step 3: Exploratory Data Analysis (EDA)
# Pclass
plt.figure(figsize=(10, 6))
sns.countplot(x='Pclass', hue='Survived', data=data)
plt.title('Survival Count by Passenger Class')
plt.xlabel('Passenger Class')
plt.ylabel('Count')
plt.legend(title='Survived', labels=['Did not survive', 'Survived'])
plt.show()

# Sex
plt.figure(figsize=(10, 6))
sns.countplot(x='Sex_male', hue='Survived', data=data)
plt.title('Survival Count by Gender')
plt.xlabel('Gender (False = Female, True = Male)')
plt.ylabel('Count')
plt.legend(title='Survived', labels=['Did not survive', 'Survived'])
plt.show()

# Age
plt.figure(figsize=(10, 6))
sns.histplot(data=data, x='Age', hue='Survived', multiple='stack', bins=30)
plt.title('Survival Count by Age')
plt.xlabel('Age')
plt.ylabel('Count')
plt.xlim(0, 90)
plt.xticks(range(0, 91, 5))
plt.legend(title='Survived', labels=['Did not survive', 'Survived'])
plt.show()

# SibSp
plt.figure(figsize=(10, 6))
sns.countplot(x='SibSp', hue='Survived', data=data)
plt.title('Survival Count by Siblings/Spouses Aboard')
plt.xlabel('Number of Siblings/Spouses Aboard')
plt.ylabel('Count')
plt.legend(title='Survived', labels=['Did not survive', 'Survived'])
plt.show()

# Parch
plt.figure(figsize=(10, 6))
sns.countplot(x='Parch', hue='Survived', data=data)
plt.title('Survival Count by Parents/Children Aboard')
plt.xlabel('Number of Parents/Children Aboard')
plt.ylabel('Count')
plt.legend(title='Survived', labels=['Did not survive', 'Survived'])
plt.show()

# Fare
plt.figure(figsize=(10, 6))
sns.histplot(data=data, x='Fare', hue='Survived', multiple='stack', bins=30)
plt.title('Survival Count by Fare')
plt.xlabel('Fare')
plt.ylabel('Count')
plt.xlim(0, 300)
plt.xticks(range(0, 301, 25))
plt.legend(title='Survived', labels=['Did not survive', 'Survived'])
plt.show()

# Embarked
plt.figure(figsize=(10, 6))
sns.countplot(x='Embarked_Q', hue='Survived', data=data)
plt.title('Survival Count by Port of Embarkation')
plt.xlabel('Port of Embarkation (False = Southampton or Cherbourg, True = Queenstown)')
plt.ylabel('Count')
plt.legend(title='Survived', labels=['Did not survive', 'Survived'])
plt.show()

plt.figure(figsize=(10, 6))
sns.countplot(x='Embarked_S', hue='Survived', data=data)
plt.title('Survival Count by Port of Embarkation')
plt.xlabel('Port of Embarkation (False = Cherbourg or Queenstown , True = Southampton)')
plt.ylabel('Count')
plt.legend(title='Survived', labels=['Did not survive', 'Survived'])
plt.show()

# Correlation Matrix
corr_matrix = data.corr()
plt.title("Correlation Matrix of the Titanic Dataset")
plt.figsize = (60, 40)
sns.heatmap(corr_matrix, annot=False, cmap="coolwarm")
xlabels = data.columns
plt.xticks(ticks=np.arange(len(xlabels)), labels=xlabels, fontsize=8)
ylabels = data.columns
plt.yticks(ticks=np.arange(len(ylabels)), labels=ylabels, fontsize=8)
plt.tight_layout()
plt.show()

# Step 4: Split the data into 60% training, 20% validation, 20% testing
df_train_full, df_test = train_test_split(data, test_size=0.2, random_state=42)
df_train, df_val = train_test_split(df_train_full, test_size=0.25, random_state=42)

# Step 5: Model Training
# Train a baseline Random Forest classifier with Default parameters
print()
print("Training a baseline Random Forest classifier...")
X_train = df_train.drop('Survived', axis=1)
y_train = df_train['Survived']
X_test = df_test.drop('Survived', axis=1)
y_test = df_test['Survived']
X_val = df_val.drop('Survived', axis=1)
y_val = df_val['Survived']
rf = RandomForestClassifier()
rf.fit(X_train, y_train)

y_pred = rf.predict_proba(X_val)[:, 1]
print("ROC-AUC on validation set:")
print(roc_auc_score(y_val, y_pred))
plt.figure(figsize=(5, 5))
fpr, tpr, _ = roc_curve(y_val, y_pred)
plt.plot(fpr, tpr, color='black')
plt.plot([0, 1], [0, 1], color='black', lw=0.7, linestyle='dashed', alpha=0.5)
plt.xlim([-0.02, 1.02])
plt.ylim([-0.02, 1.02])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve for the default model using the validation set')
plt.show()

print()

y_pred = rf.predict_proba(X_test)[:, 1]
print("ROC-AUC on testing set:")
print(roc_auc_score(y_test, y_pred))
plt.figure(figsize=(5, 5))
fpr, tpr, _ = roc_curve(y_test, y_pred)
plt.plot(fpr, tpr, color='black')
plt.plot([0, 1], [0, 1], color='black', lw=0.7, linestyle='dashed', alpha=0.5)
plt.xlim([-0.02, 1.02])
plt.ylim([-0.02, 1.02])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve for the default model using the testing set')

# Step 6: Hyperparameter Testing (Experiment with different values for max_depth, min_samples_leaf, and, n_estimators)
values = []
print()
print("Hyperparameter Testing...")
print()
for m in [4, 5, 6]:
    print('depth: %s' % m)
    for s in [1, 5, 10, 15, 20, 50, 100, 200]:
        rf = RandomForestClassifier(max_depth=m, min_samples_leaf=s)
        rf.fit(X_train, y_train)
        y_pred = rf.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_pred)
        print('%s -> %f' % (s, auc))
        values.append(auc)
print("The best ROC-AUC:")
print(max(values))
print()

aucs = []
for i in range(10, 201, 10):
    rf = RandomForestClassifier(n_estimators=i, random_state=3)
    rf.fit(X_train, y_train)
    y_pred = rf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred)
    print('%s -> %f' % (i, auc))
    aucs.append(auc)
print("Best ROC-AUC:")
print(max(aucs))
print()

# Step 7: Final Model Evaluation
print("Training the final model with the best parameters...")
rf = RandomForestClassifier(n_estimators=130, max_depth=5, min_samples_leaf=1)
rf.fit(X_train, y_train)

y_pred = rf.predict_proba(X_val)[:, 1]
print("ROC-AUC on validation set:")
print(roc_auc_score(y_val, y_pred))
plt.figure(figsize=(5, 5))
fpr, tpr, _ = roc_curve(y_val, y_pred)
plt.plot(fpr, tpr, color='black')
plt.plot([0, 1], [0, 1], color='black', lw=0.7, linestyle='dashed', alpha=0.5)
plt.xlim([-0.02, 1.02])
plt.ylim([-0.02, 1.02])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve for the final model using the validation set')
plt.show()

print()

y_pred = rf.predict_proba(X_test)[:, 1]
print("ROC-AUC on testing set:")
print(roc_auc_score(y_test, y_pred))
plt.figure(figsize=(5, 5))
fpr, tpr, _ = roc_curve(y_test, y_pred)
plt.plot(fpr, tpr, color='black')
plt.plot([0, 1], [0, 1], color='black', lw=0.7, linestyle='dashed', alpha=0.5)
plt.xlim([-0.02, 1.02])
plt.ylim([-0.02, 1.02])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve for the final model using the testing set')
plt.show()

# 7b: Did it improve from the default RF model?
# No, the ROC-AUC is slighlty lower than the default model. We have noticed that the ROC-AUC of the final model is usually 0.01 lower than the default model.

# Step 8: Reporting
# 8a: Summarize the key findings, including model performance metrics
# For model performance metrics, please check the output of the code. The ROC-AUC is printed out for the default model and the final model.
# One interesting finding is that the model performance is not significantly improved by tuning the hyperparameters. The ROC-AUC of the final model is usually 0.01 lower than the default model.
# This was interesting because we were expecting a much larger ROC-AUC for the final model.

# 8b: Visulizations: see above code

# 8c: Challenges and Improvements
# One challenge that we faced was deciding which graphs to use for step 3. We struggled with understanding how to complete EDA
# Another challenge that we faced was deciding whether to use the validation set or the testing set
# Another challenge that we faced was creating a graph for the embarked column
# One improvement that we could make is to test a wider range of parameters to see a higher fluctuation in the ROC-AUC


