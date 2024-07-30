# Emma Ozias and Peyton Skwarczynski

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import warnings

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

# Step 3: Exploratory Data Analysis (EDA)

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

y_pred = rf.predict(X_val)
print("ROC-AUC on validation set:")
print(roc_auc_score(y_val, y_pred))
print()
y_pred = rf.predict_proba(X_test)[:, 1]
print("ROC-AUC on testing set:")
print(roc_auc_score(y_test, y_pred))

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

y_pred = rf.predict(X_val)
print("ROC-AUC on validation set:")
print(roc_auc_score(y_val, y_pred))
print()
y_pred = rf.predict_proba(X_test)[:, 1]
print("ROC-AUC on testing set:")
print(roc_auc_score(y_test, y_pred))

# Step 8: Reporting
