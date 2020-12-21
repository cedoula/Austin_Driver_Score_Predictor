# Import our dependencies
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd
import os
import sqlite3

# Open database
conn = sqlite3.connect('practice.db')
print ("Opened database successfully")

# Define a variable by querying database
data = conn.execute("SELECT * FROM ML_MOCK_DATA").fetchall()
# Name the columns
cols = ['Index','Crash Death Count','Crash Severity','Crash Time','Crash Total Injury Count','Crash Year','Day of Week','Highway Number','Highway System','Latitude','Light Condition','Longitude','Surface Condition','Surface Type','Weather Condition','Vehicle Body Style','Vehicle Damage Rating 1 - Severity','Vehicle Damage Rating 2 - Severity','Vehicle Make','Vehicle Model Name','Vehicle Model Year','Citation','Person Age','Person Alcohol Result','Person Drug Test Result','Person Gender','Person Injury Severity','Person Type']
# Convert the queried data into a dataframe
mock_df = pd.DataFrame(data, columns=cols)
# Drop the extra index column
mock_df.drop(columns=['Index'], inplace=True)
print(mock_df.head())

# Preprocess the data

print(mock_df.columns)

print(mock_df['Crash Severity'].value_counts())

print(mock_df['Person Type'].value_counts())

print(mock_df['Vehicle Body Style'].value_counts())

# Select features and output columns
mock_df = mock_df[['Crash Severity', 'Person Age', 'Person Gender', 'Person Type', 'Vehicle Model Year', 'Vehicle Body Style', 'Vehicle Make']]
print(mock_df.head())

print(mock_df.shape)

# Drop rows with Unknown
mock_df = mock_df[mock_df['Crash Severity'] != "99 - UNKNOWN"]
print(mock_df.shape)

mock_df = mock_df[(mock_df['Person Type'] != "2 - PASSENGER/OCCUPANT") & (mock_df['Person Type'] != "4 - PEDESTRIAN") & (mock_df['Person Type'] != "99 - UNKNOWN") & (mock_df['Person Type'] != "98 - OTHER (EXPLAIN IN NARRATIVE)") & (mock_df['Person Type'] != "6 - PASSENGER/OCCUPANT ON MOTORCYCLE TYPE VEHICLE")]
print(mock_df.shape)

print(mock_df['Vehicle Body Style'].value_counts())

# Clean Vehicle Body Style column
mock_df = mock_df[(mock_df['Vehicle Body Style'] != "99 - UNKNOWN") & (mock_df['Vehicle Body Style'] != "No Data") & (mock_df['Vehicle Body Style'] != "98 - OTHER  (EXPLAIN IN NARRATIVE)") & (mock_df['Vehicle Body Style'] != "EV - NEV-NEIGHBORHOOD ELECTRIC VEHICLE") & (mock_df['Vehicle Body Style'] != "FE - FARM EQUIPMENT")]
print(mock_df.shape)

print(mock_df.head())

# Drop Vehicle Make column
mock_df = mock_df.drop(columns=['Vehicle Make'], index=1)
print(mock_df.head())

# Clean values for Person Gender and Person Type columns
mock_df.loc[mock_df['Person Gender'] == "2 - FEMALE", 'Person Gender'] = "FEMALE"

mock_df.loc[mock_df['Person Gender'] == "1 - MALE", 'Person Gender'] = "MALE"

mock_df.loc[mock_df['Person Type'] == "1 - DRIVER", 'Person Type'] = "DRIVER"
mock_df.loc[mock_df['Person Type'] == "5 - DRIVER OF MOTORCYCLE TYPE VEHICLE", 'Person Type'] = "MOTORCYCLE DRIVER"
mock_df.loc[mock_df['Person Type'] == "3 - PEDALCYCLIST", 'Person Type'] = "PEDALCYCLIST"


print(mock_df.head())

# Group Crash Severity in two classes: 0 for none or light injury and 1 for serious injury or fatality
mock_df.loc[(mock_df['Crash Severity'] == "N - NOT INJURED") | (mock_df['Crash Severity'] == "B - NON-INCAPACITATING INJURY") | (mock_df['Crash Severity'] == "C - POSSIBLE INJURY"), 'Crash Severity'] = 0


mock_df.loc[(mock_df['Crash Severity'] == "A - SUSPECTED SERIOUS INJURY") | (mock_df['Crash Severity'] == "K - KILLED"), 'Crash Severity'] = 1


print(mock_df['Crash Severity'].value_counts())


print(mock_df.head())

print(mock_df.dtypes)

mock_df["Crash Severity"] = mock_df["Crash Severity"].astype(int)

print(mock_df['Vehicle Model Year'].value_counts())

print(mock_df.shape)

mock_df = mock_df[mock_df['Vehicle Model Year'] != "No Data"]
print(mock_df.shape)

mock_df = mock_df[mock_df['Person Age'] != "No Data"]
print(mock_df.shape)

mock_df["Person Age"] = mock_df["Person Age"].astype(int)
mock_df["Vehicle Model Year"] = mock_df["Vehicle Model Year"].astype(int)

print(mock_df.dtypes)

print(mock_df.head())

mock_df = mock_df[mock_df['Person Gender'] != "99 - UNKNOWN"]
print(mock_df.shape)

# Generate our categorical variable lists
feat_cat = ["Person Gender", "Person Type", "Vehicle Body Style"]

# Create a OneHotEncoder instance
enc = OneHotEncoder(sparse=False)

# Fit and transform the OneHotEncoder using the categorical variable list
encode_df = pd.DataFrame(enc.fit_transform(mock_df[feat_cat]))

# Add the encoded variable names to the dataframe
encode_df.columns = enc.get_feature_names(feat_cat)
print(encode_df.head())

# Merge one-hot encoded features and drop the originals
mock_df = mock_df.merge(encode_df, left_index=True, right_index=True).drop(columns=feat_cat, axis=1)
print(mock_df.head())

# Split our preprocessed data into our features and target arrays
# y=0 for no or low injury and y=1 for serious injury or fatality
y = mock_df["Crash Severity"]
X = mock_df.drop("Crash Severity", axis=1)

# Split the preprocessed data into a training and testing dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, stratify=y)

# Create a StandardScaler instances
scaler = StandardScaler()

# Fit the StandardScaler
X_scaler = scaler.fit(X_train)

# Scale the data
X_train_scaled = X_scaler.transform(X_train)
X_test_scaled = X_scaler.transform(X_test)


# ### Logistic Regression Classifier

# Define the logistic regression model
log_classifier = LogisticRegression(solver="lbfgs", max_iter=200)

# Train the model
log_classifier.fit(X_train_scaled, y_train)

# Evaluate the model
y_pred = log_classifier.predict(X_test_scaled)
print(f" Logistic regression model accuracy: {accuracy_score(y_test, y_pred):.3f}")

results = pd.DataFrame({"Prediction": y_pred, "Actual": y_test}).reset_index(drop=True)
print(results.head(6))


from sklearn.metrics import confusion_matrix, classification_report

# Calculating the confusion matrix.
cm = confusion_matrix(y_test, y_pred)

# Create a DataFrame from the confusion matrix.
cm_df = pd.DataFrame(
    cm, index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"])

cm_df

print("Classification Report")
print(classification_report(y_test, y_pred))


# ### Oversampling

from collections import Counter

print(Counter(y_train))


# ### RandomOverSampler

# Resample the training data with the RandomOversampler
from imblearn.over_sampling import RandomOverSampler
# Instantiate the model
ros = RandomOverSampler(random_state=1)
# Resample the targets
X_resampled, y_resampled = ros.fit_resample(X_train_scaled, y_train)
print(Counter(y_resampled))

# Train the Logistic Regression model using the resampled data
logreg = LogisticRegression(solver='lbfgs', max_iter=200)

# Fit
logreg.fit(X_resampled, y_resampled)

# Evaluate the model
y_pred = logreg.predict(X_test_scaled)
print(f" Logistic regression model accuracy: {accuracy_score(y_test, y_pred):.3f}")

# Calculating the confusion matrix.
cm = confusion_matrix(y_test, y_pred)

# Create a DataFrame from the confusion matrix.
cm_df = pd.DataFrame(
    cm, index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"])

cm_df

print("Classification Report")
print(classification_report(y_test, y_pred))


# ### SMOTEENN

# Resample the training data with SMOTEENN
from imblearn.combine import SMOTEENN
smoteenn = SMOTEENN(random_state=1)
X_resampled, y_resampled = smoteenn.fit_resample(X_train_scaled, y_train)
print(Counter(y_resampled))


# Train the Logistic Regression model using the resampled data
logreg.fit(X_resampled, y_resampled)
# Calculated the balanced accuracy score
y_pred = logreg.predict(X_test_scaled)
from sklearn.metrics import balanced_accuracy_score
print(accuracy_score(y_test, y_pred))

# Calculating the confusion matrix.
cm = confusion_matrix(y_test, y_pred)

# Create a DataFrame from the confusion matrix.
cm_df = pd.DataFrame(
    cm, index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"])

cm_df


print("Classification Report")
print(classification_report(y_test, y_pred))