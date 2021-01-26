#!/usr/bin/env python
# coding: utf-8

# ### This code generates the pipeline (Scaler+ML model) and the Labelencoders for the flask app


#Import our dependencies
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, accuracy_score, roc_curve, roc_auc_score
import pandas as pd
import os
from rds import db_password
from psycopg2 import sql, connect
from matplotlib import pyplot
import matplotlib.pylab as plt
#get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4


try:
    # declare a new PostgreSQL connection object
    conn = connect(
        dbname = "car_db",
        user = "postgres",
        host = "final-project.cn1djdbx7lfi.us-east-1.rds.amazonaws.com",
        port = "5432",
        password = db_password
    )

    # print the connection if successful
    print ("psycopg2 connection:", conn)

except Exception as err:
    print ("psycopg2 connect() ERROR:", err)
    conn = None



cr = conn.cursor()
cr.execute('SELECT * FROM total_crash_data;')
tmp = cr.fetchall()

# Extract the column names
col_names = []
for elt in cr.description:
    col_names.append(elt[0])

# Create the dataframe, passing in the list of col_names extracted from the description
df = pd.DataFrame(tmp, columns=col_names)


# # Preprocessing Data

# I selected these features because I think they will have a high correlation with crash severity
mock_df = df[['Crash Severity', 'Person Age', 'Person Gender', 'Person Type','Vehicle Model Year', 'Vehicle Body Style', 'Vehicle Make', 'Day of Week']]
print(mock_df.shape)

# Drop rows with Unknown
mock_df = mock_df[mock_df['Crash Severity'] != "99 - UNKNOWN"]
mock_df = mock_df[mock_df['Person Age'] != "99 - UNKNOWN"]
mock_df = mock_df[mock_df['Person Gender'] != "99 - UNKNOWN"]
mock_df = mock_df[mock_df['Person Type'] != "99 - UNKNOWN"]
mock_df = mock_df[mock_df['Vehicle Model Year'] != "99 - UNKNOWN"]
mock_df = mock_df[mock_df['Vehicle Body Style'] != "99 - UNKNOWN"]
mock_df = mock_df[mock_df['Vehicle Make'] != "99 - UNKNOWN"]
print(mock_df.shape)

# Clean Person Type Column
mock_df = mock_df[(mock_df['Person Type'] != "2 - PASSENGER/OCCUPANT") & (mock_df['Person Type'] != "4 - PEDESTRIAN") & (mock_df['Person Type'] != "98 - OTHER (EXPLAIN IN NARRATIVE)") & (mock_df['Person Type'] != "6 - PASSENGER/OCCUPANT ON MOTORCYCLE TYPE VEHICLE")]
print(mock_df.shape)

# Clean Vehicle Body Style column
mock_df = mock_df[(mock_df['Vehicle Body Style'] != "No Data") & (mock_df['Vehicle Body Style'] != "98 - OTHER  (EXPLAIN IN NARRATIVE)") & (mock_df['Vehicle Body Style'] != "EV - NEV-NEIGHBORHOOD ELECTRIC VEHICLE") & (mock_df['Vehicle Body Style'] != "FE - FARM EQUIPMENT")]
print(mock_df.shape)

# Clean Vehicle Body Style column
mock_df.loc[(mock_df['Vehicle Body Style'] == "P4 - PASSENGER CAR, 4-DOOR") , 'Vehicle Body Style'] = "SEDAN(4-DOOR)"
mock_df.loc[(mock_df['Vehicle Body Style'] == "SV - SPORT UTILITY VEHICLE"), 'Vehicle Body Style'] = "SUV"
mock_df.loc[(mock_df['Vehicle Body Style'] == "PK - PICKUP"), 'Vehicle Body Style'] = "PICKUP TRUCK"
mock_df.loc[(mock_df['Vehicle Body Style'] == "P2 - PASSENGER CAR, 2-DOOR"), 'Vehicle Body Style'] = "SEDAN(2-DOOR)"
mock_df.loc[(mock_df['Vehicle Body Style'] == "VN - VAN"), "Vehicle Body Style"] = "VAN"
mock_df.loc[(mock_df['Vehicle Body Style'] == "TR - TRUCK"), "Vehicle Body Style"] = "TRUCK(OTHER)"
mock_df.loc[(mock_df['Vehicle Body Style'] == "MC - MOTORCYCLE"), "Vehicle Body Style"] = "MOTORCYCLE"
mock_df.loc[(mock_df['Vehicle Body Style'] == "TT - TRUCK TRACTOR") | (mock_df['Vehicle Body Style'] == "PC - POLICE CAR/TRUCK") | (mock_df['Vehicle Body Style'] == "BU - BUS") | (mock_df['Vehicle Body Style'] == "AM - AMBULANCE") | (mock_df['Vehicle Body Style'] == "FT - FIRE TRUCK") | (mock_df['Vehicle Body Style'] == "SB - YELLOW SCHOOL BUS") | (mock_df['Vehicle Body Style'] == "PM - POLICE MOTORCYCLE"), "Vehicle Body Style"] = "OTHER"

# Clean Person Gender column
mock_df['Person Gender'].value_counts()

# Group Gender into two classes: 0 for Female and 1 for Male
mock_df.loc[mock_df['Person Gender'] == "2 - FEMALE", 'Person Gender'] = "FEMALE"
mock_df.loc[mock_df['Person Gender'] == "1 - MALE", 'Person Gender'] = "MALE"

# Clean values for Person Type
mock_df.loc[mock_df['Person Type'] == "1 - DRIVER", 'Person Type'] = "DRIVER"
mock_df.loc[mock_df['Person Type'] == "5 - DRIVER OF MOTORCYCLE TYPE VEHICLE", 'Person Type'] = "MOTORCYCLE DRIVER"

# Create a function that groups crash severity into two groups, 1 for No injury, 2 for all other injuries.
def group_crash(X):
    if X == "N - NOT INJURED" or X == "C - POSSIBLE INJURY":
        return 0
    else:
        return 1

# Group Crash Severity into two classes in order for the ML to predict the level of severity.
mock_df['Crash Severity'] = mock_df['Crash Severity'].apply(group_crash)

mock_df = mock_df[mock_df['Person Age'] != "No Data"]

mock_df = mock_df[mock_df['Vehicle Model Year'] != "No Data"]

mock_df["Person Age"] = mock_df["Person Age"].astype(int)

mock_df["Vehicle Model Year"] = mock_df["Vehicle Model Year"].astype(int)

# Map over the Vehicle Make column and replace the values with "OTHER" where the value count is less than 900.
mock_df[['Vehicle Make']] = mock_df[['Vehicle Make']].where(mock_df.apply(lambda x: x.map(x.value_counts()))>=600, "OTHER")

#Drop Null values
mock_df = mock_df.dropna()

mock_df['Crash Severity'] = mock_df['Crash Severity'].astype(int)

print(mock_df.shape)


# # Prepare Data for ML Model

# Drop Person Type 
mock_df = mock_df.drop("Person Type", axis=1)
print(mock_df.head())

mock_df_copy = mock_df.copy().drop('Crash Severity', axis=1).reset_index(drop=True)
print(mock_df_copy.head())

user = [33, 'MALE', 2016, 'SUV', 'NISSAN', 'WEDNESDAY']

mock_df_copy.loc[len(mock_df_copy)] = user

print(mock_df_copy.tail())

# Generate the categorical variable list
feat_cat = ['Person Gender', 'Vehicle Body Style', 'Vehicle Make', 'Day of Week']

# Create a LabelEncoder instance
le_gender = LabelEncoder()
le_body = LabelEncoder()
le_make = LabelEncoder()
le_day = LabelEncoder()

# Fit and transform the LabelEncoder using the categorical variable list
mock_df['Person Gender'] = le_gender.fit_transform(mock_df['Person Gender'])
mock_df['Vehicle Body Style'] = le_body.fit_transform(mock_df['Vehicle Body Style'])
mock_df['Vehicle Make'] = le_make.fit_transform(mock_df['Vehicle Make'])
mock_df['Day of Week'] = le_day.fit_transform(mock_df['Day of Week'])

print(mock_df.head())

# Fit and transform the LabelEncoder using the categorical variable list
mock_df_copy['Person Gender'] = le_gender.transform(mock_df_copy['Person Gender'])
mock_df_copy['Vehicle Body Style'] = le_body.transform(mock_df_copy['Vehicle Body Style'])
mock_df_copy['Vehicle Make'] = le_make.transform(mock_df_copy['Vehicle Make'])
mock_df_copy['Day of Week'] = le_day.transform(mock_df_copy['Day of Week'])

print(mock_df_copy.head())

print(mock_df_copy.tail())

# Split our preprocessed data into our features and target arrays
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

mock_df_copy_scaled = X_scaler.transform(mock_df_copy)

user_info = mock_df_copy_scaled[len(mock_df_copy_scaled)-1]

from collections import Counter
#Counter(y_train)


# # Random Forest

from sklearn.ensemble import RandomForestClassifier
# Creating the random forest classifier instance.
rf_model = RandomForestClassifier(n_estimators=128, random_state=42)
# Fitting the model.
rf_model = rf_model.fit(X_train_scaled, y_train)

# Get the probability
prob = rf_model.predict_proba(X_test_scaled)

import numpy as np
labels = np.argmax(prob, axis=1)
classes = rf_model.classes_
labels = [classes[i] for i in labels]

# Making predictions using the testing data.
predictions = rf_model.predict(X_test_scaled)

# Calculating the confusion matrix
cm = confusion_matrix(y_test, predictions)

# Create a DataFrame from the confusion matrix.
cm_df = pd.DataFrame(
    cm, index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"])

print(cm_df)

# Calculating the accuracy score.
acc_score = accuracy_score(y_test, predictions)
print(acc_score)

# Displaying results
print("Confusion Matrix")
#display(cm_df)
print(f"Accuracy Score : {acc_score}")
print("Classification Report")
print(classification_report(y_test, predictions))

feat_importances = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)
feat_importances.plot(kind='bar', title='Feature Importances')
plt.ylabel('Feature Importance Score')
plt.xticks(rotation=45, ha="right")
#plt.set_xticklabels(xticklabels, rotation = 45, ha="right")
plt.show()

rf_model.feature_importances_

# calculate roc curve
fpr, tpr, thresholds = roc_curve(y_test, predictions)

# calculate AUC
auc = roc_auc_score(y_test, predictions)
print('AUC: %.3f' % auc)


# plot the roc curve for the model
pyplot.plot(fpr, tpr, linestyle='--', label='Selected RF')

# axis labels
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
# show the legend
pyplot.legend()
# show the plot
pyplot.show()


# ### Calculate Driver Score

#X_input_user
user_info

# Get the probability of crash
user_input = user_info.reshape(1, -1)
proba_user = rf_model.predict_proba(user_input)
proba_user_crash = proba_user[0][1]
print(f'proba driver: {proba_user_crash}')

from joblib import dump, load

# Save rf model
dump(rf_model, 'rf_model_v1.joblib')


from sklearn.pipeline import make_pipeline
# Create pipeline Scaler + RF model
pipeline = make_pipeline(X_scaler, rf_model)

t = pd.DataFrame([user,user])
print(t)

# Encode user input
t[1] = le_gender.transform(t[1])
t[3] = le_body.transform(t[3])
t[4] = le_make.transform(t[4])
t[5] = le_day.transform(t[5])
print(t)


user_in = t.loc[0].tolist()
print(user_in)

prob_test = pipeline.predict_proba(t)
print(prob_test)

print(prob_test[0][1])

from pickle import dump as dump_p, load as load_p
# Save label encoders
dump(le_gender, open('le_gender.pkl', 'wb'))
dump(le_body, open('le_body.pkl', 'wb'))
dump(le_make, open('le_make.pkl', 'wb'))
dump(le_day, open('le_day.pkl', 'wb'))

# Save pipeline
dump(pipeline, 'pipeline_v1.joblib')
