# Import our dependencies
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd
import os
from pswd import db_password
from psycopg2 import sql, connect

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

df.head()

# iterating the columns 
for col in df.columns: 
    print(col)

"""# Preprocessing Data"""

# I selected these features because I think they will have a high correlation with crash severity
mock_df = df[['Crash Severity', 'Person Age', 'Person Gender', 'Person Type', 'Light Condition', 'Weather Condition', 'Vehicle Body Style', 'Vehicle Make', 'Day of Week','Rating']]
mock_df.head()

print(mock_df.shape)

# Drop rows with Unknown
mock_df = mock_df[mock_df['Crash Severity'] != "99 - UNKNOWN"]
mock_df = mock_df[mock_df['Person Age'] != "99 - UNKNOWN"]
mock_df = mock_df[mock_df['Person Gender'] != "99 - UNKNOWN"]
mock_df = mock_df[mock_df['Person Type'] != "99 - UNKNOWN"]
mock_df = mock_df[mock_df['Light Condition'] != "99 - UNKNOWN"]
mock_df = mock_df[mock_df['Weather Condition'] != "99 - UNKNOWN"]
mock_df = mock_df[mock_df['Vehicle Body Style'] != "99 - UNKNOWN"]
mock_df = mock_df[mock_df['Vehicle Make'] != "99 - UNKNOWN"]
mock_df = mock_df[mock_df['Rating'] != "99 - UNKNOWN"]
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

mock_df['Vehicle Body Style'].value_counts()

# Clean Weather Data Column
mock_df['Weather Condition'].value_counts()

# Group Weather Condition into three categories, Clear, Cloudy, and Other.
mock_df.loc[(mock_df['Weather Condition'] == "1 - CLEAR") , 'Weather Condition'] = "CLEAR"
mock_df.loc[(mock_df['Weather Condition'] == "2 - CLOUDY"), 'Weather Condition'] = "CLOUDY"
mock_df.loc[(mock_df['Weather Condition'] == "3 - RAIN") | (mock_df['Weather Condition'] == "6 - FOG") | (mock_df['Weather Condition'] == "4 - SLEET/HAIL") | (mock_df['Weather Condition'] == "98 - OTHER (EXPLAIN IN NARRATIVE)") | (mock_df['Weather Condition'] == "5 - SNOW") | (mock_df['Weather Condition'] == "7 - BLOWING SAND/SNOW") | (mock_df['Weather Condition'] == "8 - SEVERE CROSSWINDS"), "Weather Condition"] = "OTHER"
mock_df['Weather Condition'].value_counts()

# Clean Person Gender column
mock_df['Person Gender'].value_counts()

# Group Gender into two classes: 0 for Female and 1 for Male
mock_df.loc[mock_df['Person Gender'] == "2 - FEMALE", 'Person Gender'] = "FEMALE"
mock_df.loc[mock_df['Person Gender'] == "1 - MALE", 'Person Gender'] = "MALE"

# Clean Person Type column
mock_df['Person Type'].value_counts()

# Clean values for Person Type
mock_df.loc[mock_df['Person Type'] == "1 - DRIVER", 'Person Type'] = "DRIVER"
mock_df.loc[mock_df['Person Type'] == "5 - DRIVER OF MOTORCYCLE TYPE VEHICLE", 'Person Type'] = "MOTORCYCLE DRIVER"

# Clean Crash Severity column
mock_df['Crash Severity'].value_counts()

# Create a function that groups crash severity into two groups, 1 for No injury, 2 for all other injuries.
def group_crash(X):
    if X == "N - NOT INJURED":
        return 0
    else:
        return 1

# Group Crash Severity into two classes in order for the ML to predict the level of severity.
mock_df['Crash Severity'] = mock_df['Crash Severity'].apply(group_crash)

mock_df['Crash Severity'].value_counts()

mock_df.dtypes

# Clean Person Age
mock_df['Person Age'].value_counts()

mock_df = mock_df[mock_df['Person Age'] != "No Data"]

mock_df["Person Age"] = mock_df["Person Age"].astype(int)

# Clean Vehicle Make Column
mock_df['Vehicle Make'].value_counts()

# Map over the Vehicle Make column and replace the values with "OTHER" where the value count is less than 9000.
mock_df[['Vehicle Make']] = mock_df[['Vehicle Make']].where(mock_df.apply(lambda x: x.map(x.value_counts()))>=900, "OTHER")

mock_df['Vehicle Make'].value_counts()

# Clean Ratings
mock_df['Rating'].value_counts()

mock_df = mock_df[mock_df['Rating'] != "Not Rated"]
mock_df = mock_df[mock_df['Rating'] != "Vehicle Not Found."]
mock_df.shape

mock_df['Rating'].value_counts()

mock_df.head()

# Check for Null values
mock_df.isna().sum()

#Drop Null values
mock_df = mock_df.dropna()
mock_df.head()

# Clean Light Condition Column
print(mock_df['Light Condition'].value_counts())

# Create a cleaning function to bin the light condition column into Daylight and Other
def clean_light(cond):
    if cond == "1 - DAYLIGHT":
        return "DAYLIGHT"
    else:
        return "OTHER"

# Apply the function to the Light Condition column
mock_df['Light Condition'] = mock_df['Light Condition'].apply(clean_light)
mock_df.head()

mock_df.dtypes

mock_df["Rating"] = mock_df["Rating"].astype(int)
mock_df['Crash Severity'] = mock_df['Crash Severity'].astype(int)

"""# Prepare Data for ML Model"""

# Drop Person Type 
mock_df = mock_df.drop("Person Type", axis=1)
mock_df.head()

# Generate the categorical variable list
feat_cat = ['Person Gender','Light Condition', 'Weather Condition', 'Vehicle Body Style', 'Vehicle Make', 'Day of Week']

# Create a OneHotEncoder instance
enc = OneHotEncoder(sparse=False)

# Fit and transform the OneHotEncoder using the categorical variable list
encode_df = pd.DataFrame(enc.fit_transform(mock_df[feat_cat]))

# Add the encoded variable names to the dataframe
encode_df.columns = enc.get_feature_names(feat_cat)
encode_df.head()

# Merge one-hot encoded features and drop the originals
mock_df2 = mock_df.merge(encode_df, left_index=True, right_index=True).drop(columns=feat_cat, axis=1)
mock_df2.head()

mock_df2.corr()

# Split our preprocessed data into our features and target arrays
y = mock_df2["Crash Severity"]
X = mock_df2.drop("Crash Severity", axis=1)

# Split the preprocessed data into a training and testing dataset
# The training and testing data is split into 75% and 25%, respectively.
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, stratify=y)

# Create a StandardScaler instances
scaler = StandardScaler()

# Fit the StandardScaler
X_scaler = scaler.fit(X_train)

# Scale the data
X_train_scaled = X_scaler.transform(X_train)
X_test_scaled = X_scaler.transform(X_test)

"""# Undersampling"""
# We are using undersampling to normalize our data.

from imblearn.under_sampling import RandomUnderSampler
ros = RandomUnderSampler(random_state=1)
X_resampled2, y_resampled2 = ros.fit_resample(X_train_scaled, y_train)
Counter(y_resampled)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression(solver='lbfgs', random_state=1)
model.fit(X_resampled2, y_resampled2)

from sklearn.metrics import confusion_matrix
y_pred = model.predict(X_test_scaled)
confusion_matrix(y_test, y_pred)

from sklearn.metrics import balanced_accuracy_score
balanced_accuracy_score(y_test, y_pred)

from imblearn.metrics import classification_report_imbalanced
print(classification_report_imbalanced(y_test, y_pred))
