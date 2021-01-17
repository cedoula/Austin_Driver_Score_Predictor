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