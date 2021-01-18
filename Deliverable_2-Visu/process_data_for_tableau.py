#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import dependencies
import pandas as pd
import sqlite3


# In[2]:


# Prepare the csv file
# Import full crash data csv
file = "2018-2020_Austin_crash_data.csv"
full_crash_df = pd.read_csv(file)


# In[4]:


# Select columns for Driver table
driver_mock_df = full_crash_df[['Crash ID', 'Crash Severity','Person Age', 'Person Gender', 'Person Type', 'Vehicle Body Style', 'Vehicle Model Year']]
driver_mock_df.head()


# In[5]:


driver_mock_df.to_csv("driver_mock.csv")


# In[7]:


# feed driver_mock csv into mock sqlite table
#file = "driver_mock.csv"
#data = pd.read_csv(file, index_col='IDX')
#data.head()


# In[8]:


data.to_sql("Driver_mock_data", conn)


# In[9]:


conn.execute("SELECT * FROM Driver_mock_data").fetchall()


# In[10]:


# Select columns for Environment table
env_mock_df = full_crash_df[['Crash ID', 'Crash Severity','Crash Time', 'Crash Year', 'Day of Week', 'Latitude', 'Longitude', 'Light Condition', 
                            'Surface Condition', 'Weather Condition', 'Highway Number', 'Highway System']]
# Remove duplicates
env_mock_df = env_mock_df.drop_duplicates(subset='Crash ID', keep="first")
env_mock_df.head()


# In[11]:


# Clean env data and convert lat/lon to float
print(env_mock_df.shape)
env_mock_df = env_mock_df[env_mock_df['Latitude'] != 'No Data']
env_mock_df = env_mock_df[env_mock_df['Longitude'] != 'No Data']
env_mock_df = env_mock_df[env_mock_df['Crash Severity'] != '99 - UNKNOWN']
env_mock_df['Latitude'] = env_mock_df['Latitude'].astype(float)
env_mock_df['Longitude'] = env_mock_df['Longitude'].astype(float)
print(env_mock_df.shape)
env_mock_df.dtypes


# In[12]:


env_mock_df.head()


# ### Generate zip codes from lat/lon

# In[ ]:


# use geopy to get zipcodes
import geopy


# In[ ]:


def get_zipcode(df, geolocator, lat_field, lon_field):
    location = geolocator.reverse((df[lat_field], df[lon_field]))
    try:
        return location.raw['address']['postcode']
    except: return " "


# In[ ]:


geolocator = geopy.Nominatim(user_agent='my-app')


# In[ ]:


df = pd.DataFrame({
    'Lat': env_mock_df['Latitude'],
    'Lon': env_mock_df['Longitude']
})


# In[ ]:


df = df.reset_index()


# In[ ]:


#df = df.drop('index', axis=1)
df


# In[ ]:


#zipcodes = df.apply(get_zipcode, axis=1, geolocator=geolocator, lat_field='Lat', lon_field='Lon')


# In[ ]:


#zipcodes


# In[ ]:


zips = []


# In[ ]:


from geopy.geocoders import Nominatim
geolocator = Nominatim(user_agent="my_app")
for j in range(len(df)):
    location = geolocator.reverse(f'{df["Lat"][j]}, {df["Lon"][j]}')
    try:
        zips.append(location.raw['address']['postcode'])
        print(f'raw #{j} done with zipcode {location.raw["address"]["postcode"]}.')
    except: 
        zips.append(" ")
        print(f'raw #{j} done without zipcode.')


# In[ ]:


len(zips)


# In[13]:


env_mock_df = env_mock_df.reset_index()


# In[15]:


env_mock_df


# In[16]:


# get zipcodes after manually getting the missing ones
file = "BU/env_mock_or.csv"
bu_df = pd.read_csv(file)
zips = bu_df["Zip Codes"]


# In[17]:


env_mock_df["Zip Codes"] = zips


# In[18]:


env_mock_df.head()


# In[19]:


env_mock_df = env_mock_df.drop("index", axis=1)


# In[20]:


env_mock_df.head()


# ### Convert time format from hhmm to hh:mm

# In[21]:


import datetime
def convert_string_to_time(t):
    if len(str(t)) == 1:
        return str(datetime.time(hour=0, minute=int(str(t)[0])))
    elif len(str(t)) == 2:
        return str(datetime.time(hour=0, minute=int(str(t)[0:])))
    elif len(str(t)) == 3:
        return str(datetime.time(hour=int(str(t)[0]), minute=int(str(t)[1:3])))
    else:
        return str(datetime.time(hour=int(str(t)[0:2]), minute=int(str(t)[2:4])))


# In[23]:


convert_string_to_time("5")


# In[24]:


env_mock_df["Crash Time"] = env_mock_df["Crash Time"].apply(convert_string_to_time)


# In[25]:


env_mock_df.head()


# In[26]:


env_mock_df.to_csv("env_mock.csv")


# In[27]:


#data.to_sql("Env_mock_data", conn)
#env_mock_df.to_sql("Env_mock_data", conn)


# In[28]:


#conn.execute("SELECT * FROM Env_mock_data").fetchall()


# ### Use one.nhtsa.gov/webapi to retrieve car safety ratings using car year, make and model as input

# In[3]:


import requests


# In[ ]:


# base url for OWM calls
base_url = "https://one.nhtsa.gov/webapi/api/SafetyRatings/modelyear/2006/make/VOLKSWAGEN/model/BEETLE?format=json"
result = requests.get(base_url).json()
print(result['Results'][0]['VehicleId'])


# In[ ]:


base_url = "https://one.nhtsa.gov/webapi//api/SafetyRatings/VehicleId/4126?format=json"
requests.get(base_url).json()


# In[4]:


# Select columns for Car table
car_df = full_crash_df[['Vehicle Make', 'Vehicle Model Name','Vehicle Model Year']]
car_df.head()


# In[5]:


import numpy as np
car_df['Vehicle Model'] = car_df['Vehicle Model Name'].str.extract(r'(\w+\-*\s*\w*)\s')
car_df.head(30)


# In[9]:


car_df = car_df[(car_df['Vehicle Make'] != "No Data") & (car_df['Vehicle Make'] != "UNKNOWN") & (car_df['Vehicle Model'] != "UNKNOWN")]
car_df.value_counts()


# In[10]:


car_df["Vehicle Model"] = car_df["Vehicle Model"].replace(' ', '%20', regex=True)


# In[11]:


car_df.head(30)


# In[12]:


def get_rating(car_df, year, model, make):
    base_url = f'https://one.nhtsa.gov/webapi/api/SafetyRatings/modelyear/{car_df[year]}/make/{car_df[make]}/model/{car_df[model]}?format=json'
    try:
        carId = requests.get(base_url).json()['Results'][0]['VehicleId']
        base_url = f"https://one.nhtsa.gov/webapi//api/SafetyRatings/VehicleId/{carId}?format=json"
        print(f'Retrieved rating for {car_df[year]} {car_df[make]} {car_df[model]}.')
        return requests.get(base_url).json()['Results'][0]['OverallRating']
    except: 
        try:
            if "%20" in car_df[model]:
                car_df[model] = car_df[model].replace('%20', '', regex=True)
                base_url = f'https://one.nhtsa.gov/webapi/api/SafetyRatings/modelyear/{car_df[year]}/make/{car_df[make]}/model/{car_df[model]}?format=json'
                carId = requests.get(base_url).json()['Results'][0]['VehicleId']
                base_url = f"https://one.nhtsa.gov/webapi//api/SafetyRatings/VehicleId/{carId}?format=json"
                print(f'Retrieved rating for {car_df[year]} {car_df[make]} {car_df[model]}.')
                return requests.get(base_url).json()['Results'][0]['OverallRating']
        except:
            print('Vehicle Not found')
            return "Vehicle Not Found."


# In[13]:


car_df['Rating'] = car_df.apply(get_rating, axis=1, year='Vehicle Model Year', model='Vehicle Model', make='Vehicle Make')


# In[14]:


car_df['Rating'].value_counts()


# In[9]:


car_df = car_df.drop_duplicates()
car_df = car_df.reset_index()


# In[15]:


car_df.to_csv("car_data.csv")


# In[ ]:




