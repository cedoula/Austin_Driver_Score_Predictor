# Technologies Used
## Data Cleaning and Analysis
We'll be using Pandas to clean the data and perform an exploratory analysis. Further analysis will be completed using SQLite and jupyter notebook.
We'll use ORM (Object Relational Mapper) to create classes in our code to be mapped into specific tables in the database.
 - Classes will include: type of car, weather, demographics of the parties involved.

## Database Storage
SQLite is the database we intend to use, and we will integrate using Flask
We'll be using SQLAlchemy to query SQLite database
In the beginning, our code may look like this, after importing numpy and pandas: 
 - import sqlalchemy
 - from sqlalchemy.ext.automap import automap_base
 - from sqlalchemy.orm import session
 - from sqlalchemy import create_engine, func

## Machine Learning
SciKitLearn is the ML library we'll be using to create a classifier. Our training and testing setup is 75% training and 25% test. 
 - We will be using this library to assess logistic regression to determine possible relationship between types of vehicles, weather and type of driver. This can even help in determining safety of vehicle.

## Dashboard
In addition to using a Flask template, we will use HTML to utilize D3.js for a fully functioning and interactive dashboard, which can be updated in real time depending on the input by the user. It will be hosted on HTML. 