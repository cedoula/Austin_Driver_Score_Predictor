## Selected Topic
### Analyzing motor vehicle accident data in Austin.
- Using a variety of tools, we will look at how different factors may contribute to the severity of a car crash.

## Reason for Selecting Topic
- We want to build a model that will predict the severity of crash depending on different factors. 
- This information can be used by car insurance companies and by consumers who are trying to shop for cars keeping safety in mind.

## Description of the Source Data
TxDot Crash Query System -- This database uses a multitude of factors to input details on a car accident including but not limited to:
- Weather, longitutde, latitutde, severity, time and date of accident over the course of many years. 
- For the sake of our analysis, we will only use data from 2018-2020.
Additionally, we used NHTSA WebAPIs - https://one.nhtsa.gov/webapi/Default.aspx?SafetyRatings/API/5
- This webAPI gives access to the New Car Assessment Program - 5 Star Safety Rating of the US Department Of Transportation, to retrieve overall safety ratings of the cars involved in the crashes.

## Questions we hope to answer with the data
- How do different cars perform in terms of frequency and severity of car accidents?
- How do different weather types affect the frequency of car accidents?
- How demographics affect frequency of car accidents?
- Where do most accidents occur in Austin?

## Communication Protocols
- In order to keep updated on the status of each of our parts of the project, we message each other regularly through Slack. Any other questions or concerns that arise outside of our daily check-ins, we tackle during (or after) class, our group Zoom meetings, or even during weekly TA sessions.

## Tools
- Creating Database
    - PostgreSQL
    - Amazon Web Services (AWS)
- Connecting to Database
    - SQLAlchemy, Psycopg2
- Analyzing Data
    - Pandas
- Machine Learning
    - Imbalanced-learn
    - Scikit-Learn
    - Tensorflow
- Dashboard
    - Javascript
    - Flask
    - CSS

## Machine Learning Model
- The preliminary data includes columns that describe the environment for each crash that took place in Austin, TX. These features include the weather condition, crash severity, day of the week, vehicle make and model, etc.
- An ERD showcasing the inter-relationships between each of the features from the different datasets can be found [here](https://github.com/cedoula/Final_Project/blob/Deliverable2/QuickDBD-car_crash.png?raw=true). 
- After connecting to the database, we printed out the header for each column to see all of the features available. From that list, we chose the features that we believed would have the highest correlation with crash severity.
- The data was split into training and test data using the train_test_split function. We used the default 75% to 25% split.
- After careful analyzing, it was determined that the linear models only yielded about 50% correlation. Altering the parameters, such as increasing max iterations and n_jobs,  to these did not increase the accuracy. Neural network model was then used to see if it would have a higher accuracy rate. After adding 8 layers (using Relu, Swish and Sigmoid), the accuracy rate was still at 54%, with 69% loss. This means our model could only accurately predict the outcome of the severity of a crash about 50% of the time. 
- We decided to use the decision tree model for our machine learning model. We grouped our crash severity data into two categories, 0 - no injury, and 1 - injury. The benefit of this model is that it can be used to predict our binary outcome. The downside of this model is that if we choose to group our crash severity data differently (the data is grouped into 5 classifications: no injury, possible injury, non-incapacitating injury, severe injury, and fatal injury), we will not be able to use the decision tree model.

## Presentation
- Our presentation can be found here [Google Slide Presentation](https://docs.google.com/presentation/d/1dQ-wwnd6MWJ3GsWzo_puQQ2VfBtjN7rRvCD3rpHtnxI/edit)

## Dashboard
- We used Tableau as a part of our dashboard. Our Tableau analysis can be found here [Tableau Dashboard](https://public.tableau.com/profile/cedric.vanza#!/vizhome/Austin2018-2020CrashAnalysis/Dashboard1?publish=yes)
- The other part of our dashboard is an interactive webpage using machine learning to calculate a driver score. It includes an interactive element, users are able to select data that pertains to them (age, type of car, etc.) and click a button that will give the a risk score.\
The link to the dashboard repository is [Link Repo Dashboard](https://github.com/cedoula/Austin_Driver_Score).
- The blueprints of our dashboard can be found [here](https://docs.google.com/document/d/1_BNqri7IxI95_6hnyu4C56DFVbYSCSsw11DYcGi8-_k/edit)

- Dashboard Live Demo

[![Austin Driver Score - Demo](https://user-images.githubusercontent.com/68669675/106414143-88f65a00-6411-11eb-885e-dd1407d896e8.png)](https://vimeo.com/506919406 "Austin Driver Score Demo - Click to Watch!")

