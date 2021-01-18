## Selected Topic
### Analyzing motor vehicle accident data in Austin.
- Using SQLite, Python and HTML, we will look at how different factors may contribute to a car crashing.

## Reason for Selecting Topic
- We want to build a model that will predict the severity of crash depending on different factors. 
- This information can be used by car insurance companies and by consumers who are trying to shop for cars keeping safety in mind.

## Description of the Source Data
TxDot Crash Query System -- This database uses a multitude of factors to input details on a car accident including but not limited to:
- Weather, longitutde, latitutde, severity, time and date of accident over the course of many years. 
- For the sake of our analysis, we will only use data from 2018-2020.

## Questions we hope to answer with the data
- How different cars perform in terms of frequency and severity of car accidents?
- How different weather types affect the frequency of car accidents?
- How demographics affect frequency of car accidents?

## Communication Protocols
- In order to keep updated on the status of each of our parts of the project, we message each other regularly through Slack. Any other questions or concerns that arise outside of our daily check-ins, we tackle during (or after) class, our group Zoom meetings, or even during weekly TA sessions.

## Deliverable 2 - Machine Learning Model
- The preliminary data includes columns that describe the environment for each crash that took place in Austin, TX. These features include the weather condition, crash severity, day of the week, vehicle make and model, etc.
- An ERD showcasing the inter-relationships between each of the features from the different datasets can be found [here](https://github.com/cedoula/Final_Project/blob/Deliverable2/QuickDBD-car_crash.png?raw=true). 
- After connecting to the database, we printed out the header for each column to see all of the features available. From that list, we chose the features that we believed would have the highest correlation with crash severity.
- The data was split into training and test data using the train_test_split function. We used the default 75% to 25% split.
- We decided to use the decision tree model for our machine learning model. We grouped our crash severity data into two categories, 0 - no injury, and 1 - injury. The benefit of this model is that it can be used to predict our binary outcome. The downside of this model is that if we choose to group our crash severity data differently (the data is grouped into 5 classifications: no injury, possible injury, non-incapacitating injury, severe injury, and fatal injury), we will not be able to use the decision tree model.

## Deliverable 2 - Presentation
- Our presentation outline can be found here [Google Slide Presentation](https://docs.google.com/presentation/d/1dQ-wwnd6MWJ3GsWzo_puQQ2VfBtjN7rRvCD3rpHtnxI/edit)

## Deliverable 2 - Dashboard
- We will be using tableau as a part of our dashboard. Our preliminary dashboard can be found here [Tableau Dashboard](https://public.tableau.com/profile/cedric.vanza#!/vizhome/Austin2018-2020CrashAnalysis/Dashboard1?publish=yes)
- The other part of our dashboard will be a webpage with an introduction and description of the project. It will include an interactive element, users will be able to select data that pertains to them (age, type of car, etc.) and click a button that will give the a risk score.
- The blueprints of our dashboard can be found [here](https://docs.google.com/document/d/1_BNqri7IxI95_6hnyu4C56DFVbYSCSsw11DYcGi8-_k/edit)