# Disaster Response Pipeline Project

## Project Introduction

After a disaster, we usually get millions of messages either directly or via social media, and it can too difficult for disaster response organizations to pull out significant messages and arrange related organizations to take care of different parts of problem.

Therefore, the aim of this project is to build a web app that can classify each message into which categories it belongs to by machine learning according to the dataset provided by Figure Eight. For example, if you input "I need water" on the web, you will found "Related", "Request", "Air Related" and "Water" are highlighted, which means the message belongs to these categories. According to classification results, corresponding organizations can implement rescue.

Below is what the homepage looks like. It displays some visualizations about the dataset.

![homepage](C:\Users\zengh\Desktop\Project2\homepage.jpeg)

If you input messages in blank bar, and click "Classify Message" button, then you can get classification results as below.

![classify_page](C:\Users\zengh\Desktop\Project2\classify_page.jpeg)

## Description of each file

- data

> - disaster_messages.csv - including 26248 messages and their unique id.
> - disaster_categories.csv - including categories of each messages.
> - process_data.py - script used to clean and save the data.

- models

> - train_classifier.py - script used to train and save the model.
> - new_features.py - script that definite the feature classes used in train_classifier.py.

- app

> - templates
>   - master.html - html file used to build homepage.
>   - go.html - html file used to build query page.
> - run.py - script used to run the web app.

## How to run the web in your local

If you want to run the web app in your local, here are some steps you need to follow.

1. **Clone this repository to your computer**

2. **Create a terminal from the project's root directory**

3. **Run the following commands to set up your database and model**

   - To run ETL pipeline that clean the data and store it into database

     `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`

   - To run ML pipeline that train the model and save it into pickle file

     `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

4. **Run the following commands to run the web app**

   - Turn to app directory

     `cd app`

   - To run the web app

     `python run.py`

5. Go to http://localhost:3001

## Libraries

All libraries I used in this project are as following:

- sys

- numpy
- pandas
- sqlalchemy
- re
- nltk
- sklearn
- json
- plotly
- flask

## Acknowledgement

Data Source - Disaster response data from [Figure Eight](https://appen.com/)
