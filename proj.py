#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Initialize OK
from client.api.notebook import Notebook
ok = Notebook('proj3.ok')



import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns


# ## The Data
# Attributes of all [yellow taxi](https://en.wikipedia.org/wiki/Taxicabs_of_New_York_City) trips in January 2016 are published by the [NYC Taxi and Limosine Commission](https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page).
# 
# Columns of the `taxi` table in `taxi.db` include:
# - `pickup_datetime`: date and time when the meter was disengaged
# - `dropoff_datetime`: date and time when the meter was engaged
# - `pickup_lon`: the longitude where the meter was engaged
# - `pickup_lat`: the latitude where the meter was engaged
# - `dropoff_lon`: the longitude where the meter was disengaged
# - `dropoff_lat`: the latitude where the meter was disengaged
# - `passengers`: the number of passengers in the vehicle (driver entered value)
# - `distance`: trip distance
# - `duration`: duration of the trip in seconds
# 
# Goal is to predict `duration` from the pick-up time, pick-up and drop-off locations, and distance.

# ## Part 1: Data Selection and Cleaning

import sqlite3

conn = sqlite3.connect('taxi.db')
lon_bounds = [-74.03, -73.75]
lat_bounds = [40.6, 40.88]

df = pd.read_sql_query("SELECT * FROM taxi WHERE (pickup_lon BETWEEN -74.03 AND -73.75) AND (pickup_lat BETWEEN 40.6 AND 40.88) AND (dropoff_lon BETWEEN -74.03 AND -73.75) AND (dropoff_lat BETWEEN 40.6 AND 40.88)", conn)

all_taxi = df
all_taxi.head()


def pickup_scatter(t):
    plt.scatter(t['pickup_lon'], t['pickup_lat'], s=2, alpha=0.2)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Pickup locations')
    
plt.figure(figsize=(8, 8))
pickup_scatter(all_taxi)

alltaxi1 = all_taxi[all_taxi['passengers'] > 0]
alltaxi2 = alltaxi1[alltaxi1['distance']>0]
alltaxi3 = alltaxi2[alltaxi2['duration'] >= 60]
alltaxi4 = alltaxi3[alltaxi3['duration'] <= 3600]
alltaxi5 = alltaxi4[alltaxi4['distance']/(alltaxi4['duration'] /3600) <= 100]

clean_taxi = alltaxi5


# Created a DataFrame called `manhattan_taxi` that only includes trips from `clean_taxi` that start and end within a polygon that defines the boundaries of [Manhattan Island](https://www.google.com/maps/place/Manhattan,+New+York,+NY/@40.7590402,-74.0394431,12z/data=!3m1!4b1!4m5!3m4!1s0x89c2588f046ee661:0xa0b3281fcecc08c!8m2!3d40.7830603!4d-73.9712488).


polygon = pd.read_csv('manhattan.csv')
plt.scatter(polygon['lat'], polygon['lon'])


polygon.head()

clean_taxi.iloc[[0,1,4]]

def in_manhattan(x, y):
    """Whether a longitude-latitude (x, y) pair is in the Manhattan polygon."""
    polycorners = len(polygon)
    polyX = polygon['lon']
    polyY = polygon['lat']
    
    j = polygon.shape[0] - 1
    oddNodes = False
    
    for i in range(0, polygon.shape[0]):
        if ((polyY[i] < y and polyY[j] >= y) or (polyY[j] < y and polyY[i] >= y)):
            if (polyX[i] + (y - polyY[i]) / (polyY[j]-polyY[i])*(polyX[j]-polyX[i]) < x):
#                 print("hello")
                oddNodes = not oddNodes
        j = i
    return oddNodes

inMan = []
for x in range(0, len(clean_taxi)):
    inMan.append(in_manhattan(clean_taxi.iloc[x]['pickup_lon'], 
                              clean_taxi.iloc[x]['pickup_lat']) and 
                              in_manhattan(clean_taxi.iloc[x]['dropoff_lon'], clean_taxi.iloc[x]['dropoff_lat']))
    
manhattan_taxi = clean_taxi.iloc[inMan]

clean_taxi.head()

in_manhattan(-73.9857, 40.7484)

sum(inMan)

manhattan_taxi = pd.read_csv('manhattan_taxi.csv')

plt.figure(figsize=(8, 16))
pickup_scatter(manhattan_taxi)


# a summary of the data selection and cleaning

print('of the 96445 original trips'+ str(len(all_taxi) - len(clean_taxi)) + 'anomalous trips' + str((len(all_taxi) - len(clean_taxi))
                                         /len(all_taxi))+ 'were removed through data cleaning and then the'+ 
      str(len(manhattan_taxi))+ 
      'trips within Manhattan were selected for further analysis')


# ## Part 2: Exploratory Data Analysis


manhattan_taxi.head()

import datetime
def convertDate(datestring):
    date_time_obj = datetime.datetime.strptime(datestring, '%Y-%m-%d %H:%M:%S')
    return date_time_obj.date()
dates = []
for x in range(0, len(manhattan_taxi)):
    dates.append(convertDate(manhattan_taxi.iloc[x]['pickup_datetime']))
manhattan_taxi['date'] = pd.Series(data = dates, name = 'date')



# data visualization that allows us to identify which dates were affected by the historic blizzard of January 2016

added= manhattan_taxi.copy()
added['average_speed'] = added['duration'] / added ['distance']
added.sort_values('date')
sns.boxplot(x='date', y = 'average_speed', data = added)
plt.xticks(rotation=90)
plt.show()


import calendar
import re

from datetime import date

atypical = [1, 2, 3, 18, 23, 24, 25, 26]
typical_dates = [date(2016, 1, n) for n in range(1, 32) if n not in atypical]
typical_dates

print('Typical dates:\n')
pat = '  [1-3]|18 | 23| 24|25 |26 '
print(re.sub(pat, '   ', calendar.month(2016, 1)))

final_taxi = manhattan_taxi[manhattan_taxi['date'].isin(typical_dates)]



# ## Part 3: Feature Engineering



import sklearn.model_selection

train, test = sklearn.model_selection.train_test_split(
    final_taxi, train_size=0.8, test_size=0.2, random_state=42)
print('Train:', train.shape, 'Test:', test.shape)


# ### Question 3a
# 
# box plot that compares the distributions of taxi trip durations for each day


sns.boxplot(x='date', y='duration', data=train.sort_values('date'))
plt.title('Duration by Date')




def speed(t):
    """Return a column of speeds in miles per hour."""
    return t['distance'] / t['duration'] * 60 * 60

def augment(t):
    """Augment a dataframe t with additional columns."""
    u = t.copy()
    pickup_time = pd.to_datetime(t['pickup_datetime'])
    u.loc[:, 'hour'] = pickup_time.dt.hour
    u.loc[:, 'day'] = pickup_time.dt.weekday
    u.loc[:, 'weekend'] = (pickup_time.dt.weekday >= 5).astype(int)
    u.loc[:, 'period'] = np.digitize(pickup_time.dt.hour, [0, 6, 18])
    u.loc[:, 'speed'] = speed(t)
    return u
    
train = augment(train)
test = augment(test)
train.iloc[0,:] # An example row


# Used `sns.distplot` to create an overlaid histogram comparing the distribution of average speeds for taxi rides that start in the early morning (12am-6am), day (6am-6pm; 12 hours), and night (6pm-12am; 6 hours)

ax = sns.distplot(train[train['period'] == 1]['speed'], label='Early Morning')
sns.distplot(train[train['period'] == 2]['speed'], label='Day')
sns.distplot(train[train['period'] == 3]['speed'], label='Night')
plt.legend()
plt.show()


# Find the first principle component
D = train[['pickup_lon', 'pickup_lat']].values
pca_n = len(D)
pca_means = [np.mean(train['pickup_lon'].tolist()), np.mean(train['pickup_lat'].tolist())]
X = (D - pca_means) / np.sqrt(pca_n)
u, s, vt = np.linalg.svd(X, full_matrices=False)

def add_region(t):
    """Add a region column to t based on vt above."""
    D = t[['pickup_lon', 'pickup_lat']].values
    assert D.shape[0] == t.shape[0], 'You set D using the incorrect table'
    # Always use the same data transformation used to compute vt
    X = (D - pca_means) / np.sqrt(pca_n) 
    first_pc = D @ vt[0]
    t.loc[:,'region'] = pd.qcut(first_pc, 3, labels=[0, 1, 2])
    
add_region(train)
add_region(test)


plt.figure(figsize=(8, 16))
for i in [0, 1, 2]:
    pickup_scatter(train[train['region'] == i])

from sklearn.preprocessing import StandardScaler

num_vars = ['pickup_lon', 'pickup_lat', 'dropoff_lon', 'dropoff_lat', 'distance']
cat_vars = ['hour', 'day', 'region']

scaler = StandardScaler()
scaler.fit(train[num_vars])

def design_matrix(t):
    """Create a design matrix from taxi ride dataframe t."""
    scaled = t[num_vars].copy()
    scaled.iloc[:,:] = scaler.transform(scaled) # Convert to standard units
    categoricals = [pd.get_dummies(t[s], prefix=s, drop_first=True) for s in cat_vars]
    return pd.concat([scaled] + categoricals, axis=1)

design_matrix(train).iloc[0,:]  


# ## Part 4: Model Selection

test.head()

def rmse(errors):
    """Return the root mean squared error."""
    return np.sqrt(np.mean(errors ** 2))

constant_rmse = rmse(test['duration'].tolist() - np.mean(test['duration'].tolist()))
constant_rmse


from sklearn.linear_model import LinearRegression
X_train = train[['distance']]
y_train = train.loc[:, 'duration']
X_test = test[['distance']]
model = LinearRegression()

model.fit(X_train, y_train)
y_fitted = model.predict(X_train)
simple_rmse = rmse(y_train - y_fitted)

simple_rmse

model = LinearRegression()
model.fit(design_matrix(train), y_train)
linear_rmse = rmse(model.predict(design_matrix(test)) - test['duration'])
linear_rmse



model = LinearRegression()
errors = []

for v in np.unique(train['period']):
    subset = train[train['period'] == v]
    subsetTest = test[test['period'] == v]
    model.fit(design_matrix(subset), subset['duration'])
    errors.extend(model.predict(design_matrix(subsetTest)) - subsetTest['duration'])
    
period_rmse = rmse(np.array(errors))
period_rmse


# In[50]:


ok.grade("q4d");

model = LinearRegression()
model.fit(design_matrix(train), train['speed'])
predicted_speed = model.predict(design_matrix(test))
speedseconds = predicted_speed/3600
predicted_duration = test['distance'].tolist()/speedseconds

speed_rmse = rmse(predicted_duration - test['duration'])
speed_rmse

choices = ['period', 'region', 'weekend']
train.groupby(choices).size().index.shape

model = LinearRegression()
choices = ['period', 'region', 'weekend']

def duration_error(predictions, observations):
    """Error between predictions (array) and observations (data frame)"""
    return predictions - observations['duration']

def speed_error(predictions, observations):
    """Duration error between speed predictions and duration observations"""
    return observations['distance']/(predictions) *3600 - observations['duration']

def tree_regression_errors(outcome='duration', error_fn=duration_error):
    """Return errors for all examples in test using a tree regression model."""
    errors = []
    for vs in train.groupby(choices).size().index:
        v_train, v_test = train, test
        for v, c in zip(vs, choices):
            v_train = v_train[v_train[c] == v]
            v_test = v_test[v_test[c] == v]
        model.fit(design_matrix(v_train), v_train[outcome])
        errors.extend(error_fn(model.predict(design_matrix(v_test)), v_test))
    return errors

errors = tree_regression_errors()
errors_via_speed = tree_regression_errors('speed', speed_error)
tree_rmse = rmse(np.array(errors))
tree_speed_rmse = rmse(np.array(errors_via_speed))
print('Duration:', tree_rmse, '\nSpeed:', tree_speed_rmse)

models = ['constant', 'simple', 'linear', 'period', 'speed', 'tree', 'tree_speed']
pd.DataFrame.from_dict({
    'Model': models,
    'Test RMSE': [eval(m + '_rmse') for m in models]
}).set_index('Model').plot(kind='barh');




