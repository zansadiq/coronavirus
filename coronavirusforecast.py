# Libraries
import os
import pandas as pd
from fbprophet import Prophet
import json
import sys

# Read
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Initialize
casestrainingdata = {}
deathstrainingdata = {}
preds = {}
logpreds = {}
future = pd.DataFrame(pd.date_range(min(test['Date']), max(test['Date'])), columns = ['ds'])

# Prepare - training
for nm, grp in train.groupby('Country_Region'):
    
    casestrainingdata[nm] = grp[['Date', 'ConfirmedCases']].rename({'Date': 'ds', 'ConfirmedCases': 'y'}, axis = 1)
    
for nm, grp in train.groupby('Country_Region'):
    
    deathstrainingdata[nm] = grp[['Date', 'Fatalities']].rename({'Date': 'ds', 'Fatalities': 'y'}, axis = 1)
    
# Prepare - testing
for nm, grp in test.groupby('Country_Region'):
    
    preds[nm] = grp['ForecastId']
    logpreds[nm] = grp['ForecastId']
    
# Function
def linear_prophecy(location):
    
    # Get
    cases = casestrainingdata[location]
    deaths = deathstrainingdata[location]
    
    # Train
    m = Prophet()
    m.fit(cases)
    
    # Predict
    casepreds = abs(round(m.predict(future)['yhat']))
    preds[location]['ConfirmedCases'] = casepreds
    
    # Train
    m = Prophet()
    m.fit(deaths)    
    
    # Predict
    deathpreds = abs(round(m.predict(future)['yhat']))
    preds[location]['Fatalities'] = deathpreds

def log_prophecy(location):
    
    # Get
    cases = casestrainingdata[location]
    cases['floor'] = [0] * len(cases)
    cases['cap'] = [max(train.loc[train['Country_Region'] == nm]['ConfirmedCases'])] * len(cases)
        
    # Train
    m = Prophet(growth = 'logistic')
    m.fit(cases)
    
    # Predict
    fut = m.make_future_dataframe(len(future))
    fut['floor'] = 0
    fut['cap'] = [max(train.loc[train['Country_Region'] == nm]['ConfirmedCases'])] * len(fut)
    
    logpreds[location]['ConfirmedCases'] = m.predict(fut)
    
# Execute
for k, v in preds.items():
    
    linear_prophecy(k)
    log_prophecy(k)
    
# Write
for k, v in preds.items():
    
    submission = submission.append(v)
    
submission.to_csv('submission.csv', index = False)