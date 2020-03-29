# Libraries
import os
import pandas as pd
from fbprophet import Prophet
import json
import sys

# Read
os.chdir(sys.argv)

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Overview
print('List of effected countries:')
[print(i) for i in train['Country/Region'].unique()]
print('Case records for coronavirus begin on {}..'.format(min(train.Date)))
print('Training data ends on {}..'.format(max(train.Date)))

# Initialize
casestrainingdata = {}
deathstrainingdata = {}
future = pd.DataFrame(pd.date_range(min(test['Date']), max(test['Date'])), columns = ['ds'])

# Prepare - training
for nm, grp in train.groupby('Country/Region'):
    
    cases = grp[['Date', 'ConfirmedCases']].rename({'Date': 'ds', 'ConfirmedCases': 'y'}, axis = 1)
    casestrainingdata[nm] = cases
    
for nm, grp in train.groupby('Country/Region'):
    
    deaths = grp[['Date', 'Fatalities']].rename({'Date': 'ds', 'Fatalities': 'y'}, axis = 1)
    deathstrainingdata[nm] = deaths
    
# Prepare - testing
for nm, grp in test.groupby('Country/Region'):
    
    preds[nm] = grp['ForecastId']

# Function
def prophecy(location):
    
    # Get
    cases = casestrainingdata[location]
    deaths = deathstrainingdata[location]
    
    # Train
    m = Prophet()
    m.fit(cases)
    
    # Predict
    preds = abs(round(m.predict(future)['yhat']))
    preds[location]['ConfirmedCases'] = preds
    
    # Train
    m.fit(deaths)    
    
    # Predict
    preds = abs(round(m.predict(future)['yhat']))
    preds[location]['ConfirmedCases'] = preds
    
# Iterate
submission = pd.DataFrame()

for k, v in preds.items():
    
    submission = submission.append(v)
    
# Write
submission.to_csv('submission.csv', index = False)