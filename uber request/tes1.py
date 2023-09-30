import os #provides functions for interacting with the operating system
import numpy as np
from matplotlib import *
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv('Uber Request Data.csv')
print(df)
print(df.head())
print(df.describe())
print(df.info())
####################Convert timestamp datatype to object(time)
df["Request timestamp"] = pd.to_datetime(df["Request timestamp"],format='mixed')
df["Drop timestamp"] = pd.to_datetime(df["Drop timestamp"],format='mixed')
##########
for col in df.columns[1:4]:
    print(col, np.unique(df[col]))
    print(col, df[col].value_counts())
#######################
print(df.isna().sum())
#######################
print(df.describe())
###########################extract hours
df["Requesthour"] = df["Request timestamp"].dt.hour#########to know distribution of hours of request
print(df.head())
############################
sns.countplot(x='Status',data=df)
plt.show()########1-most of trips completed and alot of them there is no car available.
#################################################################################################
sns.countplot(x='Status',data=df,hue='Pickup point')
plt.show()######1-most of trips which cancelled are at city , and most of trips which has no car available are at airport
########################################################## density of request hour
sns.distplot(df.Requesthour,bins=24, color='r').set_title('distribution of request hour')########3most of requests at 20 pm and at 7 am 
plt.show()
##########################percent of requests at airbort and city
df["Pickup point"].value_counts().plot.pie(autopct='%1.0f%%')
plt.show()
#################################percent of status
df["Status"].value_counts().plot.pie(autopct='%1.0f%%')
plt.show()############we show that 60% of trips not completed due to drivers by cancelled or by no car available so we need to give them bonus at the time which most trips cancelled
#####################dividing day into 5 times 
def time_period(x):
    if x < 5:
        return "Early Morning"
    elif 5 <= x < 10:
        return "Morning"
    elif 10 <= x < 17:
        return "Day Time"
    elif 17 <= x < 22:
        return "Evening"
    else:
        return "Late Night"
df['Timeslot'] = df['Requesthour'].apply(lambda x: time_period(x))
print(df.head())
df['Timeslot'].value_counts().plot.bar()
plt.show()
################### time slots vs status
sns.countplot(x='Timeslot',hue='Status',data=df)
plt.show()########from plot we found that most time that is there is no car available is at eveninig 
#############solution:1-uper can provide ponus at this times to encourage driver.
print(df[df['Pickup point']=='Airport'].Status)
x11=df[df['Pickup point']=='Airport'].Timeslot
sns.countplot(x=x11,hue=(df[df['Pickup point']=='Airport'].Status))
plt.title('air port at rush hour')############rush hours at air port is at evenning
plt.show()
#sns.countplot(x='Time slot',data=df,hue=df[df['Pickup point']=='City'].Status)############rush hours at air port is at evenning
#plt.show()
x1=df[df['Pickup point']=='City'].Timeslot
sns.countplot(x=x1,hue=df[df['Pickup point']=='City'].Status)
plt.title('city at rush hours')
plt .show()
############################obsevation:
#1) Maximum number of cancellations are being done during morning hours from 5AM to 10AM by the drivers, this happens mainly due to less demand for the cabs from airport to city. This might be due to few number of flight arrivals at the airport in the morning. So drivers are not willing to take the trip as they will not have a booking to return back to the city, hence they cancel the trip.

#2) Customers find massive number of unavailable cars during evening hours from 5PM to 10PM, this could be due to huge number of flight arrivals and departures in the evening, that results in high demand for the cabs. Hence customers could not find a cab in the evening hours

########333solution
#1) Uber can also put some offers for the customers during late nights where demand is low and if possible increase the number of cabs during busy hours which would be benefits.
#2)uber management to provide some bonus for each trip.

