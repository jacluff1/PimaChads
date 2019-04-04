import numpy as np
import pandas as pd
import googlemaps
import pprint
api_key='AIzaSyCBMGjKqLAORj5xPpUvG0gXQOxjPaljf21'
import requests
import json

def find_landmarks(latitude,longitude,radius,business): #business should be in quotes
    url="https://maps.googleapis.com/maps/api/place/findplacefromtext/json?input={}&inputtype=textquery&fields=photos,formatted_address,name&locationbias=circle:{}@{},{},&key={}" .format(business,radius,latitude, longitude,api_key) #forms the url using the inputs given
    results=requests.get(url) #get the url content
    result=results.json() #gets the json format for the content
    return len(result)

def closest_from(latitude,longitude,business):
    radius=1609
    while find_landmarks(latitude,longitude,radius,business)==0:
        radius+=1609
    return radius



data=pd.read_csv("Main Datasheet with Owner Location Data.csv")

dist=[closest_from(data.lat[i],data.lon[i],1609,'hospital') for i in range(len(data.lon))] #distance is in meters, 1609 is 1mile

dist=pd.DataFrame(dist)

data['mindistHosp']=dist['0']/1609 #the data is appended to the data frame

# ddist=dist//(1609*5) #to get distances in less than 5miles, in 5-10miles etc
#
# ddist=pd.get_dummies(ddist, columns=['0']) #one hot encoding the new data
#
# ddist.columns=['Hosp_dist<5m','Hosp_dist5<10m', 'Hosp_dist10<15m','Hosp_dist15<20m','Hosp_dist>20m'] #renaming the data columns
#
# data=pd.concat([data,ddist], axis=1) #to append the data frame ddist to the left of data but two new columns are added that we need to remove
#
# data=data.drop(['Unnamed: 0','Unnamed: 0.1'], axis=1)

data.to_csv('Main Datasheet.csv') #save the data
