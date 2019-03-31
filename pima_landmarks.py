import googlemaps
import pprint

api_key='AIzaSyBy8qtLn1TdDj4ZC6KnTVIen-KaC6Yf-X0'
import json
gmaps=googlemaps.Client(key=api_key)

def find_landmarks(latitude,longitude,radius,business):
    location1="{},{}" .format(latitude, longitude)
    results=gmaps.places_nearby(location=location1, radius=radius,open_now=False,type=business)
    return len(results['results'])


def clossest_from(latitude,longitude,business):
    radius=25
    while find_landmarks(latitude,longitude,radius,business)==0:
        radius+=25
    return radius
