# pip install google-api-python-client
# pip install google-auth-httplib2
# pip install google-auth-oauthlib
import os
import requests_google_photos #pip install requests_google_photos
import pandas #pip install pandas
import requests #pip install requests

CLIENT_SECRET_FILE = 'client_secret_416324643060-3spbhn0quv27oon0t7vv9g08a2fgoavn.apps.googleusercontent.com.json'
API_NAME = 'photoslibrary'
API_VERSION = 'v1'
SCOPES = ['https://photos.googlepis.com/auth']


service = requests_google_photos.Create_Service(CLIENT_SECRET_FILE, API_NAME, API_VERSION, SCOPES)
my_albums = service.albims().list().execute()