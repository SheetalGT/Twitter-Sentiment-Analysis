import tweepy
import csv
import pandas as pd
import sys
import unicodedata
####input your credentials here
consumer_key = 'QcYZ7ZStYMUvwWGFqBWojKXqt'
consumer_secret = 'Msit20EnClYcr3gfAWbtHaDL3ZC8mA5H8oLFdgF8YYZoul88Ro'
access_token = '1183741008435122177-bDXnPMgEyKWPEDBtQiFkkeXw5aq17J'
access_token_secret = 'yXutFayENtHJUTRwQHjDKbVc4v2wpJroBjhdU1LkaXomC'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth,wait_on_rate_limit=True)
#####United Airlines
# Open/Create a file to append data
csvFile = open('cybertruck.csv', 'a')
#Use csv Writer
csvWriter = csv.writer(csvFile)


for tweet in tweepy.Cursor(api.search,q="#tesla",count=100,
                           lang="en",
                           since="2008-02-15").items():
    #print (tweet.created_at, tweet.text)
    #csvWriter.writerow([tweet.user.id,tweet.user.screen_name,tweet.created_at, tweet.text.encode('utf-8')])
    csvWriter.writerow([tweet.user.id,tweet.user.screen_name,tweet.created_at,unicodedata.normalize('NFKD', tweet.text).encode('ascii','ignore')])
    #print(tweet.id,tweet.user.screen_name)
