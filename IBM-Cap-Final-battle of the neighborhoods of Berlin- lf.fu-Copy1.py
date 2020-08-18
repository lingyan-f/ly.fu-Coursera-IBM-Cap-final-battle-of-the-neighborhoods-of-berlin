#!/usr/bin/env python
# coding: utf-8

# In[1]:


#ly-fu-coursera-IBM-Cap-Final-battle of the neighborhoods of Berlin


# In[45]:


# 1. Importing all the necessary libraries we will be needing to do the Ananlysis


import numpy as np # library to handle data in a vectorized manner

import pandas as pd # library for data analsysis
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

import json # library to handle JSON files

get_ipython().system("conda install -c conda-forge geopy --yes # uncomment this line if you haven't completed the Foursquare API lab")
from geopy.geocoders import Nominatim # convert an address into latitude and longitude values

import requests # library to handle requests
from pandas.io.json import json_normalize # tranform JSON file into a pandas dataframe

# Matplotlib and associated plotting modules
import matplotlib.cm as cm
import matplotlib.colors as colors

# import k-means from clustering stage
from sklearn.cluster import KMeans

# for webscraping import Beautiful Soup 
from bs4 import BeautifulSoup

import xml

get_ipython().system("conda install -c conda-forge geopy --yes # uncomment this line if you haven't completed the Foursquare API lab")
from geopy.geocoders import Nominatim # convert an address into latitude and longitude values
import geocoder
import geopy

get_ipython().system("conda install -c conda-forge folium=0.5.0 --yes # uncomment this line if you haven't completed the Foursquare API lab")
import folium # map rendering library

from pandas.core.frame import DataFrame

print('Libraries imported.')


# In[3]:


#2. Scraping Berlin Boroughs and neighborhoods table from Wikipedia
link = 'https://en.wikipedia.org/wiki/Boroughs_and_neighborhoods_of_Berlin'
page = requests.get(link).text
soup = BeautifulSoup(page, 'lxml')
print(soup.prettify)


# In[4]:


neighborhoodList = []
for i in range(1, 13):
    table = soup.find_all('table', {'class':'sortable wikitable'})[i]
    links=table.find_all('a')[1:]
    for link in links:
        neighborhoodList.append(link.get('title'))
print(type(table))
print(type(links))
#neighborhoodList
print(neighborhoodList)


# In[16]:


df=pd.DataFrame()
df['Neighborhood']=neighborhoodList

df


# In[17]:


df=df.dropna()
df=df.reset_index(drop=True)
df


# In[18]:


print(type(df))


# In[19]:


df_str=df.astype(str)


# In[32]:


# delete the rows that contain "District map of"
test1= list(df_neg.Neighborhood)
test2=list(df_str.Neighborhood)
ret=list(set(test2)^set(test1))
df = df_str[df_str.Neighborhood.isin(ret)]
df


# In[33]:


df=df.reset_index(drop=True)
df


# In[34]:


print('The dataframe has {} neighborhoods.'.format(
        
    df.shape[0])
)


# In[35]:


#3. Get the geographical coordinates


# In[36]:


# define a function to get coordinates
def get_latlng(neighborhood):
    # initialize your variable to None
    lat_lng_coords = None
    # loop until you get the coordinates
    while(lat_lng_coords is None):
        g = geocoder.arcgis('{}, Berlin, Germany'.format(neighborhood))
        lat_lng_coords = g.latlng
    return lat_lng_coords
# call the function to get the coordinates, store in a new list using list comprehension
coords = [ get_latlng(neighborhoodlist) for neighborhoodlist in df["Neighborhood"].tolist() ]
coords


# In[44]:


print(type(coords))


# In[48]:


# create temporary dataframe to populate the coordinates into Latitude and Longitude
df_coords = pd.DataFrame(coords, columns=['Latitude', 'Longitude'])
# merge the coordinates into the original dataframe
df['Latitude'] = df_coords['Latitude']
df['Longitude'] = df_coords['Longitude']
print(df.shape)
df


# In[49]:


#save df as csv
df.to_csv("cap_df.csv", index=False)


# In[50]:


#4. Create a map of Berlin, Germany with neighborhoods superimposed on top


# In[51]:


# get the coordinates of Berlin
#Made a mistake:user_agent muss be personal
address = 'Berlin, Germany'
geolocator = Nominatim(user_agent="user_agent_lyan.fu.fly@gmail.com")
location = geolocator.geocode(address)
latitude = location.latitude
longitude = location.longitude
print('The geograpical coordinate of Berlin, Germany {}, {}.'.format(latitude, longitude))


# In[52]:


# create map of New York using latitude and longitude values
map_berlin = folium.Map(location=[latitude, longitude], zoom_start=10)
# add markers to map
for lat, lng, neighborhood in zip(df['Latitude'], df['Longitude'], df['Neighborhood']):
    label = '{}'.format(neighborhood)
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=label,
        color='blue',
        fill=True,
        fill_color='#3186cc',
        fill_opacity=0.7,
        parse_html=False).add_to(map_berlin)  
    
map_berlin


# In[53]:


map_berlin.save('map_berlin.html')


# In[54]:


#5. Use the Foursquare API to explore the neighborhoods


# In[55]:


# define Foursquare Credentials and Version
CLIENT_ID = '4TKLQIUN2TA0Y2BL5BAIHOQ2UR3Q4GSVAAE3GPPYC44JGH02' # your Foursquare ID
CLIENT_SECRET = 'COKXWSMNEIIYNP1FDFIAZDYBMNWTWU3MYCTJNBW4I2DUYJNJ' # your Foursquare Secret
VERSION = '20180605' # Foursquare API version

print('Your credentails:')
print('CLIENT_ID: ' + CLIENT_ID)
print('CLIENT_SECRET:' + CLIENT_SECRET)


# In[56]:


#现在，让我们获得2000米范围内的前100个场所。
radius = 2000
LIMIT = 200

venues = []

for lat, long, neighborhood in zip(df['Latitude'], df['Longitude'], df['Neighborhood']):
    
    # create the API request URL
    url = "https://api.foursquare.com/v2/venues/explore?client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}".format(
        CLIENT_ID,
        CLIENT_SECRET,
        VERSION,
        lat,
        long,
        radius, 
        LIMIT)
    
    # make the GET request
    results = requests.get(url).json()["response"]['groups'][0]['items']
    
    # return only relevant information for each nearby venue
    for venue in results:
        venues.append((
            neighborhood,
            lat, 
            long, 
            venue['venue']['name'], 
            venue['venue']['location']['lat'], 
            venue['venue']['location']['lng'],  
            venue['venue']['categories'][0]['name']))


# In[57]:




# convert the venues list into a new DataFrame
venues_df = pd.DataFrame(venues)

# define the column names
venues_df.columns = ['Neighborhood', 'Latitude', 'Longitude', 'VenueName', 'VenueLatitude', 'VenueLongitude', 'VenueCategory']

print(venues_df.shape)
venues_df.shape
venues_df.head(15)


# In[58]:


#提取VenueCategory列里包含“Restaurant”的行
venues_df_restaurant=venues_df[venues_df['VenueCategory'].str.contains('Restaurant')]
venues_df_restaurant.shape
venues_df_restaurant


# In[70]:


#reset index
venues_df_restaurant=venues_df_restaurant.reset_index(drop=True)
venues_df_restaurant


# In[71]:


#Let's find out how many unique categories can be curated from all the returned venues
#有多少餐厅种类在Venuecategory列
#extract the rows containing "Restaurant" in the VenueCatergory column.
print('There are {} uniques categories.'.format(len(venues_df_restaurant['VenueCategory'].unique())))


# In[72]:


#各个餐厅种类各有多少餐厅
#How many restaurants are there for each type of restaurant-catogety
print(venues_df_restaurant.loc[:,'VenueCategory'].value_counts()[0:10])


# In[79]:


#turn array to dataframe
venues_df_restaurant_neighborhood = pd.DataFrame(venues_df_restaurant.groupby(["Neighborhood"]).count())
venues_df_restaurant_neighborhood.sort_values(by="Latitude" , ascending=False) 


# In[80]:


#Cluster Neighborhoods


# In[81]:


# 每个neighborhood下的前五大场所
#Let’s analyze each neighborhood to know about the top 5 venues of each one.
#one hot encoding
bl_onehot = pd.get_dummies(venues_df_restaurant[['VenueCategory']], prefix="", prefix_sep="")

# add neighborhood column back to dataframe
bl_onehot['Neighborhoods'] = venues_df_restaurant['Neighborhood'] 

# move neighborhood column to the first column
fixed_columns = [bl_onehot.columns[-1]] + list(bl_onehot.columns[:-1])
bl_onehot = bl_onehot[fixed_columns]

print(bl_onehot.shape)
bl_onehot.head()


# In[82]:


#Use pandas groupby on neighborhood column and calculate the mean of the frequency of occurrence of each venue category.
#在邻域列上使用pandas groupby并计算每个场所类别出现频率的平均值。
bl_grouped=bl_onehot.groupby("Neighborhoods").mean().reset_index()
bl_grouped


# In[83]:


#Output each neighborhood along with the top 5 most common venues:
num_top_venues=5

for hood in bl_grouped['Neighborhoods']:
    print('---'+hood+'---')
    temp=bl_grouped[bl_grouped['Neighborhoods']==hood].T.reset_index()
    temp.columns=['venue','freq']
    temp=temp.iloc[1:]
    temp['freq']=temp['freq'].astype(float)
    temp=temp.round({'freq':2})
    print(temp.sort_values('freq', ascending=False).reset_index(drop=True).head(num_top_venues))
    print('\n')


# In[84]:


#7. Cluster Neighborhoods


# In[85]:


#Run k-means to cluster the neighborhoods in Kuala Lumpur into 3 clusters.

# set number of clusters
kclusters = 5

bl_clustering = bl_grouped.drop(["Neighborhoods"], 1)

# run k-means clustering
kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(bl_clustering)

# check cluster labels generated for each row in the dataframe
kmeans.labels_[0:10]


# In[86]:


# create a new dataframe that includes the cluster as well as the top 10 venues for each neighborhood.
bl_merged = bl_grouped.copy()

# add clustering labels
bl_merged["Cluster Labels"] = kmeans.labels_
#为何rename?
bl_merged.rename(columns={"Neighborhoods": "Neighborhood"}, inplace=True)
bl_merged.head()


# In[87]:


# merge toronto_grouped with toronto_data to add latitude/longitude for each neighborhood
bl_merged = bl_merged.join(df.set_index("Neighborhood"), on="Neighborhood")

print(bl_merged.shape)
bl_merged.head() # check the last columns!


# In[88]:


# sort the results by Cluster Labels
print(bl_merged.shape)
bl_merged.sort_values(["Cluster Labels"], inplace=True)
bl_merged


# In[ ]:


#Finally, let's visualize the resulting clusters


# In[89]:


# create map
map_clusters = folium.Map(location=[latitude, longitude], zoom_start=11)

# set color scheme for the clusters
x = np.arange(kclusters)
ys = [i+x+(i*x)**2 for i in range(kclusters)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]

# add markers to the map
markers_colors = []
for lat, lon, poi, cluster in zip(bl_merged['Latitude'], bl_merged['Longitude'], bl_merged['Neighborhood'], bl_merged['Cluster Labels']):
    label = folium.Popup(str(poi) + ' - Cluster ' + str(cluster), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color=rainbow[cluster-1],
        fill=True,
        fill_color=rainbow[cluster-1],
        fill_opacity=0.7).add_to(map_clusters)
       
map_clusters


# In[90]:


# save the map as HTML file
map_clusters.save('map_clusters_cap_final.html')


# In[91]:


#8. Examine Clusters


# In[92]:


#cluster 0
bl_merged.loc[bl_merged['Cluster Labels'] == 0]


# In[93]:


#Cluster 1
bl_merged.loc[bl_merged['Cluster Labels'] == 1]


# In[94]:


#Cluster 2
bl_merged.loc[bl_merged['Cluster Labels'] == 2]


# In[95]:


#Cluster 3
bl_merged.loc[bl_merged['Cluster Labels'] == 3]


# In[96]:


#Cluster 4
bl_merged.loc[bl_merged['Cluster Labels'] == 4]


# In[ ]:


#end

