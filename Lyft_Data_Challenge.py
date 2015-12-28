import pandas as pd
import scipy as sp
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(rc={'axes.facecolor':'white','figure.facecolor':'white'})

# These are the "Tableau 20" colors as RGB.    
tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),    
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),    
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),    
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),    
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]    
  
# Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.    
for i in range(len(tableau20)):    
    r, g, b = tableau20[i]    
    tableau20[i] = (r / 255., g / 255., b / 255.)    

"""
Some exploratory analysis
"""

rides = pd.read_csv('rides.csv')
rides.head()
rides.describe()

print rides.dtypes
print len(rides)


#pickup = pd.DatetimeIndex(rides['pickup_datetime'])
#dropoff = pd.DatetimeIndex(rides['dropoff_datetime'])
#rides['pickup'] = pickup
#rides['dropoff'] = dropoff
#rides.dtypes
#print min(rides['pickup']), max(rides['pickup'])

plt.hist(rides['distance_miles'])
plt.show()


#### Most of the rides are low distance (even below 1 mile). We keep the distances that are 0. As we only have granularity to the scale of the mile we can imagine ride that are below that threshold

# Some rides have negative duration time which is weird
print 1.*len(rides[rides['duration_secs']<0])/len(rides)

# However there is only a tiny fraction of the data so we can remove these points

# What is the minimum time in seconds if we exclude rides that have negative duration?
clean = rides[rides['duration_secs']>0]
print min(clean['duration_secs'])

# 1s! It is way to low to be real. 
#### In the following we will remove all rides that are less than 2 minutes

rides2 = rides[rides['duration_secs']>=120]

plt.hist(rides2['duration_secs'])
plt.show()

print len(rides2)

"""
We want the routes that maximize the total number of expected passenger rides.
One approach is to identify the location of traveling spots as most of the passengers are expected to travel between them.
"""
# To get a sense of the hot spots we restrict the dataset to one day for plotting purposes

one_day = rides2[0:1000]

print len(one_day)
print np.median(one_day['start_lat']), np.median(one_day['start_lng'])

#import folium

# Create map of the different pickup locations
#map = folium.Map(location=[40.749954,-73.983711],
#    zoom_start=5)

#for _, df in one_day.iterrows():
#    map.circle_marker(
#        location=[df['start_lat'], df['start_lng']],
#        radius=20,
#    )

#map.create_map(path='pickups.html')


#### The data are located in New York. Let's overplot some well-known destinations to get a sense of the hot spots

# Times Square
times_square = [40.75773, -73.985708]
# Financial District
financial_district = [40.707499, -74.011153]
# LaGuardia
laguardia = [40.77725, -73.872611]
# JFK
jfk = [40.639722, -73.778889]
# Central Park
central_park = [40.783333, -73.966667]


# Plot the pick ups

plt.plot(one_day['start_lng'], one_day['start_lat'], '.', color='k', alpha=0.8)
plt.plot(times_square[1], times_square[0], 'o', color=tableau20[0])
plt.plot(financial_district[1], financial_district[0], 'o', color=tableau20[1])
plt.plot(laguardia[1], laguardia[0], 'o', color=tableau20[2])
plt.plot(jfk[1], jfk[0], 'o', color=tableau20[3])
plt.plot(central_park[1], central_park[0], 'o', color=tableau20[4])
plt.xlim(-74.1, -73.6)
plt.ylim(40.55, 40.9)
plt.legend(['Rides', 'Times Square', 'Financial District', 'LaGuardia', 'JFK', 'Central Park'], ncol=1, frameon=False, fontsize=16) 
plt.show()

#### The plot shows that Times Square and the Financial District are popular pickup spots in Manhattan. LaGuardia Airport and John F. Kennedy International Airport as well.

# Plot the drop offs

plt.plot(one_day['end_lng'], one_day['end_lat'], '.', color='k', alpha=0.8)
plt.plot(times_square[1], times_square[0], 'o', color=tableau20[0])
plt.plot(financial_district[1], financial_district[0], 'o', color=tableau20[1])
plt.plot(laguardia[1], laguardia[0], 'o', color=tableau20[2])
plt.plot(jfk[1], jfk[0], 'o', color=tableau20[3])
plt.plot(central_park[1], central_park[0], 'o', color=tableau20[4])
plt.xlim(-74.1, -73.6)
plt.ylim(40.55, 40.9)
plt.legend(['Rides', 'Times Square', 'Financial District', 'LaGuardia', 'JFK', 'Central Park'], ncol=1, frameon=False, fontsize=16) 
plt.show()

#### We see that the dropoffs can happen anywhere and spread wider than the pickups

#### We want to find the hot spots by clustering the pick ups spatially

from sklearn.cluster import KMeans

# Let's look at an even smaller sample to optimize our algoritm
one_day = rides2[0:1000]

pickup_data = one_day[['start_lat','start_lng']]
pickup_data.head()

print min(pickup_data['start_lat']), max(pickup_data['start_lat']), min(pickup_data['start_lng']), max(pickup_data['start_lng'])

xpickup_data = np.array(pickup_data)

#### We want to know how many clusters do we need

k_range = range(1, 10)
k_means_var = [KMeans(n_clusters=k).fit(xpickup_data) for k in k_range]

# Find the cluster center for each model
centroids = [X.cluster_centers_ for X in k_means_var]

from scipy.spatial.distance import cdist, pdist

# Calculate the Euclidian distance for each point to the center
k_euclid = [cdist(xpickup_data, cent, 'euclidean') for cent in centroids]

dist = [np.min(ke, axis=1) for ke in k_euclid]

# Total within-cluster sum of squares
wcss = [sum(d**2) for d in dist]

# Total sum of squares
tss = sum(pdist(xpickup_data)**2)/xpickup_data.shape[0]

plt.plot(k_range, wcss)
plt.xlabel("Number of clusters")
plt.ylabel("Within cluster variance")
plt.show()

# We want to find the elbow that minimize the variance within clusters
# We see that the optimal number of cluster is 5
# We now consider a bigger portion of the data

slice = rides2[0:500000]
pickup_data2 = slice[['start_lat','start_lng']]
pickup_data2.head()

xpickup_data2 = np.array(pickup_data2)
k_result = KMeans(n_clusters=5).fit(xpickup_data2)

pickup_data2['labels'] = k_result.labels_

plt.plot(pickup_data2['start_lng'][pickup_data2['labels']==0], pickup_data2['start_lat'][pickup_data2['labels']==0], '.', color='b')
plt.plot(pickup_data2['start_lng'][pickup_data2['labels']==1], pickup_data2['start_lat'][pickup_data2['labels']==1], '.', color='r')
plt.plot(pickup_data2['start_lng'][pickup_data2['labels']==2], pickup_data2['start_lat'][pickup_data2['labels']==2], '.', color='g')
plt.plot(pickup_data2['start_lng'][pickup_data2['labels']==3], pickup_data2['start_lat'][pickup_data2['labels']==3], '.', color='m')
plt.plot(pickup_data2['start_lng'][pickup_data2['labels']==4], pickup_data2['start_lat'][pickup_data2['labels']==4], '.', color='y')
plt.xlim(-74.1, -73.6)
plt.ylim(40.55, 40.9)
plt.show()

### Now we want to apply the same method to the drop off locations
one_day = rides2[0:1000]

dropoff_data = one_day[['end_lat','end_lng']]
xdropoff_data = np.array(dropoff_data)

k_means_var2 = [KMeans(n_clusters=k).fit(xdropoff_data) for k in k_range]

# Find the cluster center for each model
centroids2 = [X.cluster_centers_ for X in k_means_var2]

# Calculate the Euclidian distance for each point to the center
k_euclid2 = [cdist(xdropoff_data, cent, 'euclidean') for cent in centroids2]

dist2 = [np.min(ke, axis=1) for ke in k_euclid2]

# Total within-cluster sum of squares
wcss2 = [sum(d**2) for d in dist2]

#### We need 6 clusters for the dropoff locations

slice = rides2[0:500000]
dropoff_data2 = slice[['end_lat','end_lng']]

xdropoff_data2 = np.array(dropoff_data2)
k_result_drop = KMeans(n_clusters=6).fit(xdropoff_data2)

dropoff_data2['labels'] = k_result_drop.labels_

plt.plot(dropoff_data2['end_lng'][dropoff_data2['labels']==0], dropoff_data2['end_lat'][dropoff_data2['labels']==0], '.', color='b')
plt.plot(dropoff_data2['end_lng'][dropoff_data2['labels']==1], dropoff_data2['end_lat'][dropoff_data2['labels']==1], '.', color='r')
plt.plot(dropoff_data2['end_lng'][dropoff_data2['labels']==2], dropoff_data2['end_lat'][dropoff_data2['labels']==2], '.', color='g')
plt.plot(dropoff_data2['end_lng'][dropoff_data2['labels']==3], dropoff_data2['end_lat'][dropoff_data2['labels']==3], '.', color='m')
plt.plot(dropoff_data2['end_lng'][dropoff_data2['labels']==4], dropoff_data2['end_lat'][dropoff_data2['labels']==4], '.', color='y')
plt.plot(dropoff_data2['end_lng'][dropoff_data2['labels']==5], dropoff_data2['end_lat'][dropoff_data2['labels']==5], '.', color='k')
plt.xlim(-74.1, -73.6)
plt.ylim(40.55, 40.9)
plt.show()

#### Let's now consider all the data

all_picks = rides2[['start_lat','start_lng']]
xall_picks = np.array(all_picks)
k_picks = KMeans(n_clusters=5).fit(xall_picks)

all_drops = rides2[['end_lat','end_lng']]
xall_drops = np.array(all_drops)
k_drops = KMeans(n_clusters=6).fit(xall_drops)

new_data = rides2
new_data['pick_labels'] = k_picks.labels_
new_data['drop_labels'] = k_drops.labels_

center_pickups = k_picks.cluster_centers_
center_dropoffs = k_drops.cluster_centers_

#### We define the Manhattan distance

def manhattan_distance(x1, x2, y1, y2):
    alpha = 84.2
    beta = 111.2
    dis = alpha*np.abs(x1-x2) + beta*np.abs(y1-y2)
    return dis



#### By calculating the route probability using the true pickup/dropoff location as the cluster's center, we extract the most probable routes

final = {}
for nx in range(5):
    for ny in range(6):
        sub = new_data[(new_data['pick_labels']==nx) & (new_data['drop_labels']==ny)]
        Di = manhattan_distance(sub['start_lat'], center_pickups[nx][0], sub['start_lng'], center_pickups[nx][1])
        Ai = manhattan_distance(sub['end_lat'], center_dropoffs[ny][0], sub['end_lng'], center_dropoffs[ny][1])
        prob = np.exp(-(Di+Ai))
        data = (float(sub['start_lat'][prob==max(prob)]), float(sub['start_lng'][prob==max(prob)]), float(sub['end_lat'][prob==max(prob)]), float(sub['end_lng'][prob==max(prob)]))
        final[data] = max(prob)

myresult = sorted(final.items(), key=itemgetter(1))[-5:]
final_pick_lat = [myresult[i][0][0] for i in range(5)]
final_pick_lng = [myresult[i][0][1] for i in range(5)]
final_drop_lat = [myresult[i][0][2] for i in range(5)]
final_drop_lng = [myresult[i][0][3] for i in range(5)]

output = pd.DataFrame(final_pick_lng)
output.columns = ['start_lng']
output['start_lat'] = final_pick_lat
output['end_lng'] = final_drop_lng
output['end_lat'] = final_drop_lat

output.to_csv('hot_routes.csv')

"""
The 5th hottest route corresponds to a ride from Soho to Central Park
The 4th hottest route corresponds to a ride from the MET to a point close to Jacqueline Kennedy Onassis Reservoir
(in Central Park near to the Jewish museum)
The 3rd hottest route corresponds to a ride around New York University
The 2nd hottest route corresponds to a ride from Times Square to New York University
The hottest route corresponds to a ride from Soho to the Rockfeller Center
"""
