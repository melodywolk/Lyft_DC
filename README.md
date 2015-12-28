# Lyft Data Challenge

## Goal: find a set of 5 hot routes.

### Exploratory analysis
A first look at the data showed that some rides have a duration less than zero. As it was only a tiny fraction of the entire dataset, I decided to remove these points.

### Find the hot spots
Hot routes maximize the number of expected passenger rides. One reasonable approach is to find what are "the most expected rides".
To do so, we can try to identify the location of big traveling spots. Passengers are expected to travel mainly between them and as a driver it optimizes the chance to find passengers thus to make a lot of rides. <br />


I chose to cluster the spatial coordinates of expected pickups and expected dropoffs and aggregate them into respectively 5 and 6 groups.
The number of clusters was chosen by minimizing the within-cluster variance. <br />


The pickups clusters exhibit clear locations: one corresponds to JFK, another to LaGuardia, one to the Central Park region, one to center of Manhattan (Soho, East/West Village) and one to the Financial District (partially merged with Brooklyn).
It is clear that pickups are mostly clustered in Manhattan and around the airports. <br />


The dropoff clusters also display clear patterns such as the ones mentioned above as well as Brooklyn, the Bronx and the Queens.
It is interesting to notice that the dropoffs spread further than the pickups.

### Find the hot routes
In this approach the probability of a route only depends on the walking distance from the expected pickup/dropoff location to the real one. This implies that the probability of the route only depends on what happens within each clusters and not on the actual route between two distinct clusters. It makes sense considering that for a neighbor of departures and a neighbor of arrivals the route between the two destinations is very likely to be the same. <br />


Thus to identify the most probable route, the probability based on the Manhattan distance between each point of the cluster and the center is computed. This assumes that the actual pickup/dropoff location are the center of each cluster. It seems a reasonable assumption both on the passenger and on the driver sides (it is natural to go the most central part of a location to find more cabs/clients).<br />
Finally the 5 hottest routes are outputed.

### Other things to investigate
We have an information on the time in the dataset. It is likely that hot spots in the morning are different from hot spots in the evening. In the first case we can imagine people commuting to work while in the evening people go out.
Slicing the above method with a time window could lead to an interesting picture of the hot routes as a function of time.  <br />

Using the Python package geopy, I noticed that the quantity 'distance_miles' is systematically larger that the computed distance using latitude and longitude for both the Vincenty and the great-circle distances. 
This could be due to the fact that this quantity is the effective distance (opposed to the "desired distance") and thus gives an estimate of the true pickup/dropoff location. It can be used to improve our measure of the probability associated with each route by replacing the use of the center of the cluster.

