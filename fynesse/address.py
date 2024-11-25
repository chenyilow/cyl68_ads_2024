# This file contains code for suporting addressing questions in the data

"""# Here are some of the imports we might expect 
import sklearn.model_selection  as ms
import sklearn.linear_model as lm
import sklearn.svm as svm
import sklearn.naive_bayes as naive_bayes
import sklearn.tree as tree

import GPy
import torch
import tensorflow as tf

# Or if it's a statistical analysis
import scipy.stats"""

"""Address a particular question that arises from the data"""

from sklearn.cluster import KMeans
import osmnx as ox

def kmeans_clustering(locations_dict, tags, clusters=3):
    data = {feature:[] for feature in tags}
    data['City'] = []
    for city in locations_dict:
        poi_counts = count_pois_near_coordinates(locations_dict[city][0], locations_dict[city][1], tags)
        data['City'].append(city)
        for feature in tags:
            data[feature].append(poi_counts[feature])

    pd_data = pd.DataFrame(data)
    pd_data = pd_data[['City'] + [col for col in pd_data.columns if col != 'City']]

    features = pd_data[["amenity", "buildings", "historic", "leisure", "shop", "tourism", "religion", "memorial"]]
    kmeans = KMeans(n_clusters=clusters).fit(features)

    cluster_df = pd.DataFrame({"City": pd_data["City"], "Cluster": kmeans.labels_})
    return cluster_df

def map(latitude, longitude):
    kilometer = 2/111
    north = latitude + kilometer/2
    south = latitude - kilometer/2
    west = longitude - kilometer/2
    east = longitude + kilometer # I have modified this slightly so that the picture looks more squarish later
    date_threshold = '2019-12-31'

    tags_buildings = {"building": True}
    pois_buildings = ox.geometries_from_bbox(north, south, east, west, tags_buildings)
    full_address_pois = pois_buildings[pois_buildings[["addr:housenumber", "addr:street", "addr:postcode"]].notna().all(axis=1)]
    no_address_pois = pois_buildings[pois_buildings[["addr:housenumber", "addr:street", "addr:postcode"]].isna().any(axis=1)]

    graph = ox.graph_from_bbox(north, south, east, west)

    # Retrieve nodes and edges
    nodes, edges = ox.graph_to_gdfs(graph)

    # Get place boundary related to the place name as a geodataframe
    area = ox.geocode_to_gdf("Cambridge")

    fig, ax = plt.subplots()

    # Plot the footprint
    area.plot(ax=ax, facecolor="white")

    # Plot street edges
    edges.plot(ax=ax, linewidth=1, edgecolor="dimgray")

    ax.set_xlim([west, east])
    ax.set_ylim([south, north])
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")

    # Plot all POIs
    full_address_pois.plot(ax=ax, color="blue", alpha=0.7, markersize=10)
    no_address_pois.plot(ax=ax, color="green", alpha=0.7, markersize=10)
    plt.tight_layout()
    plt.show()

def count_pois_near_coordinates(latitude: float, longitude: float, tags: dict, distance_km: float = 1.0) -> dict:
    """
    Count Points of Interest (POIs) near a given pair of coordinates within a specified distance.
    Args:
        latitude (float): Latitude of the location.
        longitude (float): Longitude of the location.
        tags (dict): A dictionary of OSM tags to filter the POIs (e.g., {'amenity': True, 'tourism': True}).
        distance_km (float): The distance around the location in kilometers. Default is 1 km.
    Returns:
        dict: A dictionary where keys are the OSM tags and values are the counts of POIs for each tag.
    """
    poi_counts = {}
    df = ox.features_from_point(center_point=(latitude,longitude), tags=tags, dist=distance_km*1000) # returns all pois in a circle with radius = distance_km in metres centered at center_point
    for feature in tags:
        if feature in df.columns:
            if tags[feature] == True:
                poi_counts[feature] = df[feature].notnull().sum()
            elif type(tags[feature]) == list:
                poi_counts[feature] = df[feature].isin(tags[feature]).sum()
        else:
            poi_counts[feature] = 0

    return poi_counts