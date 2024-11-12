from .config import *
import requests
import pymysql
import csv
import time
import osmnx as ox
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd

"""These are the types of import we might expect in this file
import httplib2
import oauth2
import tables
import mongodb
import sqlite"""

# This file accesses the data

"""Place commands in this file to access the data electronically. Don't remove any missing values, or deal with outliers. Make sure you have legalities correct, both intellectual property and personal data privacy rights. Beyond the legal side also think about the ethical issues around this data. """

def data():
    """Read the data from the web or local file, returning structured format such as a data frame"""
    raise NotImplementedError

def hello_world():
    print("Hello from the data science library!")

def download_price_paid_data(year_from, year_to):
    # Base URL where the dataset is stored 
    base_url = "http://prod.publicdata.landregistry.gov.uk.s3-website-eu-west-1.amazonaws.com"
    """Download UK house price data for given year range"""
    # File name with placeholders
    file_name = "/pp-<year>-part<part>.csv"
    for year in range(year_from, (year_to+1)):
        print (f"Downloading data for year: {year}")
        for part in range(1,3):
            url = base_url + file_name.replace("<year>", str(year)).replace("<part>", str(part))
            response = requests.get(url)
            if response.status_code == 200:
                with open("." + file_name.replace("<year>", str(year)).replace("<part>", str(part)), "wb") as file:
                    file.write(response.content)

def create_connection(user, password, host, database, port=3306):
    """ Create a database connection to the MariaDB database
        specified by the host url and database name.
    :param user: username
    :param password: password
    :param host: host url
    :param database: database name
    :param port: port number
    :return: Connection object or None
    """
    conn = None
    try:
        conn = pymysql.connect(user=user,
                               passwd=password,
                               host=host,
                               port=port,
                               local_infile=1,
                               db=database
                               )
        print(f"Connection established!")
    except Exception as e:
        print(f"Error connecting to the MariaDB Server: {e}")
    return conn

def housing_upload_join_data(conn, year):
    start_date = str(year) + "-01-01"
    end_date = str(year) + "-12-31"

    cur = conn.cursor()
    print('Selecting data for year: ' + str(year))
    cur.execute(f'SELECT pp.price, pp.date_of_transfer, po.postcode, pp.property_type, pp.new_build_flag, pp.tenure_type, pp.locality, pp.town_city, pp.district, pp.county, po.country, po.latitude, po.longitude FROM (SELECT price, date_of_transfer, postcode, property_type, new_build_flag, tenure_type, locality, town_city, district, county FROM pp_data WHERE date_of_transfer BETWEEN "' + start_date + '" AND "' + end_date + '") AS pp INNER JOIN postcode_data AS po ON pp.postcode = po.postcode')
    rows = cur.fetchall()

    csv_file_path = 'output_file.csv'

    # Write the rows to the CSV file
    with open(csv_file_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        # Write the data rows
        csv_writer.writerows(rows)
    print('Storing data for year: ' + str(year))
    cur.execute(f"LOAD DATA LOCAL INFILE '" + csv_file_path + "' INTO TABLE `prices_coordinates_data` FIELDS TERMINATED BY ',' OPTIONALLY ENCLOSED by '\"' LINES STARTING BY '' TERMINATED BY '\n';")
    conn.commit()
    print('Data stored for year: ' + str(year))

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