from .config import *

from . import access

import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point, Polygon
from shapely.wkt import loads
from pyproj import CRS, Transformer
import osmnx as ox
import matplotlib.pyplot as plt

"""These are the types of import we might expect in this file
import pandas
import bokeh
import seaborn
import matplotlib.pyplot as plt
import sklearn.decomposition as decomposition
import sklearn.feature_extraction"""

"""Place commands in this file to assess the data you have downloaded. How are missing values encoded, how are outliers encoded? What do columns represent, makes rure they are correctly labeled. How is the data indexed. Crete visualisation routines to assess the data (e.g. in bokeh). Ensure that date formats are correct and correctly timezoned."""

import osmnx as ox

def data():
    """Load the data from access and ensure missing values are correctly encoded as well as indices correct, column names informative, date and times correctly formatted. Return a structured data structure such as a data frame."""
    df = access.data()
    raise NotImplementedError

def query(data):
    """Request user input for some aspect of the data."""
    raise NotImplementedError

def view(data):
    """Provide a view of the data that allows the user to verify some aspect of its quality."""
    raise NotImplementedError

def labelled(data):
    """Provide a labelled set of data ready for supervised learning."""
    raise NotImplementedError

def pois_from_coordinates(latitude, longitude, tags, box_width=0.02, box_height=0.02):
    north = latitude + box_height/2
    south = latitude - box_width/2
    west = longitude - box_width/2
    east = longitude + box_width/2
    pois = ox.features_from_bbox((west, south, east, north), tags)
    return pois

def add_feature_geometry(df: pd.DataFrame):
    wgs84 = CRS.from_epsg(4326)
    bng = CRS.from_epsg(27700)
    transformer = Transformer.from_crs(wgs84, bng)

    df[['easting', 'northing']] = df.apply(
        lambda row: pd.Series(transformer.transform(row['latitude'], row['longitude'])),
        axis=1
    )

    geometry = df.apply(lambda row: Point(row["easting"], row["northing"]), axis=1)
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:27700")
    gdf = gdf.set_crs(bng, allow_override=True)
    return gdf

def get_feature_counts(feature_gdf:gpd.GeoDataFrame, census_gdf:gpd.GeoDataFrame):
    joined_gdf = gpd.sjoin(feature_gdf, census_gdf, how="inner", predicate="within")
    joined_df = pd.DataFrame(joined_gdf)
    counts = joined_df['0a21cd'].value_counts().reset_index()
    counts.columns = ['0a21cd', 'count']
    return counts

def feature_counts(response_df:pd.DataFrame, counts:pd.DataFrame):
    feature_df = pd.merge(response_df, counts, left_on='geography_code', right_on='0a21cd', how='left')
    new_feature_df = feature_df.drop(columns=feature_df.columns[1:4])
    new_feature_df['count'] = new_feature_df['count'].fillna(0).astype(int)
    return new_feature_df

def feature_vector(feature_df: pd.DataFrame, area:pd.DataFrame):
    feature_concat = pd.concat([feature_df, area], axis=1)
    feature_vector_df = feature_concat.apply(lambda row: row["count"] / row["area"], axis=1)
    feature_vector_df = pd.DataFrame(feature_vector_df, columns=["counts_per_area"])
    return feature_vector_df

def plot_nodes(lat, lon, gdf:gpd.GeoDataFrame, box_width=0.02, box_height=0.02):

    north = lat + box_height / 2
    south = lat - box_height / 2
    west = lon - box_width / 2
    east = lon + box_width / 2

    graph = ox.graph_from_bbox((west, south, east, north))
    nodes, edges = ox.graph_to_gdfs(graph)

    lats = np.array(gdf["latitude"]).astype(float)
    lons = np.array(gdf["longitude"]).astype(float)

    gdf_points = gpd.GeoDataFrame(geometry=gpd.points_from_xy(lons, lats))
    gdf_points.set_crs("EPSG:4326", inplace=True)

    area = area.to_crs("EPSG:4326")
    edges = edges.to_crs("EPSG:4326")

    fig, ax = plt.subplots(figsize=(10, 10))

    area.plot(ax=ax, facecolor="white", edgecolor="black", alpha=0.5)
    edges.plot(ax=ax, linewidth=1, edgecolor="dimgray")
    gdf_points.plot(ax=ax, color="blue", alpha=1, markersize=50)

    ax.set_xlim([west, east])
    ax.set_ylim([south, north])
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    plt.tight_layout()
    plt.show()
