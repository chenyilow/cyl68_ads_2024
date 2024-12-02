from .config import *

from . import access

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon
from shapely.wkt import loads
from pyproj import CRS, Transformer

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

def add_feature_geometry(geodf:gpd.GeoDataFrame):
    wgs84 = CRS.from_epsg(4326)
    bng = CRS.from_epsg(27700)
    transformer = Transformer.from_crs(wgs84, bng)

    geodf[['easting', 'northing']] = geodf.apply(
        lambda row: pd.Series(transformer.transform(row['latitude'], row['longitude'])),
        axis=1
    )
    geodf["geometry"] = geodf.apply(lambda row: Point(row["easting"], row["northing"]), axis=1)
    gpd.GeoDataFrame(geodf, geometry='geometry', crs="EPSG:4326")
    return geodf

def get_feature_counts(feature_gdf:gpd.GeoDataFrame, census_gdf:gpd.GeoDataFrame):
    joined_gdf = gpd.sjoin(feature_gdf, census_gdf, how="inner", predicate="within")
    joined_df = pd.DataFrame(joined_gdf)
    counts = joined_df['0a21cd'].value_counts().reset_index()
    counts.columns = ['0a21cd', 'count']
    return counts

def feature_vector(response_df:pd.DataFrame, counts:pd.DataFrame):
    feature_df = pd.merge(response_df, counts, left_on='geography_code', right_on='0a21cd', how='left')
    new_feature_df = feature_df.drop(columns=['0a21cd', 'prop_l15', 'db_id'])
    new_feature_df['count'] = new_feature_df['count'].fillna(0).astype(int)
    return new_feature_df