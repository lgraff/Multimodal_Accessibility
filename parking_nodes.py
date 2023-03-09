#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 17:01:21 2022

@author: lindsaygraff

Get parking data and output as csv, which contains one "representative" point
for each zone

"""
#%%
# import libraries
import geopandas as gpd
import pandas as pd
import os
import matplotlib.pyplot as plt
import re

def create_parking_nodes(input_filepath, output_filepath):
    cwd = os.getcwd()
    df_park = pd.read_csv(input_filepath)
    # Remove rows that do not have both a lat and long populated
    df_park = df_park.loc[~((df_park['latitude'].isnull()) | (df_park['longitude'].isnull()))]
    df_park.loc[df_park.rate == 'Multi_Rate']
    # Remove rows that do not have a rate populated or has a "Multi-Rate"
    df_park = df_park.loc[(~(df_park['rate'].isnull()) & ~(df_park['rate'] == 'Multi-Rate'))]

    # extract rate and remove leading $ sign 
    def to_float_rate(string_rate):
        float_rate = float(re.split(r'[(|/]', string_rate)[0].lstrip('$'))
        return(float_rate)

    df_park['float_rate'] = df_park.apply(lambda row: to_float_rate(row['rate']), axis=1)  # hourly

    # For simplicity, choose just one "representative" parking point for each zone
    df_park_avg = df_park.groupby('zone').agg({'latitude':'mean', 'longitude':'mean', 'float_rate':'mean'}).reset_index()
    gdf_park_avg = gpd.GeoDataFrame(data=df_park_avg, geometry=gpd.points_from_xy(x=df_park_avg.longitude, y=df_park_avg.latitude),crs='epsg:4326')
    gdf_park_avg.plot()
    study_area_gdf = gpd.read_file(os.path.join(cwd, 'Data', 'Output_Data', 'study_area.csv'))
    gdf_park_avg_clip = gpd.clip(gdf_park_avg, study_area_gdf)

    gdf_park_avg_clip.to_file(output_filepath, driver='GeoJSON')

    fig, ax = plt.subplots()
    study_area_gdf.plot(ax=ax)
    gdf_park_avg.plot(ax=ax, color='red')

# call function
# cwd = os.getcwd()
# filepath = os.path.join(cwd, 'Data', 'Input_Data','ParkingMetersPaymentPoints.csv')
# create_parking_nodes(filepath)
# %%
