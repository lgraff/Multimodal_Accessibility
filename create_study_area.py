#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 12:33:12 2022
@author: lindsaygraff

Create the study area from the selected neighborhood in the config file
"""

# import libraries
#import yaml
import geopandas as gpd
#import pandas as pd
import os
import config as conf

def create_study_area(neighborhood_fpath, output_fpath):
    # get the study area
    path = neighborhood_fpath
    nhoods = gpd.read_file(path)  # all neighborhoods
    nhoods_keep = conf.config_data['Geography']['neighborhoods']  # neighborhoods to keep
    
    # dissolve the neighborhoods together 
    nhoods_union = nhoods[nhoods['hood'].isin(nhoods_keep)].dissolve()    
    
    # add a buffer
    #study_area_gdf = gpd.read_file(os.path.join(os.path.join(os.getcwd(), 'Data', 'Output_Data'), 'study_area.csv'))
    
    #  buffer the study area by x miles
    x = conf.config_data['Geography']['buffer']  # miles
    nhoods_union = nhoods_union.to_crs(crs='epsg:32128').buffer(x*1609).to_crs('EPSG:4326')  # 1609 meters/mile

    # save to output file
    nhoods_union.to_file(output_fpath, driver='GeoJSON')


# call function
# create_study_area(os.path.join(os.getcwd(), 'Data','Input_Data', 'Neighborhoods', 'Neighborhoods_.shp'), 
#                   os.path.join(os.getcwd(), 'Data', 'Output_Data', 'study_area.csv'))
# study_area_gdf = pgh_nhoods_union.copy()
# bbox_study_area = study_area_gdf['geometry'].bounds.T.to_dict()[0]  # bounding box of neighborhood polygon layer   

