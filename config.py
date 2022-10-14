#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 15:33:48 2022
@author: lindsaygraff

Define global variables by reading config file and study area
"""

# import libraries
import os
import yaml
import geopandas as gpd

# load config file
def load_config(config_filename):
    with open(config_filename, "r") as yamlfile:
        config_data = yaml.load(yamlfile, Loader=yaml.FullLoader)
        print("Read successful")
    return config_data

config_data = load_config('config.yaml')

# read study area file
#study_area_gdf = gpd.read_file(os.path.join(os.path.join(os.getcwd(), 'Data', 'Output_Data'), 'study_area.csv'))
#bbox_study_area = study_area_gdf['geometry'].bounds.T.to_dict()[0]  # bounding box of neighborhood polygon layer   