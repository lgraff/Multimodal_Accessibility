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
from config import config_data # import config_data

# get the study area
cwd = os.getcwd()
path = os.path.join(cwd, 'Data', 'Input_Data', "Neighborhoods_.shp")
pgh_nhoods = gpd.read_file(path)  # all neighborhoods
nhoods_keep = config_data['Geography']['neighborhoods']  # neighborhoods to keep

# dissolve the neighborhoods together 
pgh_nhoods_union = pgh_nhoods[pgh_nhoods['hood'].isin(nhoods_keep)].dissolve()    

# save to output file
filepath = os.path.join(cwd, 'Data', 'Output_Data', 'pgh_study_area.csv') 
#pgh_nhoods_union.to_pickle(filepath)   

# pgh_nhoods_union.to_file(filepath, driver='GeoJSON')    

# study_area_gdf = pgh_nhoods_union.copy()
# bbox_study_area = study_area_gdf['geometry'].bounds.T.to_dict()[0]  # bounding box of neighborhood polygon layer   

