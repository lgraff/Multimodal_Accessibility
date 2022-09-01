#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 12:33:12 2022

@author: lindsaygraff
"""

# import libraries
import yaml
import geopandas as gpd
import pandas as pd
import os
#from util_functions import *

# load config file
with open("config.yaml", "r") as yamlfile:
    config_data = yaml.load(yamlfile, Loader=yaml.FullLoader)
    print("Read successful")

# get the study area
cwd = os.getcwd()
path = os.path.join(cwd, 'Data', 'Input_Data', "Neighborhoods_.shp")
pgh_nhoods = gpd.read_file(path)

nhoods_keep = config_data['Geography']['neighborhoods']

# dissolve the neighborhoods together 
pgh_nhoods_union = pgh_nhoods[pgh_nhoods['hood'].isin(nhoods_keep)].dissolve()    

# save to output file
filepath = os.path.join(cwd, 'Data', 'Output_Data', 'pgh_study_area.csv') 
#pgh_nhoods_union.to_pickle(filepath)   

pgh_nhoods_union.to_file(filepath, driver='GeoJSON')    
   

