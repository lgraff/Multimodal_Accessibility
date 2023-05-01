#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 17:37:04 2022

@author: lindsaygraff

Build config file
"""

import yaml

# Define parameters
config_info = {
    'Geography': {
        # 'neighborhoods' : ['Hazelwood', 'Glen Hazel', 'Greenfield', 'Squirrel Hill South', 'Squirrel Hill North',
        #                    'South Oakland', 'Central Oakland', 'North Oakland', 'Point Breeze'],
        'neighborhoods': ['Central Oakland','Squirrel Hill South', 'Squirrel Hill North',
                             'North Oakland', 'Larimer', 'Point Breeze', 'Shadyside','East Liberty'],
        'buffer': 0.2 # miles
        },
    'Beta_Params': {
        'b_price': 1,
        'b_TT': 10/3600,   # when b_tt and b_rel are 10/60, the route is scoot-zip 
        'b_risk': 0.1,
        'b_disc': 0,
        'b_rel': 10/3600
        },  # ultimately need b_rel_weight * b_TT
    'Speed_Params': {
        'walk': 1.3,  # m/s
        #'parking_speed': 5 / 3600 * 1609, # miles /hr / 3600 s/hr * 1609 meter/mile = m/s
        'scoot': 2.78, # m/s. see: A comparative analysis of e-scooter and e-bike usage patterns...(Almannnaa, Ashqar, ...)
        'bike': 14.5 / 3600 * 1000,    # Characterizing the speed and pahts of shared bicycle use in Lyon (Jensen et al) 2010
        'TNC': {'wait_time': 7}  # minutes
        },   # km/hr
    'Price_Params': {
        'w': {'ppmin': 0, 'ppmile':0, 'fixed': 0},  
        'sc': {'ppmin': 0.39, 'ppmile':0, 'fixed': 0},  # $
        'sc_tx': {'ppmin': 0, 'ppmile':0, 'fixed': 1},
        'bs': {'ppmin': 25/200, 'ppmile':0, 'fixed': 0},
        't': {'ppmin': 0.18, 'ppmile': 1.08, 'fixed': 0},  # fixed price is: base fare + "long pickup fare" + "booking fee"
        't_wait': {'ppmin':0, 'ppmile':0, 'fixed': 2.92 + 2.57 + 8.32/4},  #8.32/4 is like a minfare buffer
        'board': {'ppmin':0, 'ppmile':0, 'fixed': 2.75},
        'alight': {'ppmin':0, 'ppmile':0, 'fixed': 0},
        'pt': {'ppmin':0, 'ppmile':0, 'fixed': 0},
        'rt': {'ppmin':0, 'ppmile':0, 'fixed': 0},
        'pb': {'ppmin': 0, 'ppmile':0, 'fixed': 0},
        'z': {'ppmin': 11/60, 'ppmile':0, 'fixed': 0, 'fixed_per_month': 9, 'est_num_trips': 4},
        'pv': {'ppmin':0, 'ppmile': 0.20, 'fixed': 0},
        'park': {'ppmin':0, 'ppmile':0, 'fixed':5+(2*11)}  # can add a ppmin for parking and adjust travel time of parking edge to account for fixed time
        },
    'Conversion_Factors': {
        'meters_in_mile': 1609,
        'miles_in_km': 0.621371
        },
    'Time_Intervals': {
        'interval_spacing': 10,  # sec
        'len_period': 60*60,  # sec
        'time_start': 7,  # AM
        'time_end': 9  # AM
        },

    'Reliability_Params': {
        'board':2,  # cite Daryn's paper
        't_wait':2  # cite?
        },

    'Discomfort_Params': {
        'w': 2.86/1.34,
        'sc': 3.26/1.34,  # 
        'bs': 3.26/1.34,
        't': 1.34/1.34,  # use vehicle as baseline
        't_wait': 0,
        'board': 0,  # could change if thinking about cold weather conditions
        'alight': 0,
        'pt': 2.22/1.34,
        'pb': 3.26/1.34,
        'z': 1.34/1.34,
        'pv': 1.34/1.34,
        'park': 1.34/1.34
        },

    'Risk_Crash_Idx': {
        'w':0.28,
        'sc':1.81,
        'bs':1.81,
        't':1,
        't_wait':0,
        'board':0.19,
        'alight':0.19,
        'pt':0.19,
        'pb':1.81,
        'z':1,
        'pv':1,
        'park':1
    },

    'Connection_Edge_Speed': {
        'pv': 5 / 3600 * 1609, # miles /hr / 3600 s/hr * 1609 meter/mile = m/s
        'bs': 15 / 3600 * 1000,    # 15 km/hr / (3600 s/hr ) * 1000 m/km = m/s
	'zip': 5 / 3600 * 1609, # miles /hr / 3600 s/hr * 1609 meter/mile = m/s
        },
    'Scoot_Data_Generation': {
        'num_days_of_data': 30,
        'num_obs': 1500*(2/3) # see MovePGH report. 1500 total scooters, eyeball estimate that 2/3 are located in these nhoods
        },
    'Supernetwork': {
        'modes_included': ['bs', 'z', 'sc', 't', 'pt'],
        'W_tx': 0.5,  # miles,
        'W_od': 0.75,  # miles
	    # 'org': [-79.9488, 40.4161],  # hazelwood green
	    # 'dst': [-79.9194, 40.4517], # mellon park 
        'org': [-79.91101702988759, 40.4642694061984], # larimer
        'dst': [-79.95137478282426, 40.43878091188718],  # central oakland
        'num_park_hours': 2 
    # chatham univ: [-79.92399900719222, 40.44955761072877] 
        }  #, 'pb']}
    
    }

# Write the config file
with open("config.yaml", 'w') as yamlfile:
    data = yaml.dump(config_info, yamlfile)
    print("Write successful")

