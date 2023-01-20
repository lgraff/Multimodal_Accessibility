
import os
from create_study_area import create_study_area
from download_crash_data import download_crash_data
from parking_nodes import create_parking_nodes
#from process_street_centerlines import process_street_centerlines

cwd = os.getcwd()
# create the study area
create_study_area(os.path.join(os.getcwd(), 'Data','Input_Data', 'Neighborhoods', 'Neighborhoods_.shp'), 
                  os.path.join(os.getcwd(), 'Data', 'Output_Data', 'study_area.csv'))
# download crash data
site = "https://data.wprdc.org"
resource_ids = ["514ae074-f42e-4bfb-8869-8d8c461dd824","cb0a4d8b-2893-4d20-ad1c-47d5fdb7e8d5"]
#download_crash_data(site, resource_ids)
# create parking nodes
create_parking_nodes(os.path.join(cwd, 'Data', 'Input_Data','ParkingMetersPaymentPoints.csv'))

# process street centerlines
# TODO: make process_street_centerlines.py into a callable function

# build supernetwork