# libraries
import os
import time
from create_study_area import create_study_area
from download_crash_data import download_crash_data
from parking_nodes import create_parking_nodes
from process_street_centerlines import process_street_centerlines

cwd = os.getcwd()
# create the study area as a GeoJSON file
create_study_area(os.path.join(os.getcwd(), 'Data','Input_Data', 'Neighborhoods', 'Neighborhoods_.shp'), 
                  os.path.join(os.getcwd(), 'Data', 'Output_Data', 'study_area.csv'))
print('study area created')

# download crash data and store in ~/InputData folder
site = "https://data.wprdc.org"
resource_ids = ["514ae074-f42e-4bfb-8869-8d8c461dd824","cb0a4d8b-2893-4d20-ad1c-47d5fdb7e8d5"]
crash_output_path = os.path.join(cwd, 'Data', 'Input_Data','df_crash.csv')
check_file = os.path.isfile(crash_output_path)
# only download the data if doesn't already exist
if not check_file: 
    download_crash_data(site, resource_ids, crash_output_path)  
    print('crash data downloaded')
else:
    print('crash data already downloaded')
    
# create parking nodes as a GeoJSON file
input_filepath = os.path.join(cwd, 'Data', 'Input_Data','ParkingMetersPaymentPoints.csv')
output_filepath = os.path.join(cwd, 'Data', 'Output_Data', 'parking_points.csv')
create_parking_nodes(input_filepath, output_filepath)
print('parking nodes created')

# process street centerlines: save G_drive and G_bike as pickled objects at their respective output_path locations
streets_shapefile_path = os.path.join(cwd, 'Data', 'Input_Data', 'AlleghenyCounty_StreetCenterlines202208', 
                                     'AlleghenyCounty_StreetCenterlines202208.shp')
bikemap_folder = os.path.join(cwd, 'Data', 'Input_Data', 'bike-map-2019')
studyarea_filepath = os.path.join(cwd, 'Data', 'Output_Data', 'study_area.csv')
G_drive_output_path = os.path.join(cwd, 'Data', 'Output_Data', 'G_drive.pkl')
G_bike_output_path = os.path.join(cwd, 'Data', 'Output_Data', 'G_bike.pkl')

process_street_centerlines(studyarea_filepath, streets_shapefile_path, crash_output_path, bikemap_folder,
                                G_drive_output_path, G_bike_output_path)
print('street centerlines processed')

# build supernetwork: save as pickled object for later use if necessary (avoid compiling it many times)
from build_supernetwork import build_supernetwork  # keep this line here, do not move to the top of script
start = time.perf_counter() # records start time
G_super = build_supernetwork(os.path.join(cwd, 'Data', 'Output_Data', 'G_super.pkl')) 
end = time.perf_counter()  # record end time
sec_elapsed = (end-start) # find elapsed time in seconds
print(f"Elapsed {sec_elapsed:.03f} secs.")