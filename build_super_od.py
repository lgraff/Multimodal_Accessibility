# compile supernetwork with od-connectors
import os
from od_connector import od_cnx
import config as conf

cwd = os.getcwd()
od_cnx(os.path.join(cwd, 'Data', 'Output_Data', 'G_super.pkl'),
       os.path.join(cwd, 'Data', 'Output_Data', 'G_super_od.pkl'),
       conf.config_data['Supernetwork']['org'],
       conf.config_data['Supernetwork']['dst'])

       
