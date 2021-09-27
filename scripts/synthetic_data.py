import pandas as pd
import os, sys
import pvlib
from pvlib.pvsystem import PVSystem
from pvlib.location import Location
from pvlib.modelchain import basic_chain, ModelChain
from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS
from data_processing import get_naive_times
from mcmc_functions import get_energy_day


# Check whether files already exist
def check_files(meta, data_path):
    nfiles = 0
    for system in meta:
        id = system[0]
        filename = 'pvo_mock_' + str(id) + '.csv'
        file_path = data_path + filename
        if os.path.exists(file_path):
            nfiles+=1
            print(filename, ' already exists. If you continue with the next function you will overwrite the previous version.')
        else:
            print(filename, ' does not exist yet.')
    
    if nfiles>0:
        reaction = input('one or more csv file(s) already exist(s). Select (Y/N) to continue: ')
    else:
        reaction = 'Y'
    return reaction


# Create synthetic 'observed' data
def create_observations(meta, lst_dates, data_path, time_resolution):
    # Get PVLib databases
    sandia_modules = pvlib.pvsystem.retrieve_sam('SandiaMod')
    sapm_inverters = pvlib.pvsystem.retrieve_sam('cecinverter')
        
    # Iterate through list of systems containing id number and system specifications.
    for system in meta:
        id = system[0]
        specs = system[1]
        df_E_tot = pd.DataFrame()
        # Iterate through list of dates for which to create observational data
        for single_date in lst_dates:
            l = specs[1]; b = specs[0]; theta = specs[2]; phi = specs[3]; Np = specs[4]
            year = single_date[0:4]; month = single_date[4:6]; day = single_date[6:8]
            times, ymd, hm, timezone = get_naive_times(single_date, time_resolution)
            # Function which calls PVLib functions to create synthetic profiles for a given set of parameters.
            energy = get_energy_day(b, l, theta, phi, Np, year, month, day, times, sandia_modules, sapm_inverters, timezone)
            df_E = pd.DataFrame(list(zip(ymd, hm, energy)), columns = ['date', 'time', 'energy'])
            # Retain non-zero values of profile.
            df_E = df_E[df_E['energy']>1e-5]
            df_E_tot = pd.concat([df_E_tot, df_E])
            
        df_E_tot.to_csv(data_path + 'pvo_mock_' + str(id) + '.csv', index = False)




