import pandas as pd
import datetime
import pytz, os, glob

# Retrieve the meta data of system in question. Lon/lat to be more specific.
def retrieve_meta(sid, data_path):
    meta = pd.read_csv(data_path + 'meta.csv')
    meta_system = meta[meta['sid']==sid]
    lat = meta_system['lat'].values[0]; lon = meta_system['lon'].values[0]
    
    return lat, lon

# Retrieve the measurement data of the observed profiles. Return a dictionary of 
# time (in hours and minutes) and the instantaneous energy    
def retrieve_measurements(sid, date, data_path, option):
    if (option == 'synthetic'):
        measurement = pd.read_csv(data_path + 'pvo_mock_' + str(sid) + '.csv')
    elif (option == 'real'):
        measurement = pd.read_csv(data_path + 'pvo_' + str(sid) + '.csv')
    measurement = measurement[measurement['date'].astype(str) == date]
    measurement = measurement[['time', 'energy']]
    measurement = measurement[measurement['energy']>0.0]
    hm = measurement['time'].tolist(); E_obs = measurement['energy'].tolist()
    measurement = list(zip(hm, E_obs)); dict_measurement = dict(measurement)
        
    return dict_measurement

# Check whether the date in question is in Daylight Saving Time or not. This is necessary for 
# get_naive_times function below.
def is_dst(dt, timeZone):
   aware_dt = timeZone.localize(dt)
   
   return aware_dt.dst() != datetime.timedelta(0,0)

# Take into account GMT +1 or GMT +2 (in the case of the Netherlands) so that naive Sun times may be obtained.
def get_naive_times(date, time_resolution):
    year = int(date[0:4]); month = int(date[4:6]); day = int(date[6:8])
    d0 = datetime.datetime(year, month, day)
    d1 = d0 + datetime.timedelta(days=1)
    time_resolution = time_resolution + 'min'
    naive_times = pd.date_range(start = d0, end = d1, freq = time_resolution)
    
    tz_tmp = pytz.timezone("Europe/London")
    dst = is_dst(d0, tz_tmp)
    if (dst == True):
        timezone = 'Etc/GMT-' + str(2)
    if (dst == False):
        timezone = 'Etc/GMT-' + str(1)
        
    times = naive_times.tz_localize(timezone)
    ymd = [date.strftime("%Y%m%d") for date in naive_times]
    hm = [date.strftime("%H:%M") for date in naive_times]	
    
    return times, ymd, hm, timezone

# Check whether a list of directories exists. If they do not exist, they are created.
def create_directories(paths):
    for path in paths:
        if not os.path.exists(path):
            os.mkdir(path)
            print('creating directory ', path)
        else:
            print(path, ' already exists. Ignoring directory creation!')

# Check whether chain h5 files already exist. If so, they are deleted.
def overwrite_files(chains_path, sid, date, nwalkers_initial, iterations):
    filename = chains_path + 'chain_id%s_%s_w%s_i%s.h5' %(str(sid), date, str(nwalkers_initial), str(iterations))
    existing_files = sorted(glob.glob(filename[:-3] + '*.h5'))
    if (len(existing_files)>0):
        for existing_file in existing_files:
            os.remove(existing_file)
    try:
        filename_sample = chains_path + 'chain_id%s_%s_w%s_i%s_sample.h5' %(str(sid), date, str(8), str(iterations))
        os.remove(filename_sample)
    except:
        print('Sample file is not present')
