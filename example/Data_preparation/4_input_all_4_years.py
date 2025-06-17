import pytz  # Import the pytz library for timezone handling
import ephem
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
import pysolar.solar as solar


k = 0

start_time = '20210301'
end_time   = '20241022'
cycles     = ['00', '06', '12', '18']
timezone = pytz.timezone('America/New_York') 


Observation_dir = '/lustre/code/BiasCorrection/input/Observation'
STOFS_dir = '/lustre/code/BiasCorrection/input/STOFS_Data'
forcing_dir = '/lustre/code/BiasCorrection/input/Data'
stations     = pd.read_csv('/lustre/code/BiasCorrection/input/stations.csv')


start_date = datetime.strptime(start_time, '%Y%m%d')
end_date = datetime.strptime(end_time, '%Y%m%d') # to include all nowcast data for the specified rage

dates = []
current_date = start_date
while current_date <= end_date:
    dates.append(current_date.strftime('%Y%m%d'))  # Format as YYYY-MM-DD
    current_date += timedelta(days=1)



# Calculate solar and moon Zenith angle 


def compute_zenith_angle(latitude, longitude, date_time):
    # Ensure the provided datetime object is timezone-aware
    if date_time.tzinfo is None:
        raise ValueError("Datetime object must be timezone-aware. Please provide timezone information.")

    # Compute zenith angle for a specific time, date, and location
    zenith_angle = solar.get_altitude(latitude, longitude, date_time)
    return zenith_angle


def compute_moon_zenith_angle(latitude, longitude, date_time):
    # Create an observer object with the provided latitude and longitude
    observer = ephem.Observer()
    observer.lat = str(latitude)
    observer.lon = str(longitude)

    # Set the observer's date and time
    observer.date = date_time

    # Compute the moon's position
    moon = ephem.Moon(observer)

    # Calculate the moon's altitude (zenith angle)
    moon_altitude = float(moon.alt) * 180 / ephem.pi  # Convert radians to degrees
    return moon_altitude



x = []
y = []
all_time = []

for date in dates:
    print(date)
    for cycle in cycles:

        if cycle == '00':
            current_date = datetime.strptime(date, '%Y%m%d') 
            current_date -= timedelta(days=1) # Go back one day 
            date_one_day_back = current_date.strftime('%Y%m%d')
            previous_cycle = '18'          
            time = datetime.strptime(date_one_day_back, '%Y%m%d') + timedelta(hours=int(previous_cycle)+1) 
        else:
                                               
            previous_cycle = cycles[cycles.index(cycle) - 1]                        
            time = datetime.strptime(date, '%Y%m%d') + timedelta(hours=int(previous_cycle)+1)
            
        n_stations = len(stations.nos_id)  
        time_steps = 186  

        obs_values = np.full((n_stations, time_steps), np.nan)
        obs_time = []
        
        for ns, nos_id in enumerate(stations.nos_id):
            obs = pd.read_csv(f'{Observation_dir}/{nos_id}.csv')
            obs['time'] = pd.to_datetime(obs['time']).dt.tz_localize(None)
            index_time = obs[obs['time'] == pd.to_datetime(time).strftime('%Y-%m-%d %H:%M:%S')].index
            if len(index_time) > 0:
                obs_values[ns,:] = obs[index_time[0]:index_time[0]+186]['value']
                obs_times = obs[index_time[0]:index_time[0]+186]['time']

            else:
                continue  
        
            
        current_x = np.full((n_stations, 8, time_steps), np.nan)  # This will hold values for all stations at this k
        current_y = np.full((n_stations, time_steps), np.nan)


        if cycle != '00' and date!= '20210825':
           Forecast = pd.read_csv(f'{STOFS_dir}/STOFS_2D_{date}_{cycle}.csv')
           v_wind = pd.read_csv(f'{forcing_dir}/v_wind_{date}_{cycle}.csv')
           u_wind = pd.read_csv(f'{forcing_dir}/u_wind_{date}_{cycle}.csv')
           surface_pressure = pd.read_csv(f'{forcing_dir}/surface_pressure_{date}_{cycle}.csv')
            
           for n , nos_id in enumerate(stations.nos_id):
 
                
                   current_x [n,0,:] = Forecast[str(nos_id)] # add STOFS data
                   current_x [n,1,:] = v_wind[str(nos_id)][1:] # add wind data
                   current_x [n,2,:] = u_wind[str(nos_id)][1:] # add wind data
                   current_x [n,3,:] = surface_pressure[str(nos_id)][1:] # add surface_pressure data
       
                   latitude = stations['lat'][stations[stations['nos_id'] == nos_id].index[0]]
                   longitude = stations['lon'][stations[stations['nos_id'] == nos_id].index[0]]
                 
                   for nt, t in enumerate(obs_times):
                
                       date_time = timezone.localize(t) # Set timezone
                       current_x [n,4,nt] = compute_zenith_angle(latitude, longitude, date_time)  # add solar zenith angle
                       current_x [n,5,nt] = compute_moon_zenith_angle(latitude, longitude, date_time) # add moon zenith angle

                
                   current_x [n,6,:] = np.full(186, float(latitude))  # add latitude 
                   current_x [n,7,:] = np.full(186, float(longitude)) # add longitude


                   # Calculate the bias
                   obs = pd.read_csv(f'{Observation_dir}/{nos_id}.csv')
                   obs['time'] = pd.to_datetime(obs['time']).dt.tz_localize(None)
                   index_time = obs[obs['time'] == pd.to_datetime(time).strftime('%Y-%m-%d %H:%M:%S')].index
                   if len(index_time) > 0:
                         obs_t = obs[index_time[0]:index_time[0]+186]['value'].reset_index().value
                         current_y [n,:] = Forecast[str(nos_id)]-obs_t
                   else:
                         continue  
        
            
        x.append(current_x)
        y.append(current_y)
        all_time.append(obs_times)
             
x = np.array(x) 
y = np.array(y)
all_time = np.array(all_time)


np.save('x_array_4_years.npy', x)
np.save('y_array_4_years.npy', y)
np.save('time_array_4_years.npy', all_time)
