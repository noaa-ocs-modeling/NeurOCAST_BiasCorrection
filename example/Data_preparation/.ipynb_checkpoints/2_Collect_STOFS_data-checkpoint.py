import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import xarray as xr
import s3fs
import tempfile
import multiprocessing
import shapely  # Importing shapely for geometric operations 
import argparse
import thalassa 
from typing import List


#dir_name = '/lustre/code/BiasCorrection/Codes/Data'
dir_name = '/lustre/code/BiasCorrection/Codes/STOFS_Data'

# Argument parser setup
parser = argparse.ArgumentParser(description="Fetch STOFS forecast data.")
parser.add_argument("date", type=str, help="The date in YYYYMMDD format.")
parser.add_argument("cycle", type=str, help="The cycle (e.g., 00, 06, 12, 18).")

# Parse the command-line arguments
args  = parser.parse_args()
date  = args.date
cycle = args.cycle

# load the sations instaed
stations     = pd.read_csv('stations.csv')
bucket_name  = 'noaa-gestofs-pds'
STOFS_file   = 'fields.cwl'
name         = 'stofs_2d_glo' 

def read_netcdf_from_s3(bucket_name, key):
    """
    Function to read a NetCDF file from an S3 bucket using thalassa API.
    
    Parameters:
    - bucket_name: Name of the S3 bucket
    - key: Key/path to the NetCDF file in the bucket
    
    Returns:
    - ds: xarray Dataset containing the NetCDF data
    """
    s3 = s3fs.S3FileSystem(anon=True)
    url = f"s3://{bucket_name}/{key}"
    ds = xr.open_dataset(s3.open(url, 'rb'), drop_variables=['nvel'])
    return ds

def normalize_data(ds, name, cycle, bucket_name, base_key, field_cwl , filename, date):
    """
    Function to modify/normalize a dataset using the Thalassa package.

    Parameters:
    - ds: xarray Dataset containing the data
    - name: folder name 
    - bucket_name: Name of the S3 bucket
    - base_key: Base key for the dataset in the S3 bucket
    - schout: adcirc like file name
    - filename: Original filename to be replaced
    - date: Date string for the new filename
    
    Returns:
    - normalized_ds: Thalassa dataset ready for cropping or visualizing
    """

    if 'element' in ds:
        normalized_ds = thalassa.normalize(ds)
        box = (-74.5, -67, 40, 44)
        bbox = shapely.box(box[0], box[2], box[1], box[3])  # Create a shapely box from the bounding box coordinates
        subset_ds = thalassa.crop(normalized_ds, bbox)  # Crop the dataset using the bounding box
    else:
        
        key = f'{base_key}/{name}.{filename}'
        ds_with_element_key = key.replace(filename,  f't{cycle}z.fields.cwl.maxele.nc')
        ds_with_element = read_netcdf_from_s3(bucket_name, ds_with_element_key)  # Read NetCDF data from S3 bucket

        # Modify the field2d.nc file based on schout_adcirc.nc file
        ds['nele'] = ds_with_element['nele']
        ds['nvertex'] = ds_with_element['nvertex']
        ds['element'] = ds_with_element['element']

        # Normalize data
        normalized_ds = thalassa.normalize(ds)
        box = (-74.5, -67, 40, 44)
        bbox = shapely.box(box[0], box[2], box[1], box[3])  # Create a shapely box from the bounding box coordinates
        subset_ds = thalassa.crop(normalized_ds, bbox)  # Crop the dataset using the bounding box
    return subset_ds


def  get_indices_of_nearest_nodes(ds: xr.Dataset, lon: float, lat: float, num_nodes: int ) -> List[int]:
    # https://www.unidata.ucar.edu/blogs/developer/en/entry/accessing_netcdf_data_by_coordinates
    # https://github.com/Unidata/python-workshop/blob/fall-2016/notebooks/netcdf-by-coordinates.ipynb
    dist = abs(ds.lon - lon) ** 2 + abs(ds.lat - lat) ** 2
    indices_of_nearest_nodes = dist.argsort()[:num_nodes]
    return indices_of_nearest_nodes.values


def Fetch_STOFS_data(stations, name, cycle, STOFS_file, bucket_name, date, dir_name):
    """
    Function to read, normalize dataset and process stations data.

    Parameters:
    - stations: DataFrame containing station information (nos_id, lon, lat)
    - name: folder name
    - cycle: cycle identifier
    - STOFS_file: name of the STOFS file
    - bucket_name: S3 bucket name
    - date: date string
    - dir_name: directory to save the CSV file
    
    Returns:
    - Forecast: DataFrame containing forecasted data for each station
    """


    # Read NetCDF data from S3 bucket

    try : 
        base_key = f'{name}.{date}'
        filename = f't{cycle}z.{STOFS_file}.nc'
        key = f'{base_key}/{name}.{filename}'
        dataset = read_netcdf_from_s3(bucket_name, key)
    except:
        print(f"Reading failed with name {name}")
        name = 'estofs'  # Change name if normalization fails
        base_key = f'{name}.{date}'
        filename = f't{cycle}z.{STOFS_file}.nc'
        key = f'{base_key}/{name}.{filename}'
        dataset = read_netcdf_from_s3(bucket_name, key)

    # Normalize dataset
    normalize_dataset = normalize_data(dataset, name, cycle, bucket_name, base_key, 'fields.cwl', filename, date)
    
    # Initialize an empty DataFrame to store the data
    index = pd.DataFrame()

    # Initialize an empty list to store data
    data = []
   

    # Get nearest nodes for each station
    for nos_id, x, y in zip(stations['nos_id'], stations['lon'], stations['lat']):
        index_values = get_indices_of_nearest_nodes(normalize_dataset, x, y, 5)
        data.append({'nos_id': nos_id, 'index_value': index_values})

    # Convert the list of dictionaries to a DataFrame
    index = pd.DataFrame(data)

    # Initialize an empty DataFrame to store the forecasted data
    Forecast = pd.DataFrame()

    # Loop over each nos_id and extract forecast data
    for nos_id in stations.nos_id:
        subset_values = normalize_dataset['zeta'].isel(node=index.index_value[index['nos_id'] == nos_id].values[0])
        flattened_values = np.nanmean(subset_values, axis=1)
        Forecast[int(nos_id)] = flattened_values

    # Save the forecast to a CSV file
    output_filename = f"{dir_name}/STOFS_2D_{date}_{cycle}.csv"
    Forecast.to_csv(output_filename, index=False)

    print(f"Saved {output_filename}")


Fetch_STOFS_data(stations, name, cycle, STOFS_file, bucket_name, date, dir_name)
