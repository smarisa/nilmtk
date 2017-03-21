from __future__ import print_function, division
from re import compile as regex
from os.path import join, isfile
from six import iteritems
from nilmtk import DataSet
from nilmtk.utils import get_datastore, check_directory_exists
from nilmtk.datastore import Key
from nilm_metadata import save_yaml_to_datastore
import pandas as pd # See http://pandas.pydata.org/pandas-docs/stable/dsintro.html
import numpy as np
import yaml
import glob

def convert_sortd(input_path, output_filename, format='HDF'):
    """Converts the dataset to NILMTK HDF5 format.

    For more information about the SOR test dataset, contact Samuel Marisa.

    Parameters
    ----------
    input_path : str
        The root path of the dataset.  It is assumed that the YAML
        metadata is in 'input_path/metadata'.
    output_filename : str
        The destination filename (including path and suffix).
    format : str
        format of output. Either 'HDF' or 'CSV'. Defaults to 'HDF'

    Example usage:
    --------------
    convert('/sortd', 'store.h5')
    """
    print('Attempting to convert the SORTD dataset at %s into %s in NILMTK %s format...' % (input_path, output_filename, format))
    # Ensure that the input directory exists
    check_directory_exists(input_path)
    # Load the dataset metadata
    with open(join(input_path, 'metadata/dataset.yaml'), 'r') as stream:
      dataset_metadata = yaml.load(stream)
    # Open the datastore
    store = get_datastore(output_filename, format, mode='w')
    # Iterate through all building metadata files found in the dataset
    for metadata_file in glob.glob(join(input_path, 'metadata/building[0-9]*.yaml')):
      # Load the building metadata
      with open(metadata_file, 'r') as stream: metadata = yaml.load(stream)
      building_id = int(metadata['instance'])
      print('==> Loading building %d defined at %s. Please wait...' % (building_id, metadata_file))
      for meter_id, meter_data in metadata['elec_meters'].items():
        meter_id = int(meter_id)
        key = Key(building=building_id, meter=meter_id)
        # Load the raw data from the data location
        print('  - Loading meter %s from %s...' % (meter_id, meter_data['data_location']))
        columns = [('power', 'active')]
        df = pd.read_csv(join(input_path, meter_data['data_location']), sep=',', names=columns, dtype={m: np.float32 for m in columns})
        # Convert the timestamp index column to timezone-aware datetime
        df.index = pd.to_datetime(df.index.values, unit='s', utc=True)
        df = df.tz_convert(dataset_metadata['timezone'])
        #df = pd.read_csv(join(input_path, db_file), sep=';', names=('Datetime', 'P1', 'P2', 'P3'), dtype={'P1': np.float64, 'P2': np.float64, 'P3': np.float64}, parse_dates=[1])
        print(df.info())
        print(df.head())
        #print(df.tail())
        print("  - Storing data under key %s in the datastore..." % (str(key)))
        store.put(str(key), df)
      print("  - Building %s loaded!" % (building_id))
    print("Adding the metadata into the store...")
    save_yaml_to_datastore(join(input_path, 'metadata'), store)
    print("Closing the store...")
    store.close()
    print("Done converting SORTD dataset to HDF5!")
