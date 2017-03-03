from __future__ import print_function, division
from os import remove
from os.path import join
from six import iteritems
from nilmtk import DataSet
from nilmtk.utils import get_datastore, check_directory_exists
from nilmtk.datastore import Key
from nilm_metadata import convert_yaml_to_hdf5
import pandas as pd
import numpy as np


ONE_SEC_COLUMNS = [('power', 'active'), ('power', 'apparent'), ('voltage', '')]
TZ = 'Europe/Helsinki'


def convert_fortum(fortum_path, output_filename, format='HDF'):
    """Converts the Fortum dataset to NILMTK HDF5 format.

    For more information about the Fortum dataset, contact Samuel Marisa.

    Parameters
    ----------
    fortum_path : str
        The root path of the Fortum dataset.  It is assumed that the YAML
        metadata is in 'ukdale_path/metadata'.
    output_filename : str
        The destination filename (including path and suffix).
    format : str
        format of output. Either 'HDF' or 'CSV'. Defaults to 'HDF'

    Example usage:
    --------------
    convert('/Fortum/db', 'store.h5')    
    """
    check_directory_exists(input_path)
    files = [f for f in listdir(input_path) if isfile(join(input_path, f)) and
             '.db' in f]
    # Sorting Lexicographically
    files.sort()
    store = get_datastore(output_filename, format, mode='w')
    for i, db_file in enumerate(files):
        key = Key(building=1, meter=(i + 1))
        print('Loading file #', (i + 1), ' : ', db_file, '. Please wait...')
        df = pd.read_csv(join(input_path, csv_file))
        # Due to fixed width, column names have spaces :(
        df.columns = [x.replace(" ", "") for x in df.columns]
        df.index = pd.to_datetime(df[TIMESTAMP_COLUMN_NAME], unit='s', utc=True)
        df = df.drop(TIMESTAMP_COLUMN_NAME, 1)
        df = df.tz_localize('GMT').tz_convert(TIMEZONE)
        df.rename(columns=lambda x: columnNameMapping[x], inplace=True)
        df.columns.set_names(LEVEL_NAMES, inplace=True)
        df = df.convert_objects(convert_numeric=True)
        df = df.dropna()
        df = df.astype(np.float32)
        store.put(str(key), df)
        print("Done with file #", (i + 1))
    store.close()


    ac_type_map = _get_ac_type_map(ukdale_path)

    def _ukdale_measurement_mapping_func(house_id, chan_id):
        ac_type = ac_type_map[(house_id, chan_id)][0]
        return [('power', ac_type)]

    # Open DataStore
    store = get_datastore(output_filename, format, mode='w')

    # Convert 6-second data
    _convert(ukdale_path, store, _ukdale_measurement_mapping_func, TZ,
             sort_index=False)
    store.close()

    # Add metadata
    if format == 'HDF':
        convert_yaml_to_hdf5(join(ukdale_path, 'metadata'), output_filename)

    # Convert 1-second data
    store.open(mode='a')
    _convert_one_sec_data(ukdale_path, store, ac_type_map)

    store.close()
    print("Done converting UK-DALE to HDF5!")


def _get_ac_type_map(ukdale_path):
    """First we need to convert the YAML metadata to HDF5
    so we can load the metadata into NILMTK to allow
    us to use NILMTK to find the ac_type for each channel.
    
    Parameters
    ----------
    ukdale_path : str

    Returns
    -------
    ac_type_map : dict.  
        Keys are pairs of ints: (<house_instance>, <meter_instance>)
        Values are list of available power ac type for that meter.
    """

    hdf5_just_metadata = join(ukdale_path, 'metadata', 'ukdale_metadata.h5')
    convert_yaml_to_hdf5(join(ukdale_path, 'metadata'), hdf5_just_metadata)
    ukdale_dataset = DataSet(hdf5_just_metadata)
    ac_type_map = {}
    for building_i, building in iteritems(ukdale_dataset.buildings):
        elec = building.elec
        for meter in elec.meters + elec.disabled_meters:
            key = (building_i, meter.instance())
            ac_type_map[key] = meter.available_ac_types('power')
    ukdale_dataset.store.close()
    remove(hdf5_just_metadata)
    return ac_type_map


def _convert_one_sec_data(ukdale_path, store, ac_type_map):
    ids_of_one_sec_data = [
        identifier for identifier, ac_types in iteritems(ac_type_map)
        if ac_types == ['active', 'apparent']]

    if not ids_of_one_sec_data:
        return

    for identifier in ids_of_one_sec_data:
        key = Key(building=identifier[0], meter=identifier[1])
        print("Loading 1-second data for", key, "...")
        house_path = 'house_{:d}'.format(key.building)
        filename = join(ukdale_path, house_path, 'mains.dat')
        df = _load_csv(filename, ONE_SEC_COLUMNS, TZ)
        store.put(str(key), df)

    store.close()
