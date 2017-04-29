from __future__ import print_function, division
import os
import pandas as pd
import numpy as np
import pickle
import subprocess as sp
from shlex import split as shlexsplit
# import random as rn

from nilmtk.disaggregate import Disaggregator
from nilmtk.datastore import HDFDataStore

# Fix the seed for repeatability of experiments
SEED = 42
np.random.seed(SEED)

class NFHMMDisaggregator(Disaggregator):
  """A wrapper of an R implementation of an unsupervised NFHMM algorithm.

  Attributes
  ----------
  MIN_CHUNK_LENGTH : int
    Minimum supported chunk length.
  """

  def __init__(self):
    self.MODEL_NAME = 'NFHMM'
    self.MIN_CHUNK_LENGTH = 100
    self.NFHMM_ROOT_DIR = '/home/smarisa/snc/git/sor-nilm'
    self.model = True

  def train(self, metergroup, num_states_dict=None, **load_kwargs):
    """Train the unsupervised NFHMM model using only site meter data.

    FIXME, now it does both training and disaggregation in the
    disaggregation methods.

    Parameters
    ----------
    metergroup : a nilmtk.MeterGroup object
    num_states_dict : dict
    **load_kwargs : keyword arguments passed to `meter.power_series()`
    """
    pass

  def train_on_chunk(self, chunk, meter):
    pass

  def disaggregate(self, mains, output_datastore, **load_kwargs):
    """Disaggregate mains with the NFHMM algorithm.

    Parameters
    ----------
    mains : nilmtk.ElecMeter or nilmtk.MeterGroup
    output_datastore : instance of nilmtk.DataStore subclass
        For storing power predictions from disaggregation algorithm.
    sample_period : number, optional
        The desired sample period in seconds. Set to 60 by default.
    sections : TimeFrameGroup, optional
        Set to mains.good_sections() by default.
    **load_kwargs : key word arguments
        Passed to `mains.power_series(**kwargs)`
    """
    # Run any required pre disaggregation checks
    load_kwargs = self._pre_disaggregation_checks(load_kwargs)
    # Set defaults
    load_kwargs.setdefault('sample_period', 60)
    load_kwargs.setdefault('sections', mains.good_sections())
    # Construct data paths
    building_path = '/building{}'.format(mains.building())
    mains_data_location = building_path + '/elec/meter1'
    # Initialize variables
    timeframes = []
    data_is_available = False
    # Iterate over the power timeseries chunks
    for chunk in mains.power_series(**load_kwargs):
      # Skip too short chunks
      if len(chunk) < self.MIN_CHUNK_LENGTH:
        continue
      # Record metadata
      timeframes.append(chunk.timeframe)
      measurement = chunk.name
      # Disaggregate the data chunk into a dataframe with an index equal to that
      # of chunk and a column for each disaggregated appliance
      appliance_powers = self.disaggregate_chunk(chunk)
      print('Disaggregated appliances:\n%s' % (appliance_powers))
      # Add the power series of each disaggregated appliance into the datastore
      for i, appliance_power in enumerate(appliance_powers):
        # Load the disaggregated time series for appliance `i`
        appliance_power = appliance_powers[i]
        # Skip it if empty
        if len(appliance_power) == 0:
          continue
        # Reorganize the time series as a dataframe with the column headings
        # of the mains power series
        cols = pd.MultiIndex.from_tuples([chunk.name])
        df = pd.DataFrame(appliance_power.values, index=appliance_power.index,
            columns=cols)
        print('Storing disaggregation result for meter %s:\n%s' % (i, df))
        key = '{}/elec/meter{}'.format(building_path, i)
        output_datastore.append(key, df)
      # Copy mains data to disag output
      mains_df = pd.DataFrame(chunk, columns=cols)
      output_datastore.append(key=mains_data_location, value=mains_df)
    # Store all metadata into the datastore
    self._save_metadata_for_disaggregation(
        output_datastore=output_datastore,
        sample_period=load_kwargs['sample_period'],
        measurement=measurement,
        timeframes=timeframes,
        building=mains.building(),
        meters=[i for i in range(len(appliance_powers))]
    )

  def disaggregate_chunk(self, mains):
    """In-memory disaggregation of mains data using the pre-trained model.

    The function writes the chunk into shared memory and then calls the actual
    R implementation of the NFHMM algorithm to do the disaggregation and then
    read the results from a CSV the R script writes.

    Parameters
    ----------
    mains : pd.Series

    Returns
    -------
    appliance_powers : pd.DataFrame where the rows represent time and each
        column represents a disaggregated appliance.
    """
    # Refuse too short chunks
    if len(mains) < self.MIN_CHUNK_LENGTH:
      raise RuntimeError('Chunk is too short.')
    # Write the series into a CSV file
    pfi = '/run/shm/nfhmm_in.csv'
    pfo = '/run/shm/nfhmm_out.csv'
    mains.to_csv(pfi)
    # Run the actual R implementation of the algorithm
    p = sp.Popen(shlexsplit('Rscript src/r_fhmm.R -b -i "%s" -o "%s" -v' % (pfi, pfo)),
        cwd=self.NFHMM_ROOT_DIR)
    p.wait()
    print('NFHMM R implementation finished with rv %d!' % (p.returncode))
    # Reand the disaggregation results into a dataframe and return it
    appliance_powers = pd.read_csv(pfo)
    print(appliance_powers.index)
    print(mains.index)
    appliance_powers.index = mains.index
    # Remove the temporary IO files
    for pf in pfi, pfo: os.unlink(pf)
    return appliance_powers

  def import_model(self, filename):
      with open(filename, 'r') as f:
        imported_model = pickle.load(f)
      self.model = imported_model.model
      self.MIN_CHUNK_LENGTH = imported_model.MIN_CHUNK_LENGTH

  def export_model(self, filename):
      # Can't pickle datastore, so convert to filenames
      with open(filename, 'wb') as f:
        pickle.dump(self.model, f)
