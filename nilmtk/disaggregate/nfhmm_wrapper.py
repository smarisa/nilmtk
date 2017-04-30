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

  def __init__(self, heuristic_parameter=40, sampling_iterations=20):
    self.MODEL_NAME = 'NFHMM'
    self.MIN_CHUNK_LENGTH = 100
    self.MAX_DISAG_ATTEMPTS_PER_CHUNK = 5
    self.HEURISTIC_PARAMETER = heuristic_parameter
    self.SAMPLING_ITERATIONS = sampling_iterations
    self.DISAG_ATTEMPT_TIMEOUT = 30 + 10*self.SAMPLING_ITERATIONS # [seconds]
    self.NFHMM_ROOT_DIR = '/home/smarisa/snc/git/sor-nilm'
    self.model = True

  def train(self, metergroup, **load_kwargs):
    """Train the unsupervised NFHMM model using only site meter data.

    FIXME, now it does both training and disaggregation in the
    disaggregation methods.

    Parameters
    ----------
    metergroup : a nilmtk.MeterGroup object
    **load_kwargs : keyword arguments passed to `meter.power_series()`
    """
    pass

  def train_on_chunk(self, chunk, meter):
    pass

  def disaggregate(self, mains, output_datastore, max_appliances, **load_kwargs):
    """Disaggregate mains with the NFHMM algorithm.

    Parameters
    ----------
    mains : nilmtk.ElecMeter or nilmtk.MeterGroup
    output_datastore : instance of nilmtk.DataStore subclass
        For storing power predictions from disaggregation algorithm.
    max_appliances : int, maximum number of appliances to disaggregate mains into
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
    load_kwargs.setdefault('sample_period', max(mains.sample_period(), 30))
    load_kwargs.setdefault('chunksize', 8000*load_kwargs['sample_period']/mains.sample_period() - 1)
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
      # Disaggregate the data chunk into a dataframe with an index equal to that
      # of chunk and a column for each disaggregated appliance
      appliance_powers = self.disaggregate_chunk(chunk, appliances=max_appliances)
      # Ignore undisaggregateable chunks
      if type(appliance_powers) != pd.DataFrame:
        continue
      # Ignore appliances found in excess
      if appliance_powers.shape[1] > max_appliances:
        print('Note: Ignoring %d excess appliances!'
            % (appliance_powers.shape[1] - max_appliances))
        appliance_powers.drop(appliance_powers.columns[range(max_appliances,
            appliance_powers.shape[1])], axis=1, inplace=True)
      print('Disaggregated data %s:\n%s' % (appliance_powers.shape, appliance_powers.head()))
      # Create a multi index with the column headings of the mains power series
      cols = pd.MultiIndex.from_tuples([chunk.name])
      # Add the power series of each disaggregated appliance into the datastore
      for i, appliance in enumerate(appliance_powers):
        # Load the disaggregated time series for the appliance
        appliance_power = appliance_powers[appliance]
        # Reorganize the time series as a dataframe with the column headings
        # of the mains power series
        df = pd.DataFrame(appliance_power.values, index=appliance_power.index,
            columns=cols)
        key = '{}/elec/meter{}'.format(building_path, i+2) # meter1 is for mains
        print('Storing disaggregation result for [%d] "%s" under key %s:\n%s\n%s\n%s'
            % (i, appliance_powers.columns[i], key, df.head(), df.tail(), df.describe()))
        output_datastore.append(key, df)
      # Copy mains data to disag output
      mains_df = pd.DataFrame(chunk, columns=cols, dtype='float64')
      output_datastore.append(key=mains_data_location, value=mains_df)
      # Record metadata
      timeframes.append(chunk.timeframe)
      measurement = chunk.name
    # Store all metadata into the datastore
    self._save_metadata_for_disaggregation(
        output_datastore=output_datastore, # target of data storage
        sample_period=load_kwargs['sample_period'],
        measurement=measurement, # eg. ("power", "active")
        timeframes=timeframes,
        building=mains.building(), # building instance number
        supervised=False, # this was unsupervised disaggregation
        num_meters=appliance_powers.shape[1] # number of appliances/submeters
    )

  def disaggregate_chunk(self, mains, appliances=5):
    """In-memory disaggregation of mains data using the pre-trained model.

    The function writes the chunk into shared memory and then calls the actual
    R implementation of the NFHMM algorithm to do the disaggregation and then
    read the results from a CSV the R script writes.

    Parameters
    ----------
    mains : pd.Series
    appliances : int, initial guess of the number of total appliances

    Returns
    -------
    appliance_powers : pd.DataFrame where the rows represent time and each
        column represents a disaggregated appliance.
    """
    # Refuse too short chunks
    if len(mains) < self.MIN_CHUNK_LENGTH:
      raise RuntimeError('Chunk is too short.')
    # Define temporary IO file paths
    pfi = '/run/shm/nfhmm_in.csv'
    pfo = '/run/shm/nfhmm_out.csv'
    # Write the series into a CSV file with specific column headings
    chunk = pd.Series(mains, name='Aggregate')
    chunk.to_csv(pfi, index_label='Timestamp', header=True)
    # Run the actual R implementation of the algorithm
    cmd = 'Rscript src/r_fhmm.R -b -i "%s" -o "%s" -a %d -p %d -n %d -v' \
        % (pfi, pfo, appliances, self.HEURISTIC_PARAMETER, self.SAMPLING_ITERATIONS)
    for i in range(1, self.MAX_DISAG_ATTEMPTS_PER_CHUNK+1):
      print('Running "%s" (timeout=%ds)...' % (cmd, self.DISAG_ATTEMPT_TIMEOUT))
      p = sp.Popen(shlexsplit(cmd), cwd=self.NFHMM_ROOT_DIR)
      try:
        p.wait(timeout=self.DISAG_ATTEMPT_TIMEOUT)
        msg = 'with code %d' % (p.returncode)
      except sp.TimeoutExpired:
        p.kill()
        msg = 'due to surpassing the timeout (%ds)' % (self.DISAG_ATTEMPT_TIMEOUT)
      if p.returncode == 0:
        break
      print('Run attempt %d/%d failed %s!' % (i,
          self.MAX_DISAG_ATTEMPTS_PER_CHUNK, msg))
    if p.returncode != 0:
      print('Warning: Disaggregating the chunk failed!')
      return None
    print('The R NFHMM implementation finished succesfully!')
    # Read the disaggregation results into a dataframe
    appliance_powers = pd.read_csv(pfo)
    # Reuse the existing index instead of the newly read timestamp column; These
    # should be equal despite the frequency downsampling
    del appliance_powers['Timestamp']
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
