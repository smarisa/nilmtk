from __future__ import print_function, division
import warnings
import pandas as pd
import numpy as np
import pickle
import copy

from nilmtk.utils import find_nearest
from nilmtk.feature_detectors import cluster
from nilmtk.disaggregate import Disaggregator
from nilmtk.datastore import HDFDataStore

# Fix the seed for repeatability of experiments
SEED = 42
np.random.seed(SEED)


class DummyDisaggregator(Disaggregator):
  """A dummy disaggregator that performs disaggregation using very simple logic.

  Attributes
  ----------
  MIN_CHUNK_LENGTH : int
    Minimum supported chunk length.
  """

  def __init__(self):
    self.MODEL_NAME = 'DUMMY'
    self.MIN_CHUNK_LENGTH = 10
    self.model = []

  def train(self, metergroup, num_states_dict=None, **load_kwargs):
    """Train the dummy model using submetered data.

    The dummy uses the data to only learn the mean consumptions of each
    submeter.

    Parameters
    ----------
    metergroup : a nilmtk.MeterGroup object
    num_states_dict : dict
    **load_kwargs : keyword arguments passed to `meter.power_series()`
    """
    # Iterate over each submeter
    for i, meter in enumerate(metergroup.submeters().meters):
      # Store a new model dict for the meter
      self.model.append({'training_metadata': meter})
      print("Training model for submeter [%s] '%s'" % (i, meter))
      # Generate a model for the meter using its power series data
      power_series = meter.power_series(**load_kwargs)
      for chunk in power_series:
        self.train_on_chunk(chunk, meter)
    print("Done training!")

  def train_on_chunk(self, chunk, meter):
    # Load the model dict for this meter
    model = [model for model in self.model if model['training_metadata'] == meter][0]
    # The model is updated to simply reflect the mean of the last chunk
    print('Training model meter %s with chunk:\n%s' % (meter, chunk.describe()))
    model['mean'] = chunk.mean()
    print('Mean: %s' % (model['mean']))

  def disaggregate(self, mains, output_datastore, **load_kwargs):
    '''Disaggregate mains according to a dumb algorithm.

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
    '''
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
      # Disaggregate the data chunk into a dataframe where each column
      # represents a disaggregated appliance with column names corresponding to
      # the integer indices of the appliances in `self.model`.
      appliance_powers = self.disaggregate_chunk(chunk)
      print('Disaggregated data %s:\n%s' % (appliance_powers.shape, appliance_powers.head()))
      # Convert the dataframe into XX??
      for i, model in enumerate(self.model):
        meter = model['training_metadata']
        # Load the disaggregated time series for appliance `i`
        appliance_power = appliance_powers[i]
        # Skip it if empty
        if len(appliance_power) == 0:
          continue
        data_is_available = True
        # Reorganize the time series as a dataframe with the column headings
        # of the mains power series
        cols = pd.MultiIndex.from_tuples([chunk.name])
        df = pd.DataFrame(appliance_power.values, index=appliance_power.index,
            columns=cols)
        key = '{}/elec/meter{}'.format(building_path, meter.instance())
        print('Storing disaggregation result for model [%d] under key %s:\n%s\n%s\n%s'
            % (i, key, df.head(), df.tail(), df.describe()))
        output_datastore.append(key, df)
      # Copy mains data to disag output
      mains_df = pd.DataFrame(chunk, columns=cols)
      output_datastore.append(key=mains_data_location, value=mains_df)
    # Store all metadata into the datastore
    if data_is_available:
      self._save_metadata_for_disaggregation(
          output_datastore=output_datastore,
          sample_period=load_kwargs['sample_period'],
          measurement=measurement,
          timeframes=timeframes,
          building=mains.building(),
          meters=[d['training_metadata'] for d in self.model]
      )

  def disaggregate_chunk(self, mains):
    """In-memory disaggregation of mains data using the pre-trained model.

    Parameters
    ----------
    mains : pd.Series

    Returns
    -------
    appliance_powers : pd.DataFrame where the rows represent time and each
        column represents a disaggregated appliance. Column names are the
        integer index into `self.model` for the appliance in question.
    """
    # Refuse too short chunks
    if len(mains) < self.MIN_CHUNK_LENGTH:
      raise RuntimeError('Chunk is too short.')
    ## Make a prediction for each learned model and store it into a dict by
    #  index
    appliance_powers_dict = {}
    mean_sum = sum([model['mean'] for model in self.model])
    for i, model in enumerate(self.model):
      # Load meter
      meter = model['training_metadata']
      print("Estimating power demand for [%s] '%s'" % (i, meter))
      # Make power series prediction for the meter based on learned model
      predicted_power = (model['mean']/mean_sum)*mains.values
      column = pd.Series(predicted_power, index=mains.index, name=i)
      # Save the series by index
      appliance_powers_dict[i] = column
    # Convert the dict into the dataframe and return it
    appliance_powers = pd.DataFrame(appliance_powers_dict, dtype='float32')
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
