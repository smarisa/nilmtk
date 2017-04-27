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
    """A dumb hack.

    Attributes
    ----------
    MIN_CHUNK_LENGTH : int
      Minimum supported chunk length.
    """

    def __init__(self):
      self.MODEL_NAME = 'DUMMY'
      self.MIN_CHUNK_LENGTH = 100
      self.model = []

    def train(self, metergroup, num_states_dict=None, **load_kwargs):
        """Train using 1D CO. Places the learnt model in the `model` attribute.

        Parameters
        ----------
        metergroup : a nilmtk.MeterGroup object
        num_states_dict : dict
        **load_kwargs : keyword arguments passed to `meter.power_series()`

        Notes
        -----
        * only uses first chunk for each meter (TODO: handle all chunks).
        """
        for i, meter in enumerate(metergroup.submeters().meters):
          self.model.append({'training_metadata': meter})
          print("Training model for submeter '{}'".format(meter))
          power_series = meter.power_series(**load_kwargs)
          for chunk in power_series:
            self.train_on_chunk(chunk, meter)
        print("Done training!")

    def train_on_chunk(self, chunk, meter):
      pass

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
        load_kwargs = self._pre_disaggregation_checks(load_kwargs)

        load_kwargs.setdefault('sample_period', 60)
        load_kwargs.setdefault('sections', mains.good_sections())

        timeframes = []
        building_path = '/building{}'.format(mains.building())
        mains_data_location = building_path + '/elec/meter1'
        data_is_available = False

        for chunk in mains.power_series(**load_kwargs):
          # Check that chunk is sensible size
          if len(chunk) < self.MIN_CHUNK_LENGTH:
            continue

          # Record metadata
          timeframes.append(chunk.timeframe)
          measurement = chunk.name

          appliance_powers = self.disaggregate_chunk(chunk)

          for i, model in enumerate(self.model):
            meter = model['training_metadata']
            appliance_power = appliance_powers[i]
            if len(appliance_power) == 0:
              continue
            data_is_available = True
            cols = pd.MultiIndex.from_tuples([chunk.name])
            meter_instance = meter.instance()
            df = pd.DataFrame(
                appliance_power.values, index=appliance_power.index,
                columns=cols)
            key = '{}/elec/meter{}'.format(building_path, meter_instance)
            output_datastore.append(key, df)

          # Copy mains data to disag output
          mains_df = pd.DataFrame(chunk, columns=cols)
          output_datastore.append(key=mains_data_location, value=mains_df)

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
        """In-memory disaggregation.

        Parameters
        ----------
        mains : pd.Series

        Returns
        -------
        appliance_powers : pd.DataFrame where the rows represent time and each
            column represents a disaggregated appliance. Column names are the
            integer index into `self.model` for the appliance in question.
        """
        if len(mains) < self.MIN_CHUNK_LENGTH:
            raise RuntimeError('Chunk is too short.')

        appliance_powers_dict = {}
        for i, model in enumerate(self.model):
          meter = model['training_metadata']
          print("Estimating power demand for '{}'".format(meter))
          predicted_power = [500] * mains.index.size
          column = pd.Series(predicted_power, index=mains.index, name=i)
          appliance_powers_dict[i] = column
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
