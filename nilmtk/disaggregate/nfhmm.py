from __future__ import print_function, division
import warnings
import pandas as pd
import numpy as np
import pickle
import copy
import scipy.stats
#import random as rn

from nilmtk.utils import find_nearest
from nilmtk.feature_detectors import cluster
from nilmtk.disaggregate import Disaggregator
from nilmtk.datastore import HDFDataStore
from nilmtk.disaggregate.nfhmm_ars import ARS

# Fix the seed for repeatability of experiments
SEED = 42
np.random.seed(SEED)


class NFHMMDisaggregator(Disaggregator):
    """Our NFHMM implementation.

    Attributes
    ----------
    MIN_CHUNK_LENGTH : int
      Minimum supported chunk length.
    """

    def __init__(self):
      self.MODEL_NAME = 'NFHMM'
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

        Y = mains.values
        zY = len(Y)
        alpha = 4
        gamma = 1
        delta = 1
        mu_theta = np.nanmean(Y)
        sigma_epsilon = 80
        sigma_theta = np.nanstd(Y)
        Kdag = 2
        Z = np.ones((zY, Kdag-1))
        for y in np.random.sample(np.arange(0, zY), round(0.3*zY):
          Z[y, 0] = 0
        Z = np.vstack(Z, np.zeros(zY))
        mu = np.random.beta(alpha/Kdag, 1, Kdag)
        mu[::-1].sort()
        b = np.random.beta(gamma, delta, Kdag)
        theta = np.random.normal(mu_theta, sigma_theta, Kdag)
        def fmuk(mu, alpha, t, N=10):
          sum = 0
          for i in range(1, N+1):
            sum += 1 / i*((1 - mu)^i)
          return alpha * sum + t*np.log(1 - mu) + (alpha - 1)*log(mu)
        def dfmuk(mu, alpha, t, N=10):
          return alpha * ((1 - mu)^N - 1)/mu - t/(1 - mu) + (alpha - 1)/mu
        def bmuk(mu, ck00, ck01):
          return ck00*np.log(1-mu) + (ck01 - 1)*np.log(mu)
        def dbmuk(mu, ck00, ck01):
          return -ck00/(1-mu) + (ck01 - 1)/mu
        def cfun(i, j, k, Z):
          column = Z[:,k]
          i_inds = np.where(column==i)
          c_val = 0
          for i_ind in i_inds:
            if i_ind < len(column) - 1:
              if column[i_ind + 1] == j:
                c_val += 1
          if i == 0:
            c_val += 1
          return c_val
        def grind_sample_alpha(mu, Kdag, lb=0, ub=5):
          grid = np.linspace(lb, ub, step=0.001)
          post = np.ones(len(grid))
          for i in range(len(grid)):
            for k in range(len(mu)):
              post[i] = post[i]*scipy.stats.beta(mu[k], grid[i]/Kdag, 1)
          post = post/sum(post)
          cpost = np.cumsum(post)
          u = np.random.uniform((0, 1))
          ind = min(np.where(cpost > u))
          return grid[ind]
        for iternum in range(10):
          ActApp = np.where(Z.sum(axis=0) > 0)
          if len(ActApp) == 0:
            Kact = 0
          else:
            Kact = max(ActApp)
          if Kact == 0:
            ub = 1
          else:
            ub = mu(Kact)
          s = np.random.uniform((0, ub))
          Kdag = max(np.where(Z.sum(axis=0) == 0))
          order_mu = mu.ravel().argsort()[::-1][:]
          mu = mu[order_mu]
          b = b[order_mu]
          theta = theta[order_mu]
          Kstar = max(np.where(mu > s))
          while Kstar >= Kdag:
            ub = mu[len(mu) - 1]
            sp = np.random.uniform(0, ub, (4,))
            sp.sort()
            mu_k = ARS(fmuk, dfmuk, xi=sp, lb=0, ub=ub,
                use_lower=False, ns=50, alpha=alpha, t=len(Z))
            b_k = np.random.beta(gamma, delta, 1)
            theta_k = np.random.normal(mu_theta, sigma_theta, 1)
            mu = np.append(mu, mu_k)
            b = np.append(b, b_k)
            theta = np.append(theta, theta_k)
            order_mu = mu.ravel().argsort()[::-1][:]
            mu = mu[order_mu]
            b = b[order_mu]
            theta = theta[order_mu]
            Z = np.vstack(Z, np.zeros(zY))
            Kdag = max(where(Z.sum(axis=0) == 0))
            k_b = Kstar
            Kstar = max(where(mu > s))
            K_a = Kstar
        # TODO

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
