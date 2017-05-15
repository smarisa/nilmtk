#!/usr/bin/env python3
"""
compare.py -- Compare CO, FHMM and NFHMM disaggregator performance.

Usage: python3 scripts/compare.py

Run "source ./bashrc.sh" to setup the Python environment (works for Samuel).
"""

## Import resources

# NILMTK imports
from nilmtk import DataSet, HDFDataStore
from nilmtk.dataset_converters import convert_sortd
from nilmtk.disaggregate import CombinatorialOptimisation, fhmm_exact, \
    DummyDisaggregator, NFHMMDisaggregator
from nilmtk.utils import print_dict
from nilmtk.metrics import f1_score

# Other imports and configuration
import sys
from pathlib import Path
from pylab import rcParams
import matplotlib.pyplot as plt
rcParams['figure.figsize'] = (13, 6) # (13, 6)
plt.style.use('ggplot')

## Define and load the dataset

dataset_name = 'sortd-d10-e00'
dataset_directory = Path.cwd() / 'data' / dataset_name
dataset_directory_str = str(dataset_directory)
datastore_file = dataset_directory / ('%s.h5' % (dataset_name.lower()))
datastore_file_str = str(datastore_file)
# If the datastore does not exist convert it
if not datastore_file.exists():
  convert_sortd(dataset_directory_str, datastore_file_str)
# Then load the dataset into memory
dataset = DataSet(datastore_file_str)

## Plot ground truth wiring and consumption data

for bkey in dataset.buildings:
  building = dataset.buildings[bkey]
  elec = building.elec
  _, ax = elec.draw_wiring_graph()
  ax.set_title("B%d Ground truth wiring graph" % (bkey));
  plt.savefig('results/%s__b%d__wiring__truth.png' % (dataset_name, bkey)); plt.clf()
  ax = elec.plot()
  ax.set_title("B%d Ground truth data" % (bkey));
  plt.savefig('results/%s__b%d__elec__truth.png' % (dataset_name, bkey)); plt.clf()

# Define the buildings to be used for training and disaggregation
train_building = 1
disag_building = 1

#~ ## Dummy training and disaggregation

### Training
dum = DummyDisaggregator()
print('\n== dum.train(dataset.buildings[%d].elec)' % (train_building))
dum.train(dataset.buildings[train_building].elec)

### Disaggregation
dum_outfile = dataset_directory / ('%s-da-co.h5' % (dataset_name.lower()))
output = HDFDataStore(str(dum_outfile), 'w')
print('\n== dum.disaggregate(dataset.buildings[%d].mains(), output)' % (disag_building))
dum.disaggregate(dataset.buildings[disag_building].elec.mains(), output)
output.close()

### Results
print('\n== Plotting Dummy disaggregation results...')
da_data = DataSet(str(dum_outfile))
da_elec = da_data.buildings[disag_building].elec
ax = da_elec.plot()
ax.set_title("B%d Dummy disaggregation results" % (disag_building));
plt.savefig('results/%s__b%d__elec__dummy.png' % (dataset_name, disag_building)); plt.clf()
f1 = f1_score(da_elec, dataset.buildings[disag_building].elec)
f1.index = da_elec.get_labels([int(i) for i in f1.index])
ax = f1.plot(kind='barh')
ax.set_ylabel('appliance');
ax.set_xlabel('f-score');
ax.set_title("B%d Dummy disaggregation accuracy" % (disag_building));
plt.savefig('results/%s__b%d__fscore__dummy.png' % (dataset_name, disag_building)); plt.clf()
da_data.store.close()

## CO training and disaggregation

### Training
co = CombinatorialOptimisation()
print('\n== co.train(dataset.buildings[%d].elec)' % (train_building))
co.train(dataset.buildings[train_building].elec)

### Disaggregation
co_outfile = dataset_directory / ('%s-da-co.h5' % (dataset_name.lower()))
output = HDFDataStore(str(co_outfile), 'w')
print('\n== co.disaggregate(dataset.buildings[%d].mains(), output)' % (disag_building))
co.disaggregate(dataset.buildings[disag_building].elec.mains(), output)
output.close()

### Results
print('\n== Plotting CO disaggregation results...')
da_data = DataSet(str(co_outfile))
da_elec = da_data.buildings[disag_building].elec
print('\n== da_elec.meters')
print(da_elec.meters)
print('\n== da_elec.appliances')
print(da_elec.appliances)
_, ax = da_elec.draw_wiring_graph()
ax.set_title("B%d CO wiring graph" % (disag_building));
plt.savefig('results/%s__b%d__wiring__co.png' % (dataset_name, disag_building)); plt.clf()
ax = da_elec.plot()
ax.set_title("B%d CO disaggregation results" % (disag_building));
plt.savefig('results/%s__b%d__elec__co.png' % (dataset_name, disag_building)); plt.clf()
f1 = f1_score(da_elec, dataset.buildings[disag_building].elec)
f1.index = da_elec.get_labels([int(i) for i in f1.index])
ax = f1.plot(kind='barh')
ax.set_ylabel('appliance');
ax.set_xlabel('f-score');
ax.set_title("B%d CO disaggregation results" % (disag_building));
plt.savefig('results/%s__b%d__fscore__co.png' % (dataset_name, disag_building)); plt.clf()
da_data.store.close()

## FHMM training and disaggregation

### Training
fhmm = fhmm_exact.FHMM()
print('\n== fhmm.train(dataset.buildings[%d].elec)' % (train_building))
fhmm.train(dataset.buildings[train_building].elec)

### Disaggregation
fhmm_outfile = dataset_directory / ('%s-da-fhmm.h5' % (dataset_name.lower()))
output = HDFDataStore(str(fhmm_outfile), 'w')
print('\n== fhmm.disaggregate(dataset.buildings[%d].mains(), output)' % (disag_building))
fhmm.disaggregate(dataset.buildings[disag_building].elec.mains(), output)
output.close()

### Results
print('\n== Plotting FHMM disaggregation results...')
da_data = DataSet(str(fhmm_outfile))
da_elec = da_data.buildings[disag_building].elec
print('\n== da_elec.meters')
print(da_elec.meters)
print('\n== da_elec.appliances')
print(da_elec.appliances)
_, ax = da_elec.draw_wiring_graph()
ax.set_title("B%d FHMM wiring graph" % (disag_building));
plt.savefig('results/%s__b%d__wiring__co.png' % (dataset_name, disag_building)); plt.clf()
ax = da_elec.plot()
ax.set_title("B%d FHMM disaggregation results" % (disag_building));
plt.savefig('results/%s__b%d__elec__fhmm.png' % (dataset_name, disag_building)); plt.clf()
f1 = f1_score(da_elec, dataset.buildings[disag_building].elec)
f1.index = da_elec.get_labels([int(i) for i in f1.index])
ax = f1.plot(kind='barh')
ax.set_ylabel('appliance');
ax.set_xlabel('f-score');
ax.set_title("B%d FHMM disaggregation accuracy" % (disag_building));
plt.savefig('results/%s__b%d__fscore__fhmm.png' % (dataset_name, disag_building)); plt.clf()
da_data.store.close()

## NFHMM training and disaggregation

# Define parameter combinations with which to run the algorithm
vconf = [
  #~ (10, 30,  60),
  #~ (10, 60,  60),
  #~ (20, 200, 60),
  (10, 30,  10),
  (10, 60,  10),
  (10, 120, 10),
  (20, 120, 10),
  (10, 200, 10),
  (20, 200, 10),
]

# Run the algorithm
for hp, si, sp in vconf:
  print('\n== Running NFHMM with HP=%d, SI=%d, SP=%d...' % (hp, si, sp))
  sparam = 'HP%d-SI%d-SP%d' % (hp, si, sp)

  ### Training
  nfhmm = NFHMMDisaggregator(heuristic_parameter=hp, sampling_iterations=si)
  print('\n== nfhmm.train(dataset.buildings[%d].elec)' % (train_building))
  nfhmm.train(dataset.buildings[train_building].elec)
  # Note that the train function does not actually do anything atm.

  ### Disaggregation
  nfhmm_outfile = dataset_directory / ('%s-da-nfhmm.h5' % (dataset_name.lower()))
  output = HDFDataStore(str(nfhmm_outfile), 'w')
  print('\n== nfhmm.disaggregate(dataset.buildings[%d].mains(), output)' % (disag_building))
  nfhmm.disaggregate(dataset.buildings[disag_building].elec.mains(), output, max_appliances=len(elec.appliances), sample_period=sp)
  output.close()

  ### Results
  print('\n== Plotting NFHMM disaggregation results...')
  da_data = DataSet(str(nfhmm_outfile))
  da_elec = da_data.buildings[disag_building].elec
  print('\n== da_elec.meters')
  print(da_elec.meters)
  print('\n== da_elec.appliances')
  print(da_elec.appliances)
  _, ax = da_elec.draw_wiring_graph()
  ax.set_title("B%d NFHMM wiring graph (HP=%d, SI=%d, SP=%d)" % (disag_building, hp, si, sp));
  plt.savefig('results/%s__b%d__wiring__nfhmm__%s.png' % (dataset_name, disag_building, sparam)); plt.clf()
  ax = da_elec.plot()
  ax.set_title("B%d NFHMM disaggregation results (HP=%d, SI=%d, SP=%d)" % (disag_building, hp, si, sp));
  plt.savefig('results/%s__b%d__elec__nfhmm__%s.png' % (dataset_name, disag_building, sparam)); plt.clf()
  f1 = f1_score(da_elec, dataset.buildings[disag_building].elec)
  f1.index = elec.get_labels([int(i) for i in f1.index])
  ax = f1.plot(kind='barh')
  ax.set_ylabel('appliance')
  ax.set_xlabel('f-score')
  ax.set_title("B%d NFHMM disaggregation accuracy (HP=%d, SI=%d, SP=%d)" % (disag_building, hp, si, sp));
  plt.savefig('results/%s__b%d__fscore__nfhmm__%s.png' % (dataset_name, disag_building, sparam)); plt.clf()
  da_data.store.close()