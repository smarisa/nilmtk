#!/usr/bin/env python3
"""
dsexp.py -- Explore dataset disaggregators.

Usage: python3 dsexp.py DATASET

where DATASET is one of sortd, fortum or eco.

Run "source ./bashrc.sh" to setup the Python environment (works for Samuel).
"""
import sys
from pathlib import Path
from nilmtk import DataSet, HDFDataStore
import nilmtk.dataset_converters as ntkdsc
from pylab import rcParams
import matplotlib.pyplot as plt
rcParams['figure.figsize'] = (13, 6)
plt.style.use('ggplot')
from nilmtk.disaggregate import CombinatorialOptimisation, fhmm_exact, DummyDisaggregator, NFHMMDisaggregator
from nilmtk.utils import print_dict
from nilmtk.metrics import f1_score

if len(sys.argv) < 2:
  print(__doc__)
  sys.exit(1)
dataset_name = sys.argv[1]
# Define paths
for nd in dataset_name, dataset_name.upper(), dataset_name.lower():
  dataset_directory = Path.cwd() / 'data' / nd
  if dataset_directory.exists():
    break
dataset_directory_str = str(dataset_directory)
datastore_file = dataset_directory / ('%s.h5' % (dataset_name.lower()))
datastore_file_str = str(datastore_file)
# If the datastore does not exist (data not converted yet, load it)
if not datastore_file.exists():
  if dataset_name.startswith('sortd'):
    ntkdsc.convert_sortd(dataset_directory_str, datastore_file_str)
  elif dataset_name == 'fortum':
    ntkdsc.convert_fortum(dataset_directory_str, datastore_file_str)
  elif dataset_name == 'eco':
    ntkdsc.convert_eco(dataset_directory_str, datastore_file_str, 'CET')
  elif dataset_name == 'redd':
    ntkdsc.convert_redd(dataset_directory_str, datastore_file_str)
# Then load the dataset into memory
dataset = DataSet(datastore_file_str)
# print basic info of the dataset)
print('\n\n%s\n#### DATASET %s\n' % ('#'*80, dataset_name))
print('\n== dataset.metadata')
print(dataset.metadata)


## Exploring dataset

for bkey in dataset.buildings:
  building = dataset.buildings[bkey]
  elec = building.elec
  print('\n== elec.meters')
  print(elec.meters)
  print('\n== elec.appliances')
  print(elec.appliances)
  _, ax = elec.draw_wiring_graph()
  ax.set_title("Ground truth wiring graph");
  plt.savefig('results/%s__b%d__wiring__truth.png' % (dataset_name, bkey)); plt.clf()
  ax = elec.plot()
  ax.set_title("Ground truth data");
  plt.savefig('results/%s__b%d__elec__truth.png' % (dataset_name, bkey)); plt.clf()
  print('\n== elec.mains().good_sections()')
  print(elec.mains().good_sections())

if len(dataset.buildings) == 1:
  train_building = 1
  disag_building = 1
else:
  train_building = 1
  disag_building = 2

## Dummy training and disaggregation

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
ax.set_title("Dummy disaggregation results");
plt.savefig('results/%s__b%d__elec__dummy.png' % (dataset_name, disag_building)); plt.clf()
f1 = f1_score(da_elec, dataset.buildings[disag_building].elec)
f1.index = da_elec.get_labels([int(i) for i in f1.index])
ax = f1.plot(kind='barh')
ax.set_ylabel('appliance');
ax.set_xlabel('f-score');
ax.set_title("Dummy disaggregation accuracy");
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
ax.set_title("CO wiring graph")
plt.savefig('results/%s__b%d__wiring__co.png' % (dataset_name, disag_building)); plt.clf()
ax = da_elec.plot()
ax.set_title("CO disaggregation results");
plt.savefig('results/%s__b%d__elec__co.png' % (dataset_name, disag_building)); plt.clf()
f1 = f1_score(da_elec, dataset.buildings[disag_building].elec)
f1.index = da_elec.get_labels([int(i) for i in f1.index])
ax = f1.plot(kind='barh')
ax.set_ylabel('appliance');
ax.set_xlabel('f-score');
ax.set_title("CO disaggregation accuracy");
plt.savefig('results/%s__b%d__fscore__co.png' % (dataset_name, disag_building)); plt.clf()
da_data.store.close()

## NFHMM training and disaggregation
vconf = [
  #~ ( 5,  20,  30),
  #~ (10,  20,  30),
  #~ (20,  20,  30),
  #~ (40,  20,  30),
  #~ ( 5,  40,  30),
  #~ (10,  40,  30),
  #~ (20,  40,  30),
  #~ (40,  40,  30),
  ( 5,  60,  30),
  (10,  60,  30),
  (20,  60,  30),
  (40,  60,  30),
#  ( 5, 120,  30),
#  (10, 120,  30),
#  (20, 120,  30),
#  (40, 120,  30),
  #~ (10,  20,  60),
  #~ (20,  20,  60),
  #~ (40,  20,  60),
  #~ (10,  40,  60),
  #~ (20,  40,  60),
  #~ (40,  40,  60),
  #~ (10,  60,  60),
  #~ (20,  60,  60),
  #~ (40,  60,  60),
  #~ (10, 120,  60),
  #~ (20, 120,  60),
  #~ (40, 120,  60),
  #~ (10,  20, 600),
  #~ (20,  20, 600),
  #~ (40,  20, 600),
  #~ (10,  40, 600),
  #~ (20,  40, 600),
  #~ (40,  40, 600),
  #~ (10,  60, 600),
  #~ (20,  60, 600),
  #~ (40,  60, 600),
  #~ (10, 120, 600),
  #~ (20, 120, 600),
  #~ (40, 120, 600),
]
for hp, si, sp in vconf:
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
  ax.set_title("NFHMM wiring graph")
  plt.savefig('results/%s__b%d__wiring__nfhmm__%s.png' % (dataset_name, disag_building, sparam)); plt.clf()
  ax = da_elec.plot()
  ax.set_title("NFHMM disaggregation results")
  plt.savefig('results/%s__b%d__elec__nfhmm__%s.png' % (dataset_name, disag_building, sparam)); plt.clf()
  f1 = f1_score(da_elec, dataset.buildings[disag_building].elec)
  f1.index = elec.get_labels([int(i) for i in f1.index])
  ax = f1.plot(kind='barh')
  ax.set_ylabel('appliance')
  ax.set_xlabel('f-score')
  ax.set_title("NFHMM disaggregation accuracy")
  plt.savefig('results/%s__b%d__fscore__nfhmm__%s.png' % (dataset_name, disag_building, sparam)); plt.clf()
  da_data.store.close()

sys.exit()

## FHMM training and disaggregation

### Training

#~ fhmm = fhmm_exact.FHMM()
#~ fhmm.train(dataset.buildings[train_building].elec)

### Disaggregation

# TODO