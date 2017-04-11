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
from nilmtk.disaggregate import CombinatorialOptimisation, fhmm_exact
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
  if dataset_name == 'sortd':
    ntkdsc.convert_sortd(dataset_directory_str, datastore_file_str)
  elif dataset_name == 'sortd2':
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
  plt.savefig('results/%s__b%d__draw_wiring_graph.png' % (dataset_name, bkey)); plt.clf()
  ax = elec.plot()
  ax.set_title("Ground truth data");
  plt.savefig('results/%s__b%d__elec_plot.png' % (dataset_name, bkey)); plt.clf()
  print('\n== elec.mains().good_sections()')
  print(elec.mains().good_sections())

if len(dataset.buildings) == 1:
  train_building = 1
  disag_building = 1
else:
  train_building = 1
  disag_building = 2

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

## FHMM training and disaggregation

### Training

fhmm = fhmm_exact.FHMM()
fhmm.train(elec)

### Disaggregation

#TODO

## Exploring disaggregation results

### CO
print('\n== Plotting disaggregation results...')
co_data = DataSet(str(co_outfile))
co_elec = co_data.buildings[disag_building].elec
ax = co_elec.plot()
ax.set_title("CO disaggregation results");
plt.savefig('results/%s__b%d__co_da_results.png' % (dataset_name, disag_building)); plt.clf()
f1 = f1_score(co_elec, elec)
f1.index = co_elec.get_labels([int(i) for i in f1.index])
ax = f1.plot(kind='barh')
ax.set_ylabel('appliance');
ax.set_xlabel('f-score');
ax.set_title("CO disaggregation accuracy");
plt.savefig('results/%s__b%d__co_da_accuracy.png' % (dataset_name, disag_building)); plt.clf()
co_data.store.close()