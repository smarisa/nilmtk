# export PYTHONPATH=/home/smarisa/snc/git/nilmtk:/home/smarisa/snc/git/nilm_metadata PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin:/snap/bin
# ipython notebook
"""
dsexp.py -- Explore datasets.

Usage: python3 dsexp.py DATASET

where DATASET is one of sortd, fortum or eco.
"""
import sys
from pathlib import Path
from nilmtk import DataSet
import nilmtk.dataset_converters as ntkdsc
import matplotlib.pyplot as plt

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

for bkey in dataset.buildings:
  building = dataset.buildings[bkey]
  print('\n\n%s\n==== BUILDING %s\n' % ('='*80, bkey))
  print('\n== building')
  print(type(building), building)
  print('\n== building.describe()')
  print(building.describe(compute_expensive_stats=False))

  elec = building.elec
  print('\n\n%s\n==== ELEC for B%s\n' % ('='*80, bkey))
  print('\n== elec')
  print(type(elec), elec)
  print('\n== elec.plot_good_sections()')
  ax = elec.plot_good_sections()
  plt.savefig(dataset_name + '__plot_good_sections.png'); plt.clf()
  #for meter in elec:
  #  print(meter)
  #print('== elec.draw_wiring_graph()')
  #print(elec.draw_wiring_graph())

#elec = dataset.buildings[2].elec
#for meter in elec:
#  print(meter)

sys.exit(0)

elec = dataset.buildings[1].elec
meter = dataset.buildings[1].elec[1]

#print('== elec.meters_directly_downstream_of_mains()')
#print(elec.meters_directly_downstream_of_mains())
# RuntimeError: Tree has more than one root!

print('== meter')
print(meter)
print('== meter.device')
print(meter.device)
print('== meter.get_timeframe()')
print(meter.get_timeframe())
print("== meter.available_ac_types('power')")
print(meter.available_ac_types('power'))
print('== meter.total_energy()')
print(meter.total_energy())
#print('== meter.power_series_all_data().head()')
#print(meter.power_series_all_data().head())
# Takes several seconds.
print('== meter.good_sections()')
# Detect gaps: Identifies pairs of consecutive samples where the time between them is larger than a predefined threshold.
print(meter.good_sections())
# IndexError: only integers, slices (`:`), ellipsis (`...`), numpy.newaxis (`None`) and integer or boolean arrays are valid indices
print('== meter.uptime()')
# Up-time: The total time which a sensor was recording data.
print(meter.uptime())
# Depends on electric.good_sections
print('== meter.dropout_rate()')
# Dropout rate both including and excluding gaps: The total number of recorded samples divided by the number of expected samples.
print(meter.dropout_rate())
print('== meter.diagnose()')
# Diagnose: Checks for all the issues listed above.
#print('== elec.submeters().energy_per_meter()')
#elec.submeters().energy_per_meter()
#KeyError: 'No object named /building1/elec/meter4 in the file'
#print('== meter.proportion_of_energy_submetered()')
#print(meter.proportion_of_energy_submetered())
#AttributeError: 'ElecMeter' object has no attribute 'proportion_of_energy_submetered'
# Proportion of energy sub-metered: Quantifies the proportion of total energy measured by sub-metered channels.
#print('== meter.XXX()')
# Downsample: Down-samples data sets to a specified frequency using aggregation functions such as mean and median.
#print(meter.proportion_of_energy_submetered())
#print('== meter.XXX()')
# Voltage normalisation: Normalises power demands to take into account fluctuations in mains voltage.
print('== elec.select_top_k()')
# Top-k appliances: Identifies the top-k energy consuming appliances.
print(elec.select_top_k())
print('== elec.activity_histogram()')
# ?? Daily appliance usage histograms.
print(elec.activity_histogram())

#https://github.com/nilmtk/nilmtk/issues/355
#https://github.com/nilmtk/nilmtk/blob/master/docs/manual/development_guide/writing_a_disaggregation_algorithm.md