#!/usr/bin/env python3
"""
dsexp.py -- Explore datasets.

Usage: python3 dsexp.py DATASET

where DATASET is one of sortd, fortum or eco.

Run "source ./bashrc.sh" to setup the Python environment (works for Samuel).
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

for bkey in dataset.buildings:
  building = dataset.buildings[bkey]
  print('\n\n%s\n==== BUILDING building%s\n' % ('='*80, bkey))
  print('\n== building')
  print(type(building), building)
  print('\n== building.describe()')
  try: print(building.describe(compute_expensive_stats=False))
  except: print('Failed!')

  elec = building.elec
  print('\n\n%s\n==== ELEC building%s/elec\n' % ('='*80, bkey))
  print('\n== elec')
  print(type(elec), elec)
  print('\n== elec.label()')
  # Returns a string listing all the appliance types.
  print(elec.label())
  print('\n== elec.dominant_appliance()')
  # Returns the most dominant appliance in the metergroup if any.
  try:
    print(elec.dominant_appliance())
  except:
    print('No dominant appliance found!')
  print('\n== elec.dominant_appliances()')
  # Returns the most dominant appliance or None for each submeter in the metergroup.
  print(elec.dominant_appliances())
  #print('\n== elec.meters_directly_downstream_of_mains()')
  #print(elec.meters_directly_downstream_of_mains())
  # RuntimeError: Tree has more than one root!
  #print('\n== elec.plot_good_sections()')
  #ax = elec.plot_good_sections()
  #plt.savefig('results/%s__plot_good_sections.png' % (dataset_name)); plt.clf()
  #print('\n== elec.plot_power_histogram()')
  #ax = elec.plot_power_histogram()
  #plt.savefig('results/%s__plot_power_histogram.png' % (dataset_name)); plt.clf()
  # --> ValueError: 'physical_quantity' is not in list
  #print('\n== elec.draw_wiring_graph()')
  #print('\n== elec.draw_wiring_graph()')
  #print(elec.draw_wiring_graph())
  print('\n== elec.available_physical_quantities()')
  print(elec.available_physical_quantities())
  for apq in elec.available_physical_quantities():
    print('\n== elec.available_ac_types(%s)' % (apq))
    print(elec.available_ac_types(apq))
  print('\n== elec.is_site_meter()')
  print(elec.is_site_meter())
  print('\n== elec.submeters().select_top_k(k=3)')
  # Top-k appliances: Identifies the top-k energy consuming appliances.
  print(elec.submeters().select_top_k(k=3))
  #print('\n== elec.correlation_of_sum_of_submeters_with_mains()')
  #print(elec.correlation_of_sum_of_submeters_with_mains())
  # --> KeyError: 'Level physical_quantity not found'
  #print('\n== elec.plot()')
  # Plots the submeters with respect to power and time.
  #ax = elec.plot()
  #plt.savefig('results/%s__elec_plot.png' % (dataset_name)); plt.clf()
  #print('\n== elec.activity_histogram()')
  # ?? Daily appliance usage histograms.
  #print(elec.activity_histogram())

  for meter in elec.meters:
    print('\n\n%s\n==== METER building%s/elec/%s\n' % ('='*80, meter.building(), meter.instance()))
    print('\n== meter')
    print(meter)
    print('\n== meter.device')
    print(meter.device)
    print('\n== meter.label()')
    # Returns a string describing this meter.
    print(meter.label())
    print('\n== meter.get_timeframe()')
    print(meter.get_timeframe())
    print('\n== meter.is_site_meter()')
    print(meter.is_site_meter())
    print('\n== meter.available_columns()')
    print(meter.available_columns())
    print('\n== meter.available_physical_quantities()')
    print(meter.available_physical_quantities())
    for apq in meter.available_physical_quantities():
      print('\n== meter.available_ac_types(%s)' % (apq))
      # Finds available alternating current types for a specific physical quantity.
      print(meter.available_ac_types(apq))
    print('\n== meter.dominant_appliance()')
    print(meter.dominant_appliance())
    print('\n== meter.total_energy()')
    print(meter.total_energy())
    #print('\n== meter.power_series_all_data().head()')
    #print(meter.power_series_all_data().head())
    # Takes several seconds.
    print('\n== meter.good_sections()')
    # Detect gaps: Identifies pairs of consecutive samples where the time between them is larger than a predefined threshold.
    print(meter.good_sections())
    # IndexError: only integers, slices (`:`), ellipsis (`...`), numpy.newaxis (`None`) and integer or boolean arrays are valid indices
    print('\n== meter.uptime()')
    # Up-time: The total time which a sensor was recording data.
    print(meter.uptime())
    # Depends on electric.good_sections
    print('\n== meter.dropout_rate()')
    # Dropout rate both including and excluding gaps: The total number of recorded samples divided by the number of expected samples.
    print(meter.dropout_rate())
    #print('\n== elec.submeters().energy_per_meter()')
    #elec.submeters().energy_per_meter()
    #KeyError: 'No object named /building1/elec/meter4 in the file'
    #print('\n== meter.proportion_of_energy_submetered()')
    #print(meter.proportion_of_energy_submetered())
    #AttributeError: 'ElecMeter' object has no attribute 'proportion_of_energy_submetered'
    # Proportion of energy sub-metered: Quantifies the proportion of total energy measured by sub-metered channels.
    #print('\n== meter.XXX()')
    # Downsample: Down-samples data sets to a specified frequency using aggregation functions such as mean and median.
    #print(meter.proportion_of_energy_submetered())
    #print('\n== meter.XXX()')
    # Voltage normalisation: Normalises power demands to take into account fluctuations in mains voltage.
    def print_chunks(chunk_generator):
      i = 1
      while True:
        try:
          print('Loading chunk %d...' % (i))
          chunk = next(chunk_generator)
        except StopIteration:
          print('No more chunks!')
          break
        else:
          print(chunk)
          i += 1
    print('\n== meter.load()')
    # Load data
    print_chunks(meter.load())
    for td in 3600, 86400:
      print('\n== meter.load(sample_period=%d)' % (td))
      # Load data
      print_chunks(meter.load(sample_period=td))

print('\n== FINISHED!')

#https://github.com/nilmtk/nilmtk/issues/355
#https://github.com/nilmtk/nilmtk/blob/master/docs/manual/development_guide/writing_a_disaggregation_algorithm.md