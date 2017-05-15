#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""fortum_data_converter.py -- Fortum dataset data converter

Usage: python3 convert.py CSV+

Each given CSV argument must be a csv file with the following format:

    DeviceId;TypeId;ValueTimeStamp;DataValue
    20;1;2016-03-05 08:00:14;9.1
    20;2;2016-03-05 08:00:14;1.8
    20;3;2016-03-05 08:00:14;0.5
    ...

The script groups phase current data for each device and saves the data along
with the aggregate power using the formula 230*(p1+p2+p3) into four files under
a directory tree with format building<DeviceId>/elec. The phase data is saved
in files p[123].csv with format

    timestamp,current
    1459011417,14.500
    1459011418,14.500
    1459011419,14.500
    ...

and the power is saved into file ap.csv with format

    timestamp,power
    1459011417,9000
    1459011418,9150
    1459011419,9050
    ...
"""

# Standard
import sys
import os.path
from datetime import datetime as pydt
from calendar import timegm
from time import time

phase_files = {}
power_files = {}
phase_cache = {}

if len(sys.argv) <= 1:
  print(__doc__)
  sys.exit(1)

if sys.argv[1] in ('-h', '--help'):
  print(__doc__)
  sys.exit(0)

ts0 = time()
def pinfo(msg):
  print('%08.3f: %s' % (time() - ts0, msg))

for arg in sys.argv[1:]:
  csv_file = arg
  with open(csv_file, 'r') as stream:
    heading = stream.readline().strip()
    if heading != 'DeviceId;TypeId;ValueTimeStamp;DataValue':
      pinfo('Invalid file format: %s!' % (heading))
      sys.exit(1)
    pinfo('Starting to process input file "%s"...' % (csv_file))
    for ln in stream:
      device_id, type_id, datetime_string, current = ln.split(';')
      building_id = int(device_id)
      phase = int(type_id)
      datetime = pydt.strptime(datetime_string, "%Y-%m-%d %H:%M:%S")
      timestamp = timegm(datetime.timetuple())
      current = float(current)
      # Ensure base dir for the building's time series exists
      building_elec_dir = 'building%d/elec' % (building_id)
      if not os.path.exists(building_elec_dir):
        os.makedirs(building_elec_dir)
      # Write the phase data into the correct phase file
      phase_file = '%s/p%d.csv' % (building_elec_dir, phase)
      if phase_file not in phase_files:
        phase_files[phase_file] = f = open(phase_file, 'a')
        f.write('timestamp,current\n')
        pinfo('New phase file "%s" created!' % (phase_file))
      phase_files[phase_file].write('%d,%.1f\n' % (timestamp, current))
      # Ensure aggregate power is computed and written when each phase value for
      # the time instant is read
      if building_id not in phase_cache:
        phase_cache[building_id] = {}
      if timestamp not in phase_cache[building_id]:
        phase_cache[building_id][timestamp] = {}
      phase_cache[building_id][timestamp][phase] = current
      # Flush the phase cache into the power time series file if all phases are
      # available
      if len(phase_cache[building_id][timestamp]) == 3:
        power_file = '%s/ap.csv' % (building_elec_dir)
        if power_file not in power_files:
          power_files[power_file] = f = open(power_file, 'a')
          f.write('timestamp,power\n')
          pinfo('New power file "%s" created!' % (power_file))
        d = phase_cache[building_id][timestamp]
        power = 230 * (d[1] + d[2] + d[3])
        power_files[power_file].write('%d,%d\n' % (timestamp, power))
        del phase_cache[building_id][timestamp]

# Close all opened file descriptors
open_files = {}
open_files.update(phase_files)
open_files.update(power_files)
for file_path, fd in open_files.items():
  pinfo('Closing file "%s"...' % (file_path))
  fd.close()

# EOF
