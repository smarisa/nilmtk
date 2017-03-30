#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
gen.py -- Fake disaggregation data generation tool.

Usage: gen.py METADATA_DIR

Generates fake data for the buildings according to the metadata specified at the
given metadata dir.
"""
# Standard
import sys
from datetime import datetime as pydt
from datetime import timedelta as pytd
from calendar import timegm
from pathlib import Path
from yaml import load
from random import gauss

# Elec meters
DE = {}

class ElecMeter:
  def __init__(self, key, pfd, dts, dte):
    self.key = key
    self.pfd = Path(pfd)
    self.par = None
    self.dm = {}
    self.vsa = []
    self.dts = dts
    self.dte = dte

  def gen(self, error_rate):
    dt = self.dts
    while dt <= self.dte:
      tot = 0
      for sa in self.vsa:
        if sa == 'electric space heater':
          if (dt.hour >= 22 or dt.hour < 8) or dt.minute % 2 == 0:
            tot += 1000
        elif sa == 'electric oven':
          if dt.hour >= 18 and dt.hour < 20:
            tot += 2000
        elif sa == 'fridge':
          if dt.minute % 5 == 0:
            tot += 500
        elif sa == 'light':
          if dt.hour >= 6 and dt.hour < 23:
            tot += 200
      self.mset(dt, tot, error_rate)
      dt += TDIV

  def mset(self, dt, val, error_rate):
    timestamp = timegm(dt.timetuple())
    if self.par:
      self.par.dm[timestamp] += val*(1 + gauss(0.0, error_rate)) - (self.dm[timestamp] if timestamp in self.dm else 0)
    self.dm[timestamp] = val*(1 + gauss(0.0, error_rate))

  def madd(self, dt, val):
    timestamp = timegm(dt.timetuple())
    if self.par:
      self.par.dm[timestamp] += val*(1 + gauss(0.0, error_rate))
    self.dm[timestamp] += val*(1 + gauss(0.0, error_rate))

  def write(self):
    if not self.pfd.parent.exists():
      self.pfd.parent.mkdir(parents=True)
    vm = [(ts, val) for ts, val in self.dm.items()]
    with self.pfd.open('w') as fd:
      for ts, val in sorted(vm):
        fd.write('%d,%d\n' % (ts, val))

try:
  assert len(sys.argv) >= 2
  pdm = Path(sys.argv[1])
  error_rate = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
  print('Loading metadata from %s...' % (pdm))
  vpfmb = [pfmb for pfmb in pdm.glob('building*.yaml')]
  with (pdm / 'meter_devices.yaml').open('r') as src:
    dmm = load(src)
except:
  print(__doc__)
  sys.exit(1)

# Measurement interval, Hz
FREQ = dmm['FakeMeter']['sample_period']
TDIV = pytd(seconds=FREQ)

for pfm in vpfmb:
  with pfm.open('r') as src:
    dm = load(src)
  tf = dm['timeframe']
  dts = pydt.strptime(tf['start'], "%Y-%m-%dT%H:%M:%S%z")
  dte = pydt.strptime(tf['end'], "%Y-%m-%dT%H:%M:%S%z")
  for key, de in dm['elec_meters'].items():
    DE[key] = ElecMeter(key, pdm.parent / de['data_location'], dts, dte)
    if 'submeter_of' in de:
      DE[key].par = DE[de['submeter_of']]
  for da in dm['appliances']:
    for sm in da['meters']:
      DE[sm].vsa.append(da['type'])
  for em in DE.values():
    print('Generating fake data for meter %s with error rate %.3f...' % (em.key, error_rate))
    em.gen(error_rate)
  for em in DE.values():
    print('Writing the data for meter %s...' % (em.key))
    em.write()

# EOF
