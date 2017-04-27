#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""resample.py -- Data resampling tool for disaggregation research.

Usage: resample.py DURATION DATA_FILES+

Converts CSV files of the form

  1456790400,1474
  1456790460,1037
  1456790520,1044
  1456790580,1046
  ...

Into the same form but resampling (aggregating) the data to the given sample
rate. For example the above with a 120 s sample rate would become:

  1456790430,1256
  1456790550,1045
  ...
"""

# Standard
import sys
from pathlib import Path

try:
  assert len(sys.argv) >= 2
  td = float(sys.argv[1])
  vpf = [Path(pf) for pf in sys.argv[2:]]
except:
  print(__doc__)
  sys.exit(1)

for pf in vpf:
  with pf.open('r') as stream:
    fsout = (pf.parent / ('%s-d%d%s' % (pf.stem, td, pf.suffix))).open('w')
    print('Resampling "%s" to %d...' % (pf, td))
    ts1 = 0
    vv = []
    for ln in stream:
      st, sv = ln.split(',')
      if float(st) - ts1 >= td:
        if len(vv) > 0:
          fsout.write('%d,%.2f\n' % (ts1 + td/2, sum(vv)/len(vv)))
        ts1 = float(st)
        vv = []
      vv.append(float(sv))
    if len(vv) > 0:
      fsout.write('%d,%d\n' % (ts1 + td/2, sum(vv)/len(vv)))
    fsout.close()

# EOF
