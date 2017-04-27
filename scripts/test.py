#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
#
# For documentation and other information (including about licensing),
# read the README at the project root.

# Standard
import pandas as pd
import numpy as np

ts = [1, 2, 3, 4, 5, 6]

appliance_powers_dict = {}
for app in 'A', 'B', 'C':
  predicted_power = [500]*len(ts)
  column = pd.Series(predicted_power, index=ts, name=app)
  appliance_powers_dict[app] = column
appliance_powers = pd.DataFrame(appliance_powers_dict, dtype='float32')
print(appliance_powers)

# EOF