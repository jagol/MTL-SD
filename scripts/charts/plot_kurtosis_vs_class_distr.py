import json

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from matplotlib import gridspec

spec = gridspec.GridSpec(ncols=2, nrows=1,
                         width_ratios=[2, 1])

arc_data = []
with open('data/arc/train.jsonl') as fin:
    for line in fin:
        label = json.loads(line)['label_orig']
        arc_data.append(label)

PERSPECTRUM_data = []
with open('data/PERSPECTRUM/train.jsonl') as fin:
    for line in fin:
        label = json.loads(line)['label_orig']
        PERSPECTRUM_data.append(label)

# [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
# bins = [i + 0.3 for i in range(7)]
# bins = [-0.3, 0.3, 1.3, 2.3, 3.8, 4.8, 5.3]
# plt.hist([arc_data, PERSPECTRUM_data],
#          density=True,
#          # edgecolor='k',
#          width=1,
#          align='mid',
#          bins=bins
#          )
# plt.ylabel('Relative Frequency')
# plt.xlabel('ARC Labels')
# plt.show()

fig = plt.figure()
ax1 = fig.add_subplot(spec[0])
ax2 = fig.add_subplot(spec[1])

# fig, (ax1, ax2) = plt.subplots(1, 2, sharey='all')
# fig.suptitle('Kurtosis of Different Class Distributions')

ax1.hist(arc_data, density=True, bins=[-0.5, 0.5, 1.5, 2.5, 3.5], rwidth=0.9, color='#ff9999')
ax1.set_ylabel('Relative Frequency')
ax1.set_xlabel('Kurtosis: 2.3')
ax1.set_title('ARC')

ax2.hist(PERSPECTRUM_data, density=True, bins=[-0.5, 0.5, 1.5], rwidth=0.9, color='#66b3ff')
ax2.set_xlabel('Kurtosis: 1.0')
ax2.set_title('PERPSECTRUM')
ax2.sharey(other=ax1)

ax2.xaxis.set_tick_params(labelsize=9)

plt.savefig('../images/arc_PERPSECTRUM_kurtosis.png')
