import matplotlib.pyplot as plt

fig1, ax1 = plt.subplots()  # figsize=(10, 5)
fig1.set_size_inches(12, 5, forward=True)
# ax = fig.add_axes([0, 0, 1, 1])
# plt.figure(figsize=(20, 3))  # width:20, height:3

datasets = ['arc', 'ArgMin', 'FNC-1', 'IAC', 'IBMCS', 'PERPSECTRUM',
            'SCD', 'SE2016T6', 'SE2019T7', 'Snopes']
kurtosis = [2.33, 1.00, 2.22, 1.5, 1.0, 1.0, 1.0, 1.5, 2.33, 1.00]
# x_spaces = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
# x_spaces = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
# plt.xticks(x_spaces, datasets)
ax1.bar(datasets, kurtosis, color='#66b3ff')
plt.ylabel('kurtosis')

ax1.axhline(y=1.5, linewidth=1, color='k')

plt.savefig('../images/kurtosis_sd_benchmark.png')
