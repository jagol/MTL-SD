import matplotlib.pyplot as plt

from compute_class_distr import compute_kurtosis


data = {
    1: compute_kurtosis([1]),
    2: compute_kurtosis([1, 2]),
    3: compute_kurtosis([1, 2, 3]),
    4: compute_kurtosis([1, 2, 3, 4])
}


x = sorted(data.keys())
y = [data[f] for f in x]


plt.bar(x, y)
# plt.plot(x, y)
plt.show()
