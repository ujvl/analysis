from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import sys

if (len(sys.argv) == 1):
    print ("Usage: cdf.py [file-name] [xlim-low] [xlim-high]")
    sys.exit(0)

filename = sys.argv[1]
col = 0

data = np.loadtxt(fname=filename, usecols=(col,))

counts, bin_edges = np.histogram(data, bins=50, normed=True)
cdf = np.cumsum(counts)
cdf /= cdf[-1]

title = input("Title: ")
x_label = input("x-axis label: ")
y_label = "F(x)"

xlow =np.min(bin_edges[1:]) 
xhi = np.max(bin_edges[1:])

plt.title(title)
plt.xlabel(x_label)
plt.ylabel(y_label)
plt.plot(bin_edges[1:], cdf)
plt.show()
