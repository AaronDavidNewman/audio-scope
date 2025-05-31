import numpy as np
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
import random
binCount = 82
histSize = 5
binHist = []
valHist = []
binVals = []
binHistIx = 0
for i in range(histSize):
  bins =[]
  binvals = []
  for j in range(5):
    bins.append(random.randint(0, binCount))
    binvals.append(random.random())
  binHist.append(bins)
  valHist.append(binvals)
X = np.zeros((12 + 1, 12 + 1))

colormap =  [(ar[0], ar[1], ar[2]) for ar in plt.colormaps['tab20c'](range(0,binCount+1))]
fig = plt.figure(figsize=(5.334, 5.334), dpi=96)
a = fig.add_axes([0,0,1,1])
a.set_xlim(12)
a.set_ylim(12)
a.set_axis_off()
for i in range(histSize):
  bins = binHist[i]
  binvals = valHist[i]
  for j in range(len(bins)):
    bc = bins[j]
    val = binvals[j]
    yval = np.int32(12 * val)
    X[bc % 12, np.min((12, yval)) ] += bc

print(f'{np.min(X)} and {np.max(X)}')
a.imshow(X, cmap='YlGn')
c1 = np.linspace(0, np.pi * 2, 1000)
c2 = np.linspace(0, np.pi *14, 1000)
r1=5
r2=3
for i in range(histSize):
  bins = binHist[i]
  binvals = valHist[i]
  for j in range(len(bins)):
    b = fig.add_axes([0,0,1,1])
    b.set_axis_off()
    bc = np.float32(bins[j])/binCount
    val = binvals[j]
    yval = 12 * val
    XX = (np.array(r1*np.cos(c1) + bc*yval*np.cos(c2)))*val
    YY = (np.array(r1*np.sin(c1) + bc*yval*np.sin(c2)))*val
    b.plot(XX,YY, color=colormap[np.int32(yval)])
plt.show()


