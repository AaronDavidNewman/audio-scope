import numpy as np
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
binCount = 82
bins = [14, 4, 24, 7, 18]
lastX = 256
lastY = 256
w=128
angle = 2*np.pi
h = 12

pts=[]
rects=[]
fig,a = plt.subplots()
pts.append((lastX, lastY))
allLcs = []
segs = np.array(np.empty)
widths = np.array(np.empty)
segcount = 100
for bin in bins:
  div = bin/binCount
  x1 = np.cos(angle*(div))*(w)+lastX
  y1 = np.sin(angle*(div))*(w)+lastY
  xxs = np.linspace(lastX, x1, segcount)
  yys = np.linspace(lastY, y1, segcount)
  lasth = h
  lastw = w
  h = 2*h/3
  w = w/2
  pts1 = np.array((xxs, yys)).T.reshape(-1, 1, 2)
  if len(segs.shape) == 0:
    segs = np.concatenate([pts1[:-1], pts1[1:]], axis=1)
    widths = 1+np.linspace(lasth, h, segcount)
  else:
    newsegs = np.concatenate([pts1[:-1], pts1[1:]], axis=1) 
    widths =  np.concatenate((widths, 1+np.linspace(h, h-2, segcount)))

    print(f'{segs.shape} {newsegs.shape}')
    segs = np.concatenate((segs, newsegs))
  lastX = np.cos(angle*(div))*lastw+lastX
  lastY = np.sin(angle*(div))*lastw+lastY
  # segcount = np.int64(4*segcount/5)
  # print(f'calc shape {segs.shape} yys shape {yys.shape} pts {pts1.shape}')

lc = LineCollection(segs, linewidths=widths,color='blue')
lc.set_capstyle('round')
print(f'lcshape {segs.shape}')
a.add_collection(lc)


x = np.linspace(0,4*np.pi,100)
y = np.cos(x)
lwidths=1+x[:-1]
ar = np.array([x,y])
art = ar.T
arts = art.reshape(-1, 1, 2)
print(f'ar: {ar.shape} art: {art.shape}  arts: {arts.shape} arts5s:{arts[5].shape}')
print(f'arts 5, 0, 1: {arts[5][0][1]} args5: {arts[5]} ar0: {art[0]}')
points = np.array([x, y]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)
print(f'segs: {segments.shape}')
a.set_xlim(0,512)
a.set_ylim(0,512)
plt.show()