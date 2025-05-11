import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
import scipy as sp
from PIL import Image, ImageDraw, ImageEnhance


SIZE=10
MAXIX = 2

x = list(plt.colormaps)


def f(x, y):
    return np.sin(x) ** 10 + np.cos(10 + y * x)
print(f'{x}')
cmap = plt.colormaps['inferno']
print(f'{cmap}')
w = 128
h = 16
binCount = 84
bins = [4, 15, 23]
rects = []
lastX = np.float64(256)
lastY = np.float64(256)
for bin in bins:
  pts = []
  x1 = np.cos(2*np.pi*(bin/binCount))*w+lastX 
  y1 = np.sin(2*np.pi*(bin/binCount))*w+lastY
  x2 = np.cos(2*np.pi*(bin/binCount) - np.pi)*h+lastX
  y2 = np.cos(2*np.pi*(bin/binCount) - np.pi)*h+lastY
  x3 = np.cos(2*np.pi*(bin/binCount) - np.pi)*h + x1
  y3 = np.cos(2*np.pi*(bin/binCount)- np.pi)*h + y1
  pts.append((lastX, lastY))
  pts.append((x1, y1))
  pts.append((x3,y3))
  pts.append((x2, y2))
  rects.append(pts)
  lastX = x1
  lastY = y1
  h = h / 2
  w = w / 2

def drawRect(dr, pts):
  dr.polygon(pts, fill='red')
  # dr.line((pts[0], pts[1]), fill='red')
  # dr.line((pts[1], pts[3]), fill='red')
  # dr.line((pts[3], pts[2]), fill='red')
  # dr.line((pts[2], pts[0]), fill='red')
   
# X, Y = np.meshgrid(x, y)
# Z = f(X, Y) # temperature
# plt.figsize=(8, 8)
# plt.axis('equal')
X = [pt[0] for pt in pts]
Y = [pt[1] for pt in pts]
img = Image.new(mode='RGB', size=(512, 512))
for rect in rects:
  dr = ImageDraw.Draw(img)
  print(f'rect: {rect}')
  drawRect(dr, rect)

img.show()
img