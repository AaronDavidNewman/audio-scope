import numpy as np
import os
import pyaudio
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
from io import BytesIO

def createBlurredCircle(radius, fn):
  xar = np.array([0.5])
  yar = np.array([0.5])
  # fig, af = plt.subplots(figsize=(7, 7), subplot_kw={'projection': 'polar'})
  fig, af = plt.subplots(figsize=(7, 7))
  fig.set_frameon(False)
  dpi = fig.figure.get_dpi()
  fig.set_size_inches(512/dpi, 512/dpi)
  af.set_axis_off()
  af.set_xlim(0, 1)
  af.set_ylim(0, 1)
  af.scatter(xar, yar,color=np.array((0, 0, 0)).T, sizes=np.array([radius*dpi]))
  buffer = BytesIO()
  plt.savefig(buffer, format='png')
  with Image.open(buffer) as img:
    im_blurred = img.filter(filter=ImageFilter.GaussianBlur(radius = radius/8.0))
    offset1 = 20

    offset1 = -1 * offset1
    im_blurred = im_blurred.resize((offset1 + 512, offset1 + 512))
    im_blurred = im_blurred.crop((2 * offset1, 2 * offset1, 512 - offset1 * 2, 512 - offset1 * 2))
    im_blurred.save(fn)

createBlurredCircle(256, 'blorf.png')    

