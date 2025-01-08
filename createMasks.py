import sys
import numpy as np
import base64
import os
from io import BytesIO
import pyaudio
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
from scipy.fft import rfft, fft
from scipy.io import wavfile
from tkinter import TclError
from matplotlib.backend_bases import MouseButton
from matplotlib.transforms import blended_transform_factory

# Calculate the frequency of each note, in HZ, through the range of octaves
def calculateNoteFrequencies(ova, ar, namear):
    for oo in range(ova):
      base = START_FREQ * np.pow(2, oo)
      for step in range(STEPS_PER_OCTAVE):
          ix = STEPS_PER_OCTAVE * oo + step
          oolbl = oo + 1
          # by convention, octaves start at C, ie. A1 is above C1, so compensate here
          if step > 3:
             oolbl = oolbl + 1
          noteName = f'{LETTER_NAMES[step]} {oolbl}'
          namear.append(noteName)
          freq = base * np.pow(HALF_STEP_MANTISSA, step)
          ar[ix] = freq

# put indices of specific notes into output, based on an FFT
# of windowSize, with sampling rate sampleRate, and noteFreq array
# as input, containing the frequencies of the notes we care about
def calculateNoteIndices(sampleRate, windowSize, noteFrequencies, output):
  nfix = 0
  fq = 0
  fdelta = sampleRate / windowSize  # frequency steps for each fft window, window (n) is fdelta*n hz    
  while nfix < noteFrequencies.shape[0]:
    fdiff = np.abs((fq * fdelta) - noteFrequencies[nfix])  
    prevdiff = fdiff
    while fq < windowSize // 2 and prevdiff >= fdiff:
      fq = fq + 1
      prevdiff = fdiff
      fdiff = np.abs((fq * fdelta) - noteFrequencies[nfix])
    fq = fq - 1
    output.append(fq)
    nfix += 1


def createBlurredCircle(radius):
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
  colors = np.ones(1)
  af.scatter(xar, yar,color=np.array((colors, colors, colors)).T, sizes=np.array([radius*dpi]))
  buffer = BytesIO()
  plt.savefig(buffer, format='png')
  plt.close()
  with Image.open(buffer) as img:
    blurRadius = np.max(np.array([radius/16.0, 2]))
    im_blurred = img.filter(filter=ImageFilter.GaussianBlur(radius = blurRadius))
    blurbuf = BytesIO()
    im_blurred.save(blurbuf, format='png')
    img_data = base64.b64encode(blurbuf.getvalue())
    return img_data.decode('utf-8')

START_FREQ = 55  # A0
STEPS_PER_OCTAVE = 12
LETTER_NAMES = ['A', 'Bb', 'B' ,'C', 'Db', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab']
HALF_STEP_MANTISSA = np.pow(2, 1./12)     # equal tempermant, each note is octavebase * this
IMAGESIZE=512
MASKRESIZE=1.05

def getSeconds(filename):
  sample_rate, audio = wavfile.read(filename)
  lchannel, rchannel = audio.T
  seconds = lchannel.shape[0] / sample_rate
  return seconds
   
def createFftMasks(filename):
  # filename="air-breath2.wav"
  folder = './masks/'
  # filename="poly1.wav"
  #  filename="noogarpy.wav"
  windowSize = np.pow(2, 14)  # FFT window size, 8192 == 2^13.  Must be < sample_rate // 2
  octaves = 7   # from a1 to g# 7
  note_count = octaves * STEPS_PER_OCTAVE
  sample_rate, audio = wavfile.read(filename)
  lchannel, rchannel = audio.T
  seconds = lchannel.shape[0] / sample_rate
  print(f'sample rate is {sample_rate}, recording is {seconds} seconds')
  soundNorm = np.pow(2, 16)  # we assume input is 2^32 bit samples, we graph wavform on a 0-2^16 scale
  notes = np.zeros(octaves * STEPS_PER_OCTAVE)  # will contain frequencies for notes, one per half-step
  # name the notes, for labelling the axis
  noteNames = []

  calculateNoteFrequencies(octaves, notes, noteNames)
  notesix = []  # will contain index of the FFT window for each note
  calculateNoteIndices(noteFrequencies=notes, output=notesix, sampleRate=sample_rate, windowSize = windowSize)
  # the graphs, one for the waveform and one for the frequencies
  for fn in os.listdir(folder):
    filepath = os.path.join(folder, fn)
    if os.path.isfile(filepath):
        os.unlink(filepath)
  # Normalize the input samples to 16-bit numbers to avoid exponentials
  # calculate abs value of each channel
  lchannel = np.sqrt(np.int64(lchannel)** 2)/soundNorm
  rchannel = -1 * np.sqrt(np.int64(rchannel)** 2)/soundNorm
  maxEnergy = 0
  samples = len(lchannel)
  npix = 0
  # go through the samples 1 time to get max energy
  while npix < (samples - windowSize):
    sll =  np.mean(lchannel[npix: (npix + windowSize)])
    slr =  np.mean(rchannel[npix: (npix + windowSize)])
    maxSample = np.max(np.array(np.mean(np.abs(sll)), np.mean(np.abs(slr))))
    if maxSample > maxEnergy:
      maxEnergy = maxSample
    npix = npix + windowSize

  maxEnergy = np.max(np.array([maxEnergy, 1]))
  xformx = np.linspace(0, 2 * np.pi, len(notesix))
  sample = 0
  fignum = 1
  imgBufs = [None, None, None, None, None]
  def slideImgBufs():
    ll = len(imgBufs)
    for i in range(ll - 1):
      imgBufs[i + 1] = imgBufs[i]
  
  slidingWindowSize=5
  def blendBufs(buf1, buf2):
    with Image.open(buf1) as mask1, Image.open(buf2) as mask2:
      mask2 = mask2.resize((int(MASKRESIZE * IMAGESIZE), int(MASKRESIZE* IMAGESIZE)))
      offset = int(MASKRESIZE * IMAGESIZE) - IMAGESIZE
      mask2 = mask2.crop((offset, offset, IMAGESIZE + offset, IMAGESIZE + offset))
      blended = Image.blend(mask1, mask2, 0.8)
      return blended
  runningWindows = np.zeros((slidingWindowSize, len(notesix)))
  runningWindowIx = 0
  while sample < samples - windowSize:
    if fignum % 100 == 0:
      print(f'printing mask {fignum}')
    fig, af = plt.subplots(figsize=(7, 7), subplot_kw={'projection': 'polar'})
    fig.set_frameon(False)
    dpi = fig.figure.get_dpi()
    fig.set_size_inches(512/dpi, 512/dpi)
    af.set_axis_off()
    af.set_ylim(0, 1)
    slc = (lchannel[sample: (sample + windowSize)] + np.abs(rchannel[sample: (sample + windowSize)]))/2
    xform = fft(slc)
    xformy = np.abs(xform)[1: 1 + (windowSize // 2)]
    yvals=np.zeros(note_count)
    yaggr = np.zeros(note_count)
    for yix in range(0, len(notesix)):
        yvals[yix] = (xformy[notesix[yix]])
    yValMax = np.max(np.array([1, np.max(yvals)]))
    energy = np.mean(np.abs(slc))/maxEnergy
    yvals = (energy * yvals) / yValMax  
    runningWindows[runningWindowIx] = yvals
    for rix in range(slidingWindowSize):
      yaggr = yaggr + runningWindows[rix]
    af.set_ylim(0, 1)
    # axis.plot(xformx, yvals)
    # axis.scatter(xformx, yvals)
    colors = np.where(yaggr > 1, 0, 1 - yaggr)
    af.bar(xformx, yaggr, color=np.array((colors, colors, colors)).T)
    figstr = f'{fignum:04d}'
    filename = f'{folder}figure_{figstr}'
    buf = BytesIO()
    slideImgBufs()
    fig.savefig(buf, format='PNG')
    imgBufs[0] = buf
    if imgBufs[4] != None:
      img1 = blendBufs(imgBufs[3], imgBufs[4])
      img2 = blendBufs(imgBufs[1], imgBufs[2])
      buf1 = BytesIO()
      buf2 = BytesIO()
      buf3 = BytesIO()
      img1.save(buf1, format='PNG')
      img2.save(buf2, format='PNG')
      img3 = blendBufs(buf1, buf2)
      img3.save(buf3, format='PNG')
      buf4 = blendBufs(imgBufs[0], buf3)
      buf4.save(f'{filename}.png')
    else:
      sv = Image.open(buf)
      sv.save(f'{filename}.png')
    plt.close()
    sample += windowSize//2
    fignum += 1
    runningWindowIx = (1 + runningWindowIx) % slidingWindowSize

## cc = createBlurredCircle(512)
## buf = base64.b64decode(cc)
## img = Image.open(BytesIO(buf))
## img.save('blorf.png')

