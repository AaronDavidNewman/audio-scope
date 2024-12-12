import sys
import numpy as np
import os
import pyaudio
import matplotlib.pyplot as plt
from scipy.fft import rfft, fft
from scipy.io import wavfile
from tkinter import TclError
from matplotlib.backend_bases import MouseButton
from matplotlib.transforms import blended_transform_factory

def plotSpectrum(x):
        global colorWindowIndex
        sample = int((x / seconds) * samples)
        slc = (lchannel[sample: (sample + windowSize)] + np.abs(rchannel[sample: (sample + windowSize)]))/2
        xform = fft(slc)
        xformy = np.abs(xform)[1: 1 + (windowSize // 2)]
        yvals=np.zeros(note_count)
        for yix in range(0, len(notesix)):
           yvals[yix] = (xformy[notesix[yix]])
        yValMax = np.max(yvals)
        energy = np.mean(np.abs(slc))/maxEnergy
        yvals = (energy * yvals) / yValMax
        af.clear()
        af.set_ylim(0, 1)
        af.plot(xformx, yvals)
        af.set_ylabel(f'spectrum at {x} seconds')
        top = np.argsort(-1 * yvals)[:5]
        for t in top:
          if yvals[t] > nameThreshold:
            af.annotate(noteNames[t], xy=(xformx[t], yvals[t]))
           
        fig.canvas.draw()
        return af

def on_click(event):    
    if event.button is MouseButton.LEFT:
      plotSpectrum(event.xdata)

# Calculate the frequency of each note, in HZ, through the range of octaves
def calculateNoteFrequencies(ova, ar, namear):
    for oo in range(ova):
      base = START_FREQ * np.pow(2, oo)
      for step in range(STEPS_PER_OCTAVE):
          ix = STEPS_PER_OCTAVE * oo + step
          oolbl = oo
          # by convention, octaves start at 2, ie. A1 is above C1, so compensate here
          if step > 3:
             oolbl = oolbl + 1
          noteName = f'{LETTER_NAMES[step]} {oolbl}'
          namear.append(noteName)
          freq = base * np.pow(HALF_STEP_MANTISSA, step)
          ar[ix] = freq

def calculateNoteIndices(noteFreq, ar):
  nfix = 0
  fq = 0
  while nfix < noteFreq.shape[0]:
    fdiff = np.abs((fq * fdelta) - noteFreq[nfix])  
    prevdiff = fdiff
    while fq < windowSize // 2 and prevdiff >= fdiff:
      fq = fq + 1
      prevdiff = fdiff
      fdiff = np.abs((fq * fdelta) - noteFreq[nfix])
    fq = fq - 1
    ar.append(fq)
    nfix += 1

START_FREQ = 55  # A0
STEPS_PER_OCTAVE = 12
LETTER_NAMES = ['A', 'Bb', 'B' ,'C', 'Db', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab']
HALF_STEP_MANTISSA = np.pow(2, 1./12)     # equal tempermant, each note is octavebase * this

filename="air-breath2.wav"
# filename="poly1.wav"
#  filename="noogarpy.wav"
# filename="ween24-2-softer.wav"

figure_width = 5
windowSize = np.pow(2, 14)  # FFT window size, 8192 == 2^13.  Must be < sample_rate // 2
octaves = 7   # from a0 to g# 7
note_count = octaves * STEPS_PER_OCTAVE
sample_rate, audio = wavfile.read(filename)
lchannel, rchannel = audio.T
seconds = lchannel.shape[0] / sample_rate
print(f'sample rate is {sample_rate}, recording is {seconds} seconds')
soundNorm = np.pow(2, 16)  # we assume input is 2^32 bit samples, we graph wavform on a 0-2^16 scale
fdelta = sample_rate / windowSize  # frequency steps for each fft window, window (n) is fdelta*n hz
notes = np.zeros(octaves * STEPS_PER_OCTAVE)  # will contain frequencies for notes, one per half-step
# name the notes, for labelling the axis
noteNames = []
nameThreshold = 0.1

calculateNoteFrequencies(octaves, notes, noteNames)
notesix = []  # will contain index of the FFT window for each note
calculateNoteIndices(notes, notesix)
# the graphs, one for the waveform and one for the frequencies
fig = plt.figure(figsize = (figure_width, figure_width), layout='constrained')
spec = fig.add_gridspec(ncols=2, nrows=1)
ax = fig.add_subplot(spec[0,0])
af = fig.add_subplot(spec[0, 1])

# Normalize the input samples to 16-bit numbers to avoid exponentials
# calculate abs value of each channel
lchannel = np.sqrt(np.int64(lchannel)** 2)/soundNorm
rchannel = -1 * np.sqrt(np.int64(rchannel)** 2)/soundNorm
maxEnergy = 0
samples = len(lchannel)
# we can't display every sample so just display the waveform at FFT-window intervals
ampWindowSize = np.pow(2, 10)
windows = samples/ampWindowSize
xscale = np.linspace(0, seconds, int(windows))
npix = 0
soundl = []
soundr = []
xformx = np.linspace(0, len(notesix), len(notesix))

# we display the running average at each amplitude window.
while npix < (samples - ampWindowSize):
  sll =  np.mean(lchannel[npix: (npix + ampWindowSize)])
  slr =  np.mean(rchannel[npix: (npix + ampWindowSize)])
  maxSample = np.max(np.array(np.mean(np.abs(sll)), np.mean(np.abs(slr))))
  if maxSample > maxEnergy:
     maxEnergy = maxSample
  npix = npix + ampWindowSize
  soundl.append(np.mean(sll))
  soundr.append(np.mean(slr))


ax.plot(xscale, soundl, label='left channel')
ax.plot(xscale, soundr, label='right channel')
# plotSpectrum(0.5)
plt.connect('button_press_event', on_click)
plt.show()


