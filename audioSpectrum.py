import numpy as np
from scipy.fft import rfft, fft
from scipy.signal import find_peaks, peak_prominences,peak_widths
from noteFreq import NoteFrequency
from audioFrequency import AudioFrequency
import matplotlib.pyplot as plt
from impulse import Impulse

class AudioSpectrum:
  fftWindow = 16384
  impulseWindow = 32
  beatWindowSize = 4096
  def __init__(self, lbuffer, rbuffer, sampleRate):
    self.lbuffer  = lbuffer
    self.rbuffer = rbuffer
    self.sampleRate = sampleRate
    self.noteData = NoteFrequency()
    self.currentSample = 0
    self.audioBufs = []
    self.impulse = Impulse(self.impulseWindow)
    self.firstnz = np.min([np.nonzero(self.lbuffer)[0][0],np.nonzero(self.rbuffer)[0][0]])
    for pitchIndex in range(len(self.noteData.frequencies)):
      freq = self.noteData.frequencies[pitchIndex]
      # windowIndex*(sampleRate/windowSize) = frequency, so 
      # windowIndex = (windowSize/sampleRate)*frequency
      binIx = np.int64((self.fftWindow / self.sampleRate) * freq)
      buf = AudioFrequency(freq, self.noteData.noteNames[pitchIndex], binIx)
      self.audioBufs.append(buf)
  def length(self):
    return len(self.lbuffer)
  def jsonString(self):
    rv = { 'buffers': [] }
    for buf in self.audioBufs:
      rv['buffers'].append(buf.jsonString())
    rv['impulse'] = self.impulse.jsonString()
    return rv
  def getFftAbs(self, buffer, sample):    
    fbuf = fft(buffer)
    realsize = np.int64(self.fftWindow / 2)
    realbuf = np.abs(fbuf)[:realsize]
    return realbuf/self.fftWindow
  def getFrequencies(self):
    sample = self.firstnz
    printCount = 0
    while sample + self.fftWindow < self.lbuffer.shape[0]:
      printCount += 1
      if printCount % self.sampleRate == 0:
        print(f'frequency analysis sample {sample}')
      arl = self.lbuffer[sample:sample + self.fftWindow]
      arr = self.rbuffer[sample:sample + self.fftWindow]
      far = (self.getFftAbs(arl, sample) +  self.getFftAbs(arr, sample)) / 2
      comps = [far[buf.binIx] for buf in self.audioBufs]
      csort = np.argsort(np.array(comps))
      top10 = csort[-10:]
      [self.audioBufs[ix].record(sample, comps[ix]) for ix in top10]
      sample += np.int64(self.fftWindow / 2)
  def getBeats(self):
    sample = self.firstnz
    beatWindow = np.zeros(self.beatWindowSize)
    runningTempo = 0
    beatIndex = 0
    printCount = 0
    while sample + self.impulseWindow < self.lbuffer.shape[0]:
      printCount += 1
      if printCount % self.sampleRate == 0:
        print(f'impulse analysis sample {sample}')
      lbuf = self.lbuffer[sample: sample + self.impulseWindow]
      rbuf = self.rbuffer[sample: sample + self.impulseWindow]
      beatWindow[beatIndex] = (np.std(lbuf) + np.std(rbuf))/2
      beatIndex += 1
      if beatIndex == self.beatWindowSize:
        prominence = np.max(beatWindow)/2
        beatBins = find_peaks(beatWindow, prominence=prominence, distance=400)[0]
        beatValues = [beatWindow[x] for x in beatBins]
        if beatBins.shape[0] > 1:
          beatDistance = np.average(np.diff(beatBins))
          std = np.std(np.diff(beatBins))
          tempo = 60/(beatDistance*(self.impulseWindow/self.sampleRate))
          if runningTempo == 0 or std < 1:
            runningTempo = tempo
          elif std < beatDistance:
            ratio = std/beatDistance
            runningTempo = tempo * (1 - ratio) + runningTempo * ratio
          self.impulse.storeBeat(sample, tempo, beatBins, beatValues)
          # print(f'beats width sample {sample} tempo {tempo} running {runningTempo} error {std} distance {beatDistance}')

        beatIndex = 0
        # sample -= np.int64(self.beatWindowSize / 2)
      sample += self.impulseWindow







