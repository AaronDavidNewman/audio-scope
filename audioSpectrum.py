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
  """Music and audio analysis routines"""
  def __init__(self, lbuffer, rbuffer, sampleRate, jsonObj=None):
    self.lbuffer  = lbuffer
    self.rbuffer = rbuffer
    self.sampleRate = sampleRate
    self.noteData = NoteFrequency()
    self.currentSample = 0
    self.maxFft = 0
    self.freqCount = len(self.noteData.frequencies)
    # Fudge factor since higher frequencies are less accurate
    self.freqAdj = np.linspace(0, np.sqrt(2), self.freqCount)

    self.bufToBinIndexMap = dict()
    self.freqCount = len(self.noteData.frequencies)
    self.audioBufs = []
    self.loaded = False
    impulseJson = None
    if jsonObj != None and len(jsonObj['buffers'])>= len(self.noteData.frequencies):
      self.loaded = True
      self.maxFft = np.float64(jsonObj['maxFft'])
      impulseJson = jsonObj['impulse']

    self.impulse = Impulse(self.impulseWindow, jsonObj = impulseJson)
    self.firstnz = np.min([np.nonzero(self.lbuffer)[0][0],np.nonzero(self.rbuffer)[0][0]])
    for pitchIndex in range(self.freqCount):
      freq = self.noteData.frequencies[pitchIndex]
      # windowIndex*(sampleRate/windowSize) = frequency, so 
      # windowIndex = (windowSize/sampleRate)*frequency
      jsonFreq = None
      if jsonObj != None and len(jsonObj['buffers'])>pitchIndex:
        jsonFreq = jsonObj['buffers'][pitchIndex]
      binIx = np.int64((self.fftWindow / self.sampleRate) * freq)
      self.bufToBinIndexMap[binIx] = pitchIndex
      buf = AudioFrequency(freq, self.noteData.noteNames[pitchIndex], binIx, jsonFreq)
      self.audioBufs.append(buf)
  def length(self):
    return len(self.lbuffer)
  def frequencyBins(self):
    return len(self.audioBufs)
  # Find the dominant frequencies at a specific sample
  def getFrequencyEnergyAtSample(self, sample):
    energies = []
    for buf in self.audioBufs:
      if len(buf.indices) > 0:
        lte = np.argmin(np.abs(np.array(buf.indices) - sample)) - 1
        lte = 0 if lte < 0 else lte
        if np.abs(buf.indices[lte] - sample) < self.fftWindow:
          e = buf.samples[buf.indices[lte]]
          
          e += e*self.freqAdj[self.bufToBinIndexMap[buf.binIx]]
          energies.append(
            {'bin': buf.binIx, 'energy': e / self.maxFft })
    return energies
  def impulseEnergyAtSample(self, sample):
    return self.impulse.binValueAtIndex(sample)
  def jsonString(self):
    rv = { 'sampleRate': self.sampleRate, 'firstnz': self.firstnz.item(), 
          'maxFft': self.maxFft.item(),
          'buffers': [] }
    for buf in self.audioBufs:
      rv['buffers'].append(buf.jsonString())
    rv['impulse'] = self.impulse.jsonString()
    return rv
  def getFftAbs(self, buffer, sample):    
    fbuf = fft(buffer)
    realsize = np.int64(self.fftWindow / 2)
    realbuf = np.abs(fbuf)[:realsize]
    return realbuf/self.fftWindow
  # compute the dominant frequencies of the audio via fft
  def computeFrequencies(self):
    if self.loaded:
      return
    sample = self.firstnz
    printCount = 0
    printMod = np.int64(self.sampleRate / self.fftWindow) * 16 + 1
    while sample + self.fftWindow < self.lbuffer.shape[0]:
      printCount += 1
      if printCount % printMod == 0:
        print(f'frequency analysis sample {sample}')
      # do fft on the fft window, averaging left and right
      arl = self.lbuffer[sample:sample + self.fftWindow]
      arr = self.rbuffer[sample:sample + self.fftWindow]
      far = (self.getFftAbs(arl, sample) +  self.getFftAbs(arr, sample)) / 2
      # far contains values for each FFT frequency (bin).  Put them into 
      # an array for each note (audioBufs), where 0 contains energy for A0, etc
      comps = [far[buf.binIx] for buf in self.audioBufs]
      # sort these by value
      csort = np.argsort(np.array(comps))
      top10 = csort[-10:]
      max = comps[top10[9]]
      # for ixAdj in range(top10.shape[0]):
      # update the max to normalize later
      self.maxFft = max if max > self.maxFft else self.maxFft
      # Each audioBuf keeps track of its own samples
      [self.audioBufs[ix].record(sample, comps[ix]) for ix in top10]
      sample += np.int64(self.fftWindow / 2)
  def computeBeats(self):
    if self.loaded:
      return
    sample = self.firstnz
    beatWindow = np.zeros(self.beatWindowSize)
    runningTempo = 0
    beatIndex = 0
    printCount = 0
    while sample + self.impulseWindow < self.lbuffer.shape[0]:
      printCount += 1
      # we find chunks of energy in samples of size self.impulseWindow
      if printCount % self.sampleRate == 0:
        print(f'impulse analysis sample {sample}')
      lbuf = self.lbuffer[sample: sample + self.impulseWindow]
      rbuf = self.rbuffer[sample: sample + self.impulseWindow]
      beatWindow[beatIndex] = (np.std(lbuf) + np.std(rbuf))/2
      beatIndex += 1
      # after beatWindowSize chunks, we analyze to find the beat
      if beatIndex == self.beatWindowSize:
        prominence = np.max(beatWindow)/2
        # find peaks > 1/2 of max,
        # and further than  1/10 of a second apart, that's about 400 per minutes
        beatBins = find_peaks(beatWindow, prominence=prominence, distance=400)[0]
        beatValues = [beatWindow[x] for x in beatBins]
        if beatBins.shape[0] > 1:
          # calculate the tempo from running average of distance between peaks
          beatDistance = np.average(np.diff(beatBins))
          std = np.std(np.diff(beatBins))
          tempo = 60/(beatDistance*(self.impulseWindow/self.sampleRate))
          if runningTempo == 0 or std < 1:
            runningTempo = tempo
          elif std < beatDistance:
            ratio = std/beatDistance
            runningTempo = tempo * (1 - ratio) + runningTempo * ratio
          # we store the data starting when we started sampling, on window size ago
          self.impulse.storeBeat(sample - (self.beatWindowSize * self.impulseWindow), tempo, beatBins, beatValues)
          # print(f'beats width sample {sample} tempo {tempo} running {runningTempo} error {std} distance {beatDistance}')

        beatIndex = 0
        # sample -= np.int64(self.beatWindowSize / 2)
      sample += self.impulseWindow







