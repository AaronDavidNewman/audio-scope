import numpy as np
from enum import Enum

class Impulse:
  """Record tempo and max energy in an audio file"""
  def __init__(self, impulseWindow, jsonObj = None):
    self.indices = []
    self.max = 0
    self.valuesAt = dict()
    self.temposAt = dict()
    self.impulseWindow = impulseWindow
    if jsonObj != None:
      self.max = np.float64(jsonObj['max'])
      for jval in jsonObj['values']:
        ix = np.int64(jval['ix'])
        self.indices.append(ix)
        self.valuesAt[ix] = np.float64(jval['value'])
        self.temposAt[ix] = np.float64(jval['tempo'])

  def jsonString(self):
    rv = { 'max': self.max.item(), 'values': [] }
    values = [ {'ix': ss.item(), 'value': self.valuesAt[ss].item(), 'tempo': self.temposAt[ss].item() } for ss in self.indices]
    rv['values'] = values
    return rv
  def storeBeat(self, ix, tempo, bins, values):
    binSamples = (self.impulseWindow * bins) + ix
    for binIndex in range(len(binSamples)):
      self.valuesAt[binSamples[binIndex]] = values[binIndex]
      self.temposAt[binSamples[binIndex]] = tempo
      self.indices.append(binSamples[binIndex])
      self.max = np.max(np.array([values[binIndex], self.max]))
      # print(f'beats at {ix}:sample = {binSamples[binIndex]} value {self.valuesAt[binSamples[binIndex]]}')
    
  def binValueAtIndex(self, sample):
    ix = np.int64(np.max(np.array([np.argmin(np.abs(np.array(self.indices) - np.int64(sample))) - 1, 0])))
    value = self.valuesAt[self.indices[ix]]
    if ix + 1 < len(self.indices):
      p = (sample - self.indices[ix])/(self.indices[ix + 1] - self.indices[ix])      
      value = (value * p) + (1 - p) * self.valuesAt[self.indices[ix + 1]]
    return ({'value': value / self.max, 'tempo': self.temposAt[self.indices[ix]] })
  






