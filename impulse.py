import numpy as np
from enum import Enum

class Impulse:
  
  def __init__(self, impulseWindow):
    self.indices = []
    self.valuesAt = dict()
    self.temposAt = dict()
    self.impulseWindow = impulseWindow
  def jsonString(self):
    rv = { 'valuesAt': [] }
    values = [ {'ix': ss.item(), 'values': self.valuesAt[ss].item() } for ss in self.indices]
    tempos = [ {'ix': ss.item(), 'values': self.temposAt[ss].item() } for ss in self.indices]
    rv['valuesAt'] = values
    rv['temposAt'] = tempos
    return rv
  def storeBeat(self, ix, tempo, bins, values):
    binSamples = (self.impulseWindow * bins) + ix
    for binIndex in range(len(binSamples)):
      self.valuesAt[binSamples[binIndex]] = values[binIndex]
      self.temposAt[binSamples[binIndex]] = tempo
      self.indices.append(binSamples[binIndex])
      # print(f'beats at {ix}:sample = {binSamples[binIndex]} value {self.valuesAt[binSamples[binIndex]]}')
    
  def binValueAtIndex(self, sample):    
    self.npindices = np.array(self.indices)
    ix = np.argmin(np.abs(np.array(self.indices) - np.int64(sample)))
    return self.valuesAt[self.indices[ix]]
  






