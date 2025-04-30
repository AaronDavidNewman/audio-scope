import numpy as np

class AudioFrequency:
  """Measure a specific frequency of an audio file"""

  slidingWindow = 10

  def __init__(self, frequency, noteName, binIx, jsonObj = None):
    self.frequency = frequency
    self.noteName = noteName
    self.binIx = binIx
    self.samples = dict()
    self.indices = []
  
  def jsonString(self):
    rv = { 'frequency': self.frequency.item(), 
         'binIx': self.binIx.item(), 'noteName': self.noteName,'samples': [] }
    samples = [ {'ix': ss.item(), 'sample': self.samples[ss].item() } for ss in self.indices]
    rv['samples'] = samples    
    return rv

  def record(self, ix, value):
    # print(f'Sample {ix} val={value} note = {self.noteName} freq = {self.frequency}')
    self.samples[ix] = value
    self.indices.append(ix)

  def energyAtIndex(self, sample):    
    self.npindices = np.array(self.indices)
    ix = np.argmin(np.abs(np.array(self.indices) - np.int64(sample)))
    return self.samples[self.indices[ix]]
    

    

