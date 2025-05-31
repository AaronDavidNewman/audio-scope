import numpy as np

class AudioFrequency:
  """Record a specific frequency of an audio file"""
  slidingWindow = 10

  def __init__(self, frequency, noteName, binIx, jsonObj = None):
    self.frequency = frequency
    self.noteName = noteName
    self.binMax = np.float32(0)
    self.binIx = binIx
    self.samples = dict()
    self.indices = []
    if jsonObj != None:
      self.binMax = np.float32(jsonObj['binMax'])
      for sample in jsonObj['samples']:
        ix = np.int64(sample['ix'])
        self.samples[ix] = np.float64(sample['sample'])
        self.indices.append(ix)
  
  def jsonString(self):
    rv = { 'frequency': self.frequency.item(), 
         'binMax': self.binMax.item(),
         'binIx': self.binIx.item(), 'noteName': self.noteName,'samples': [] }
    samples = [ {'ix': ss.item(), 'sample': self.samples[ss].item() } for ss in self.indices]
    rv['samples'] = samples    
    return rv

  def record(self, ix, value):
    # print(f'Sample {ix} val={value} note = {self.noteName} freq = {self.frequency}')
    self.samples[ix] = value
    self.binMax = value if value > self.binMax else self.binMax
    self.indices.append(ix)

  def energyAtIndex(self, sample):    
    self.npindices = np.array(self.indices)
    ix = np.argmin(np.abs(np.array(self.indices) - np.int64(sample)))
    return self.samples[self.indices[ix]]
    

    

