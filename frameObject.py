import numpy as np
import matplotlib.pyplot as plt
class Frame:
  def __init__(self, sample, spectrum, bufSize = 5):
    self.spectrum = spectrum
    self.bufSize = bufSize
    self.bufIx = 0
    numFreqs = spectrum.frequencyBins()
    self.risingSize = np.int64(numFreqs / 2)
    self.fallingSize = numFreqs - self.risingSize
    frequencyBins = np.zeros(numFreqs)
    freqs = spectrum.getFrequencyEnergyAtSample(sample)
    maxFft = self.spectrum.maxFft
    self.bufs = []
    self.shape = np.concatenate((np.linspace(0, maxFft, self.risingSize), 
          np.linspace(maxFft, 0, self.fallingSize))).reshape([-1, 1])
    for ix in range(self.bufSize):
      self.bufs.append(frequencyBins * self.shape)

    for freq in freqs:
      frequencyBins[spectrum.bufToBinIndexMap[freq['bin']]] = freq['energy']

    self.bufs[self.bufIx] = frequencyBins * self.shape
    self.bufIx += 1
  
  def add(self, sample):
    background = self.spectrum.impulseEnergyAtSample(sample)['value']
    rnd = np.random.rand(self.bufs[0].shape[0], self.bufs[0].shape[1])*0.95
    for bufIx in range(self.bufSize):
      buf = self.bufs[bufIx]
      buf1 = np.roll(buf, 1, 1)
      buf2 = np.roll(buf, -1, 1)
      buf1 = buf1 + np.roll(buf1.T, 1, 0)
      buf2 = buf2 + np.roll(buf2.T, -1, 0)
      buf = (buf1 + buf2) * background
      self.bufs[bufIx] = buf
    spectrum = self.spectrum
    numFreqs = spectrum.frequencyBins()
    frequencyBins = np.zeros(numFreqs)
    self.frequencyBins = np.zeros(numFreqs)
    freqs = spectrum.getFrequencyEnergyAtSample(sample)
    for freq in freqs:
      frequencyBins[spectrum.bufToBinIndexMap[freq['bin']]] = freq['energy']
    if sample > self.spectrum.impulseWindow * self.spectrum.beatWindowSize:
      self.bufs[self.bufIx] = frequencyBins * self.shape
    else:
      self.bufs[self.bufIx] = frequencyBins * self.shape
    self.bufIx = (self.bufIx + 1) % self.bufSize

  def display(self, filename):    
    copy = np.matrix.copy(self.bufs[0])
    for ix in range(1, self.bufSize):
      copy += self.bufs[ix]
    fig = plt.figure(frameon=False, figsize=(5.3333,5.3333),dpi=96)
    ax = fig.add_axes([0,0,1,1])
    ax.set_axis_off()
    ax.imshow(copy,  interpolation='bilinear', aspect='equal')
    # ax.matshow(copy)
    fig.savefig(filename)
    plt.close(fig)
    # plt.show()
    print(f'saved {filename}')

