import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from io import BytesIO
from PIL import Image, ImageFilter, ImageEnhance, ImageDraw, ImageChops

class ContourFrame:
  def __init__(self, sample, spectrum, bufSize = 5, frameRate=4):
    self.spectrum = spectrum
    # the accumulation buffer that smooths the samples
    self.bufSize = bufSize
    self.frameRate = frameRate
    # how far from the beat (peak) the last sample was
    self.distanceFromBeat = 0
    self.tempo = np.float64(120)
    self.bufIx = 0
    numFreqs = spectrum.frequencyBins()
    self.risingSize = np.int64(numFreqs / 2)
    self.fallingSize = numFreqs - self.risingSize
    frequencyBins = np.zeros(numFreqs)
    freqs = spectrum.getFrequencyEnergyAtSample(sample)
    maxFft = self.spectrum.maxFft
    self.bufs = []
    self.tempoSample = sample
    self.angles = []
    self.shape = np.concatenate((np.linspace(0, maxFft, self.risingSize), 
          np.linspace(maxFft, 0, self.fallingSize))).reshape([-1, 1])
    for ix in range(self.bufSize):
      self.bufs.append(frequencyBins * self.shape)

    for freq in freqs:
      frequencyBins[spectrum.bufToBinIndexMap[freq['bin']]] = freq['energy']
    self.anglesIx = np.argsort(frequencyBins)
    self.bufs[self.bufIx] = frequencyBins * self.shape
    self.bufIx += 1
  
  def add(self, sample):
    energy = self.spectrum.impulseEnergyAtSample(sample)
    background = energy['value']
    # instantaneous distance from peaks, and running tempo calculation
    self.distanceFromBeat = energy['between']
    self.tempo = energy['tempo'] # not directly used
    # smear the current contour by moving the previous samples up/down and 
    # attenuating them
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
    # create the frequency matrix for this sample
    frequencyBins = np.zeros(numFreqs)
    self.frequencyBins = np.zeros(numFreqs)
    freqs = spectrum.getFrequencyEnergyAtSample(sample)
    for freq in freqs:
      frequencyBins[spectrum.bufToBinIndexMap[freq['bin']]] = freq['energy']
    # scale the frequency by the overall volume
    if sample > self.spectrum.impulseWindow * self.spectrum.beatWindowSize:
      self.bufs[self.bufIx] = frequencyBins * self.shape
    else:
      self.bufs[self.bufIx] = frequencyBins * self.shape
    self.bufIx = (self.bufIx + 1) % self.bufSize

  def display(self, filename):
    # sum the accumlations matrices into a copy matrix
    copy = np.matrix.copy(self.bufs[0])
    for ix in range(1, self.bufSize):
      copy += self.bufs[ix]
    # create contour close to 512 square.  Might be better to just scale it.
    fig = plt.figure(frameon=False, figsize=(5.3333,5.3333),dpi=96)
    ax = fig.add_axes([0,0,1,1])
    ax.set_axis_off()
    ax.contourf(copy,  cmap='coolwarm')
    # ax.matshow(copy)
    #  fig.savefig(filename)
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    with Image.open(buffer) as img:
      # also sharpen the image as we get closer to the beat
      enhancer = ImageEnhance.Sharpness(img)
      sharpened = enhancer.enhance((self.distanceFromBeat * 2) + 0.5)
      # print(f'sample distance={self.distanceFromBeat}')
      # sharpened = img.filter(ImageFilter.SHARPEN)
      sharpened.save(filename, format='png')

    plt.close(fig)
    # plt.show()


class PolyFrame:
  def __init__(self, sample, spectrum, bufSize = 5, frameRate=4):
    self.spectrum = spectrum
    self.canvasSize = 512
    self.bufSize = bufSize
    self.frameRate = frameRate
    self.colormap =  [(ar[0], ar[1], ar[2]) for ar in plt.colormaps['hot'](range(0,256))]
    self.bgcolormap = [(ar[0], ar[1], ar[2]) for ar in plt.colormaps['YlOrRd'](range(0,256))]
    self.distanceFromBeat = 0
    self.previousColor = 128
    self.tempo = np.float64(120)
    self.bufIx = 0
    numFreqs = spectrum.frequencyBins()
    self.risingSize = np.int64(numFreqs / 2)
    self.fallingSize = numFreqs - self.risingSize
    frequencyBins = np.zeros(numFreqs)
    freqs = spectrum.getFrequencyEnergyAtSample(sample)
    for freq in freqs:
      frequencyBins[spectrum.bufToBinIndexMap[freq['bin']]] = freq['energy']
    freqIx = np.flip(np.argsort(frequencyBins))
    self.getEnergy(sample)
    self.image = self.createBinImage(freqIx, sample)
    
  def getEnergy(self, sample):
    energy = self.spectrum.impulseEnergyAtSample(sample)
    self.background = energy['value']
    # instantaneous distance from peaks, and running tempo calculation
    self.distanceFromBeat = energy['between']
    self.tempo = energy['tempo'] # not directly used

  def tupletfyRect(self,rect):
    nrect = []
    for pts in rect:
      nrect.append((pts[0], pts[1]))
    return nrect

  def createBinImage(self, freqIx, sample):
    size = self.canvasSize
    binCount = self.spectrum.frequencyBins()
    # create the polygon color as in index into the colormap, based on 
    # where the beat is.
    hue = min(255,256 - np.int32(256 * self.distanceFromBeat))
    hue = np.int32((self.previousColor + hue)/2)
    hue = min(255,(max(hue, 0)))

    # create the background color, also based on volume
    bgcolor = np.int32((128 + 128*self.background)/2)
    self.previousColor = hue
    # rotate the figure slowly, so the legs aren't always in the same place if 
    # there isn't a lot of variance in the spectrum
    # create the target background, expand the center based on the tempo
    samplesPerTempo = (self.spectrum.sampleRate / (self.tempo / 60)) * 16
    position = sample % samplesPerTempo
    tempoAngle = (2 * np.pi * position)/samplesPerTempo
    tempoRadius = (1 + np.cos(tempoAngle) / 2)
    radius = (size/2) * tempoRadius


    # Make many legs of the spider, based on how loud it is here
    anglecount = np.int32(3 + (12*(self.background) - 1))
    angles = []
    startangle = 2 * np.pi * (sample/self.spectrum.numSamples)
    for angle in range(anglecount):
      # angles.append(startangle + angle * (np.pi/2 * angle + (np.pi/2)*self.background))
      angles.append(startangle + angle*(tempoAngle/anglecount))
    # create a blank image with the background color
    bg = np.int32(np.array(self.bgcolormap[bgcolor]) * 256)
    img = Image.new(mode='RGB', size=(size, size), color=tuple(bg))
    segcount=100
    dr = ImageDraw.Draw(img)
    fig = plt.figure(frameon=False, figsize=(5.3333,5.3333),dpi=96)
    a = fig.add_axes([0,0,1,1])    
    a.set_axis_off()
    a.set_xlim(0,512)
    a.set_ylim(0,512)

    for circle in range(5):
      bgcolor = np.int32(bgcolor * 0.95)
      radius = np.int32(radius * 0.8)
      fill = np.int32(np.array(self.bgcolormap[bgcolor])*256)
      dr.circle((size/2, size/2), radius, fill=tuple(fill))
    segs = np.array(np.empty)
    widths = np.array(np.empty)
    for angle in angles:
      lastX = np.float64(size/2)
      lastY = np.float64(size/2)
      w = size/6
      h = np.int32(8 + 8*(binCount - freqIx[0])/binCount)
      # starta = freqIx[0] / binCount
      for bin in freqIx[:5]:
        div = (bin + 1)/(binCount + 1)
        x1 = np.cos(angle + (2*np.pi*div))*w+lastX
        y1 = np.sin(angle+(2*np.pi*div))*w+lastY
        # x2 = np.cos(angle*(div) - np.pi)*h+lastX
        # y2 = np.cos(angle*(div) - np.pi)*h+lastY
        # x3 = np.cos(angle*(div) - np.pi)*h + x1
        # y3 = np.cos(angle*(div)- np.pi)*h + y1
        xxs = np.linspace(lastX, x1, segcount)
        yys = np.linspace(lastY, y1, segcount)
        pts1 = np.array((xxs, yys)).T.reshape(-1, 1, 2)
        lasth = h
        lastw = w
        h = 2*h/3
        w = w/2
        if len(segs.shape) == 0:
          segs = np.concatenate([pts1[:-1], pts1[1:]], axis=1)
          widths = 1+np.linspace(lasth, h, segcount)
        else:
          newsegs = np.concatenate([pts1[:-1], pts1[1:]], axis=1) 
          widths =  np.concatenate((widths, 1+np.linspace(h, h-2, segcount)))
          segs = np.concatenate((segs, newsegs))        
        lastX = x1
        lastY = y1
        color = np.array(self.colormap[hue])
        lc = LineCollection(segs, linewidths=widths,color=color)
        lc.set_capstyle('round')
    a.add_collection(lc)
    self.tempoSample = sample
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    with Image.open(buffer) as img1:
      rgb = img1.convert('RGB')
      img = ImageChops.darker(img, rgb)
    plt.close(fig)

    return img
  
  # Get the FFT frequencies for this sample
  def getFrequencyBins(self, sample):
    spectrum = self.spectrum
    numFreqs = spectrum.frequencyBins()
    self.frequencyBins = np.zeros(numFreqs)
    freqs = spectrum.getFrequencyEnergyAtSample(sample)
    for freq in freqs:
      self.frequencyBins[spectrum.bufToBinIndexMap[freq['bin']]] = freq['energy']
  
  def add(self, sample):
    self.getFrequencyBins(sample)
    self.getEnergy(sample)
    freqIx = np.flip(np.argsort(self.frequencyBins))
    self.image = self.createBinImage(freqIx, sample)

  def display(self, filename):
    if filename=='./frames/image0634.png':
      print('ouch')
    self.image.save(filename, format='png')
    # plt.show()