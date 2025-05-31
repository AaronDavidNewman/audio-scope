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
    self.tempo = energy['tempo']

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
    # create concentric circles
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
      # connect a line from the center in the direction of the pitch
      # on the circle of 5ths.  Then connect that line to another 
      # line for the next-loudest overtone
      for bin in freqIx[:5]:
        div = (bin + 1)/(binCount + 1)
        x1 = np.cos(angle + (2*np.pi*div))*w+lastX
        y1 = np.sin(angle+(2*np.pi*div))*w+lastY
        xxs = np.linspace(lastX, x1, segcount)
        yys = np.linspace(lastY, y1, segcount)
        pts1 = np.array((xxs, yys)).T.reshape(-1, 1, 2)
        lasth = h
        lastw = w
        # make each line smaller, to give the spider leg effect
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
    self.image.save(filename, format='png')
    # plt.show()

class spirographFrames:
  HIST_SIZE = 5 # size of history buffer in frames
  COLOR_BINS = 12
  BEATS_RIPPLE = 16
  def __init__(self, sample, spectrum, bufSize = 5, frameRate=4):
    self.spectrum = spectrum
    # the accumulation buffer that smooths the samples
    self.bufSize = bufSize
    self.frameRate = frameRate
    # how far from the beat (peak) the last sample was
    self.distanceFromBeat = 0
    self.histSize = self.HIST_SIZE
    # binIxHist keeps a list of bins for some number of samples
    # binValHist keeps a list of values for some number of samples
    self.binIxHist = []
    self.binValHist = []
    self.binEnergyHist = []
    self.binIx = 0
    # sqrt to make the colors in the middle more promninent
    colorlin = (np.linspace(0, 1, (self.COLOR_BINS+1))**0.5)*256
    for i in range(self.histSize):
      bins = [0]
      vals = [0]
      self.binIxHist.append(bins)
      self.binValHist.append(vals)
      self.binEnergyHist.append(0)
    self.colormap =  [(ar[0], ar[1], ar[2]) for ar in plt.colormaps['gist_rainbow'](np.int32(colorlin))]
    self.tempo = np.float64(120)  # default starting tempo
    self.add(sample)
    
  def getEnergy(self, sample):
    energy = self.spectrum.impulseEnergyAtSample(sample)
    self.energy = energy['value']
    self.binEnergyHist[self.binIx] = self.energy
    # instantaneous distance from peaks, and running tempo calculation
    self.distanceFromBeat = energy['between']
    self.tempo = energy['tempo'] 

  # Get the FFT frequencies for this sample
  def getFrequencyBins(self, sample):
    spectrum = self.spectrum
    # {'bin': buf.binIx, 'energy': e / self.maxFft }
    freqs = spectrum.getFrequencyEnergyAtSample(sample)
    bins = []
    binvals = []
    for freq in freqs:
      binIx = self.spectrum.bufToBinIndexMap[freq['bin']]
      # binMax = self.spectrum.audioBufs[binIx].binMax
      binvals.append(freq['energy'])
      bins.append(freq['bin'])
    self.binIxHist[self.binIx] = bins
    self.binValHist[self.binIx] = binvals
    self.binIx = (self.binIx + 1) % self.histSize

  def add(self, sample):
    # diminsh past bins in the history by 1/2
    for i in range(self.histSize):
      bins = self.binValHist[i]
      for j in range(len(bins)):
        bins[j] *= 0.5
      self.binEnergyHist[i] *= 0.85

    self.getFrequencyBins(sample)
    self.getEnergy(sample)
    self.image = self.createBinImage(sample)

  def createBinImage(self, sample):
    # change the radius of the outer circle with the tempo
    samplesPerTempo = (self.spectrum.sampleRate / (self.tempo / 60)) * self.BEATS_RIPPLE
    position = (sample % samplesPerTempo)/samplesPerTempo
    X = np.zeros((12 + 1, 12 + 1))
    fig = plt.figure(figsize=(5.334, 5.334), dpi=96)
    a = fig.add_axes([0,0,1,1])
    a.set_xlim(12)
    a.set_ylim(12)
    a.set_axis_off()
    # make blocks based on overall energy average with bin energy
    for i in range(self.histSize):
      bins = self.binIxHist[i]
      binvals = self.binValHist[i]
      for j in range(len(bins)):
        bc = bins[j]
        val = (binvals[j] + self.binEnergyHist[i]) / 2
        yval = np.int32(self.COLOR_BINS * val)
        xpos = yval + np.abs(i - self.binIx)
        if (xpos % 2) == 0:
          xpos = np.int32((self.COLOR_BINS / 2) - (xpos / 2))
        else:
          xpos = np.int32((self.COLOR_BINS / 2) + (xpos / 2))
        xpos = self.COLOR_BINS if xpos > self.COLOR_BINS else xpos
        xpos = 0 if xpos < 0 else xpos
        X[(bc + i) % self.COLOR_BINS, xpos] += yval
    a.imshow(X, cmap='Blues', interpolation='gaussian')
    buffer1 = BytesIO()
    fig.savefig(buffer1, format='png')    
    # a.imshow(X, cmap='Blues')
    c1 = np.linspace(0, np.pi * 2, 1000)
    c2 = np.linspace(0, np.pi *14, 1000)
    r1=5 - position  # slighly change size with tempo
    r2=3
    lwidth = self.histSize
    numFreqs = self.spectrum.frequencyBins()
    for i in range(self.histSize):
      bins = self.binIxHist[i]
      binvals = self.binValHist[i]
      for j in range(len(bins)):
        b = fig.add_axes([0,0,1,1])
        b.set_axis_off()
        bc = np.float32(bins[j])/numFreqs
        val = binvals[j]
        yval = self.COLOR_BINS * val
        XX = (np.array(r1*np.cos(c1) + bc*yval*np.cos(c2)))*val
        YY = (np.array(r1*np.sin(c1) + bc*yval*np.sin(c2)))*val
        b.plot(XX,YY, color=self.colormap[np.min((np.int32(yval), self.COLOR_BINS))], linewidth=lwidth)
      lwidth = lwidth - 1
    buffer2 = BytesIO()
    fig.savefig(buffer2, format='png')
    with Image.open(buffer2) as image2:
      with Image.open(buffer1) as image1:
        img = ImageChops.darker(image2, image1)

    plt.close(fig)
    return img
  def display(self, filename):
    self.image.save(filename, format='png')



    
