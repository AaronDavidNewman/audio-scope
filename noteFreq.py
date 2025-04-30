import numpy as np

class NoteFrequency:
  '''Calculate the frequencies of notes in equal temperment'''
  START_FREQ = 55  # A0
  EQ_STEPS_PER_OCTAVE = 12
  EQ_LETTER_NAMES = ['A', 'Bb', 'B', 'C', 'Db', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab']
  OCTAVES=7
  
  def __init__(self):
    self.frequencies = []  # python array of freq    
    self.noteNames =  []  # array of name/octave
    
    ## these could be passed in or default for additional tuning schemes
    self.startFrequency = self.START_FREQ
    self.stepsPerOctave = self.EQ_STEPS_PER_OCTAVE
    self.letterNames = self.EQ_LETTER_NAMES    
    self.octaves = self.OCTAVES

    halfStepMantissa = np.pow(2, 1./self.stepsPerOctave)
    cindex = self.letterNames.index('C')
    for oo in range(self.octaves):
      base = self.startFrequency * np.pow(2, oo)
      for step in range(self.stepsPerOctave):
        ix = self.stepsPerOctave * oo + step
        oolbl = oo + 1
        # by convention, octaves start at C, ie. A1 is above C1, so compensate here
        if step > cindex:
            oolbl = oolbl + 1
        noteName = f'{self.letterNames[step]} {oolbl}'
        self.noteNames.append(noteName)
        freq = base * np.pow(halfStepMantissa, step)
        self.frequencies.append(freq)
    self.npFrequencies =np.array(self.frequencies)
  def createFftBins(self, sampleRate, windowSize):
    nfix = 0
    fq = 0
    output = []
    fdelta = sampleRate / windowSize  # frequency steps for each fft window, window (n) is fdelta*n hz    
    while nfix < self.frequencies.shape[0]:
      fdiff = np.abs((fq * fdelta) - self.frequencies[nfix])  
      prevdiff = fdiff
      while fq < windowSize // 2 and prevdiff >= fdiff:
        fq = fq + 1
        prevdiff = fdiff
        fdiff = np.abs((fq * fdelta) - self.noteFrequencies[nfix])
      fq = fq - 1
      output.append(fq)
      nfix += 1
    self.fftBins = output
    return output

