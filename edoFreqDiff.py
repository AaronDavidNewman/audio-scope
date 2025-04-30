import numpy as np

# Calculate the frequency of each note, in HZ, through the range of octaves
def calculateNoteFrequencies(ova, steps, letter_names, ar, namear):
    halfStepMantissa = np.pow(2, 1./steps)
    cindex = letter_names.index('C')
    for oo in range(ova):
      base = START_FREQ * np.pow(2, oo)
      for step in range(steps):
        ix = steps * oo + step
        oolbl = oo + 1
        # by convention, octaves start at C, ie. A1 is above C1, so compensate here
        if step > cindex:
            oolbl = oolbl + 1
        noteName = f'{letter_names[step]} {oolbl}'
        namear.append(noteName)
        freq = base * np.pow(halfStepMantissa, step)
        ar.append(freq)

START_FREQ = 55  # A0
EQ_STEPS_PER_OCTAVE = 12
EDO_STEPS_PER_OCTAVE = 31
EQ_LETTER_NAMES = ['A', 'Bb', 'B', 'C', 'Db', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab']
EDO_LETTER_NAMES = ['A', 'A+', 'A#', 'Bb', 'B-', 'B', 'B#', 'Cb' ,'C', 'C+', 'C#', 'Db', 
                    'D-', 'D', 'D+', 'D#', 'Eb', 'E-', 
                    'E', 'E#', 'Fb', 'F', 'F+', 'F#', 'Gb', 'G-', 'G', 'G+', 'G#', 'Ab', 'A-']

def closest(steps1, letters1, steps2, letters2):
  npar = np.asarray(steps1)
  for stepIx in range(len(steps2) - 1):
    stepVal = steps2[stepIx]    
    idx = np.abs(npar - stepVal).argmin()
    if idx < len(steps1) - 1:
      dfreq = steps2[stepIx] - steps1[idx]
      dcents = np.round(100*(dfreq / (steps1[idx + 1] - steps1[idx])))
      print(f'ET31 {letters2[stepIx]} ET12 {letters1[idx]} +freq {dfreq} +cents {dcents}')

et12steps = []
et12freqar= []
et12names=  []
et31steps=  []
et31names= []
calculatedEq= False

calculateNoteFrequencies(7, 12, EQ_LETTER_NAMES, et12steps, et12names)
calculateNoteFrequencies(7, 31, EDO_LETTER_NAMES, et31steps, et31names)

def forceCalculateEq():
  global calculatedEq
  global et12freqar
  if not calculatedEq:
     calculateNoteFrequencies(7, 12, EQ_LETTER_NAMES, et12steps, et12names)
     et12freqar = np.asarray(et12steps)
     calculatedEq = True
   
def closestNoteIndex(frequency):
  global et12freqar
  forceCalculateEq()
  yy = np.abs(et12freqar - frequency)
  am = np.argmin(yy)
  return am


xx = closestNoteIndex(440)
print(f'{xx}')
    
       
       
      
      

