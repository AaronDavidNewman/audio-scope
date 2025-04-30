import numpy as np
from audioSpectrum import AudioSpectrum
from scipy.io import wavfile
import json
import os
import time

def main(filename):
  jsonFilename = filename.removesuffix('.wav')
  jsonFilename += '.json'
  print(f'{jsonFilename}')
  sampleRate, audio = wavfile.read(filename)
  lchannel, rchannel = audio.T
  lchannel = np.sqrt(np.int64(lchannel)** 2)
  rchannel = np.sqrt(np.int64(rchannel)** 2)
  max = np.max([np.max(lchannel), np.max(rchannel)])
  lchannel /= max
  rchannel /= max
  start = time.time()
  spectrum = AudioSpectrum(lchannel, rchannel, sampleRate)  
  spectrum.getFrequencies()
  spectrum.getBeats()
  end = time.time()
  print(f'elapsed time {end-start}')
  jsonBuf = spectrum.jsonString()
  # print(f'{jsonBuf}')
  jsonFile = os.path.splitext(filename)[0]+'.json'
  with open(jsonFile, 'w') as jf:
    json.dump(jsonBuf, jf)

  bbb=spectrum.impulse.binValueAtIndex(500000)
  print(f'{bbb}')
  

        # if eg[0] > 0:
        #  print(f'energy sample {spectrum.index} for {spectrum.audioBufs[bufix].frequency} is {eg}')

# main("horn-e4.wav")
main("whomp-time-gated.wav")
