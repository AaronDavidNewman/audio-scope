import numpy as np
from audioSpectrum import AudioSpectrum
from frameObject import ContourFrame, PolyFrame, spirographFrames
from scipy.io import wavfile
import json
import os
from pathlib import Path
import time

def main(filename):
  jsonFolder = './json/'
  audioFolder = './audio/'
  imageFolder = './frames/'
  FRAME_RATE = 12
  SAMPLES_PER_WINDOW = 4
  if not os.path.exists(jsonFolder):
    os.makedirs(jsonFolder)
  if not os.path.exists(imageFolder):
    os.makedirs(imageFolder)
  for fn in os.listdir(imageFolder):
    filepath = os.path.join(imageFolder, fn)
    if os.path.isfile(filepath):
        os.unlink(filepath)

  jsonFilename = filename.removesuffix('.wav')
  jsonFilename = f'{jsonFolder}{jsonFilename}.json'
  audioFilename = f'{audioFolder}{filename}'
  loadedFromJson = False
  print(f'{jsonFilename}')
  jsonFile = None
  if Path(jsonFilename).is_file():
    with open(jsonFilename) as f:
      jsonContents = f.read()
      jsonFile = json.loads(jsonContents)
      loadedFromJson = True

  sampleRate, audio = wavfile.read(audioFilename)
  lchannel, rchannel = audio.T
  lchannel = np.sqrt(np.int64(lchannel)** 2)
  rchannel = np.sqrt(np.int64(rchannel)** 2)
  max = np.max([np.max(lchannel), np.max(rchannel)])
  lchannel /= max
  rchannel /= max
  start = time.time()
  spectrum = AudioSpectrum(lchannel, rchannel, sampleRate, samplesPerWindow = SAMPLES_PER_WINDOW, \
                          jsonObj = jsonFile)
  loadedFromJson = spectrum.loaded
  spectrum.computeFrequencies()
  spectrum.computeBeats()
  end = time.time()
  imageIx = np.int64((spectrum.firstnz / sampleRate) * FRAME_RATE)
  sample = spectrum.firstnz
  inc = np.int64(sampleRate / FRAME_RATE)
  print(f'firstSample: {spectrum.firstnz} elapsed compute time {end-start}')
  jsonBuf = spectrum.jsonString()
  if not loadedFromJson:
    with open(jsonFilename, 'w') as jf:
      json.dump(jsonBuf, jf)  
  frame = spirographFrames(sample, spectrum, bufSize=12, frameRate = FRAME_RATE)
  while sample < lchannel.shape[0]:
    fn = f'{imageFolder}image{imageIx:04d}.png'
    print(f'image {fn} sample {sample}')
    frame.display(fn)
    sample += inc
    imageIx += 1
    frame.add(sample)
  # print(f'{jsonBuf}')


# main("horn-e4.wav")
main("cicada-soul3.wav")
