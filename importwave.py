import sys
import numpy as np
import pyaudio
import matplotlib.pyplot as plt
from scipy.fft import rfft, fft
from scipy.io import wavfile
from tkinter import TclError

if len(sys.argv) < 2:
    print(f'Plays a wave file. Usage: {sys.argv[0]} filename.wav')
    filename="air-breath2.wav"
    # filename="poly1.wav"
    # filename="noogarpy.wav"
    # filename="ween24-2.wav"
else:
   filename = sys.argv[1]

FORMAT = pyaudio.paInt16     # audio format (bytes per sample?)
CHANNELS = 1                 # single channel for microphone
RATE = 44100                 # samples per second
dotsize = 128
imagesize = 512
frameskip = 2
bunch = 15

sample_rate, audio_time_series = wavfile.read(filename)
single_sample_data = audio_time_series[:sample_rate]
print(f"shape data is {audio_time_series.shape}")
print(f"shape samples is {single_sample_data.shape}")
print(f"filename is {filename}")
xar = np.linspace(0, 4*np.pi, 512)
def truncateArrayForWindows(ndarray, windows):
   windowSize = np.floor(len(ndarray) // windows)
   windowSlice = ndarray[0: windowSize * windows]
   rv = np.asarray(np.split(windowSlice.flatten(), windows))
   return rv

def fft_plot(audio, sample_rate):
  xenergy = []
  maxEnergy = 0
  N = len(audio)    # Number of samples
  # we could create 2 FFT window for each second of samples, each window
  # is 1/2 the sample rate
  windows = N // (sample_rate // 2)
  iii = type(audio)
  print(f"type of audio is {iii}")
  # average the stereo values and truncate the array so each window is the same size
  stereo_avg = np.mean(audio, axis=1)
  audio_windows = truncateArrayForWindows(stereo_avg, windows)

  #calculate the max amplitude in each window and overall, to normalize
  for x in range(0, len(audio_windows)):
     xe = np.abs(np.mean(audio_windows[x]))
     if xe > maxEnergy:
        maxEnergy = xe
     xenergy.append(xe)
  
  xenergy = xenergy/maxEnergy

  # added this loop, removed original code
  for x in range(0, len(audio_windows)):
    # don't make an image for every sampling window, but we combine
    # windows
    if x % frameskip != 0:
       continue
    # max freq
    # maxy = np.argmax(y_window)
    print(f"img {x}")
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={'projection': 'polar'})
    fig.set_frameon(False)
    dpi = fig.figure.get_dpi()
    fig.set_size_inches(512/dpi, 512/dpi)
    ax.set_aspect('auto')
    ax.set_axis_off()
    
    counts = dotsize * np.zeros(imagesize)
    # combine 'bunch' windows into a single image
    for subplt in range(0, bunch):
      yar=[]
      colors=[]
      # make dot sizes bigger when there are multiples of same frequency
      # ix = x - (frameskip // bunch) * (bunch - subplt)
      ix = np.max([x - (bunch - subplt), 0])
      y_window = np.abs(rfft(audio_windows[ix]))
      y_max = np.max([np.max(y_window), 1])
      y_window = (512.0 / y_max) * y_window
      # limit ourselves to the interesting human hearing frequencies
      # y_window = y_window[64:1024]
      y_freq = truncateArrayForWindows(y_window, imagesize)
      y_freq = np.mean(y_freq, axis=1)
      max_freq = np.max([np.max(y_freq), 1])
      # normalize the frequency so it falls between 0 and image size, since we have a limited image size
      # and also so that the average volume of this sample 
      # y_freq = np.subtract(subtr, np.around(y_freq * xenergy[x] * (imagesize/(sample_rate / 2))))
      #y_max = np.max(y_freq)
      for x_freq in range(0,imagesize):
        yar.append(y_freq[x_freq] * (imagesize / max_freq))
        counts[int(y_freq[x_freq]) - 1] += int(y_freq[x_freq]/max_freq)*dotsize + dotsize * (1 - xenergy[x]) / bunch
        color = 1 - xenergy[x]
        colors.append([color, color, color])
        

      # ax.plot(xar, yar, label='', linestyle=(0,(1,10)))
      ax.scatter(xar, yar, c=colors, s=counts)
      # ax.bar(xar, yar, width=counts, color=colors)
    filename = './output/figure_' + str(x)
    fig.savefig(filename)
    plt.close()
    

  # x_freq = np.linspace(0, sample_rate//2, N//2)
  # Changed from abs(y_freq[:domain]) -> y_freq[:domain]

lchannel, rchannel = audio_time_series.T
avgamp=(lchannel + rchannel)/2
length = audio_time_series.shape[0] / sample_rate
time = np.linspace(0., length, audio_time_series.shape[0])
# plt.show()
# Changed from single_sample_data -> audio_time_series
fft_plot(audio_time_series, sample_rate)