The file visualize.py will create interesting visualizations from a .wav file.  It bases the visualizations, loosely, on the frequency content of the audio.

It is not super parameterized, but basically:
1. The last line of code in visualize.py has the path to the wav file.
2. FRAME_RATE is how many images are generated per second.
3. SAMPLES_PER_WINDOW indicates how many times we calculate the frequency information per FFT buffer, higher number is smoother output but takes more calculation.
4. The frame object created on line 59 determines which type of frame image is created.  `PolyFrame` for instance creates the visualzation seen in the 'polyrhythm' video.  I will probably add more over time.
5. The output png files will go into '/frames'.  You can use ffmpeg (free download) to create a video out of the frames:
``` batch
ffmpeg -r 5  -start_number 1 -i "image%04d.png" -c:v libx264 out.mp4
```
The first number (-r) is 'frames per second'.  -start_number is the first frame (the number will depend on frame rate and when first audio appears on the file).

Also, it stores the FFT information in the `/json` directory.  So it will only actually read the wav file the first time, if the json file is present it will use that.

Other files:
* `audioSpectrum.py` does the FFT/energy calculations and runs the visualizer for each audio buffer
* `impulse.py` has the logic to store the energe (volume)
* `audioFrequency.py` stores information about the frequency/notes
* `noteFreq.py` computes the equal-temperment freqencies of all the notes.
* `frameObject.py` contains the different visualizers.

Other files are demo programs/experiments and can be ignored.


 
