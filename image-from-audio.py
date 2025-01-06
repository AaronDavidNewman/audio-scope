import os
import shutil
import requests
import base64
import io
import numpy as np
from createMasks import getSeconds, createBlurredCircle, createFftMasks
from PIL import Image, ImageFilter

# some ffmpeg commands to use:
# ffmpeg -i file.mpg -r 1/1 filename%04d.png
# ffmpeg -r 5  -start_number 1 -i "image%04d.png" -c:v libx264 out.mp4
# ffmpeg -i out.mp4 -vf minterpolate=fps=24 output-24.mp4
# This is the file that generates the SD images.  It assumes that masks have already been created
# from the sound file by createMasks.py.
IMAGESIZE = 512
FPS = 5
def getImageEncoding(fn):
  with open(fn, "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
  return encoded_string
def getBlendedImage(fn1, fn2, alpha):
  global testnum
  with Image.open(fn2) as image2:
    if fn1 in convertedImages:
      image1 = convertedImages[fn1]
    else:
      with Image.open(fn1) as image1:
        image1 = image1.resize((IMAGESIZE, IMAGESIZE)).convert("RGB")
        convertedImages[fn1] = image1

    blended = Image.blend(image1, image2, alpha)
    return blended

def blendImages(fn1, fn2, alpha):
    blended = getBlendedImage(fn1, fn2, alpha)
    buffer = io.BytesIO()
    blended.save(buffer, format="PNG")
    img_data = base64.b64encode(buffer.getvalue())
    return img_data.decode('utf-8')
  
def callApi(imgfile, prompt, destImage, denoise, mask, invert, seed):
  url="http://127.0.0.1:7860/sdapi/v1/img2img"
  payload = {
    "prompt": prompt,
    "steps": 20,
    "denoising_strength": denoise,
    "cfg_scale": 7.5,
    "mask": mask,
    "seed": seed,
    "sampler_name": "DPM++ 2M Karras",
    "inpaint_full_res": False,
    "inpainting_mask_invert": invert, 
    "inpainting_fill": 1,
    "mask_blur": 4
  }

  payload['init_images']=[imgfile]

  response = requests.post(url=url, json=payload)
  jr = response.json()
  for i in jr['images']:
    image = Image.open(io.BytesIO(base64.b64decode(i.split(",",1)[0])))
    image.save(destImage)

def clearDir(dir):
  for fn in os.listdir(dir):
    filepath = os.path.join(dir, fn)
    if os.path.isfile(filepath):
        os.unlink(filepath)

# If this is false, we reuse the existing images
createMasks = False
maskFolder = './masks/'
inputWav = "poly1.wav"
destFolder = './output/'
inputImageFolder = './input/'
srcImageFolder = './input-ordered/'
clearDir(srcImageFolder)
imageFiles = os.listdir(inputImageFolder)
# imageFiles.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
imageFiles.sort()
filenum = 1
for imageFile in imageFiles:
  srcName = f'{inputImageFolder}{imageFile}'
  newName = f'{srcImageFolder}f{filenum:04d}.png'
  with Image.open(srcName) as image2:
    image2 = image2.resize((IMAGESIZE, IMAGESIZE)).convert("RGB")
    image2.save(newName)
    filenum += 1

if createMasks:
  createFftMasks(inputWav)

audioSeconds = getSeconds(inputWav)
convertedImages = {}
clearDir(destFolder)
      
prompts = {
        1: "a colorful object is shown in a circular shape, Alfred Manessier, triadic color scheme, a raytraced image, generative art",
        2: "a colorful flower with different colored petals, glowing, rainbow hues",
        3: "an oil painting of a colorful flower with different colored petals, glowing, rainbow hues",
        4: "a colorful bird, photograph of a tropical bird with bright feathers, yellow beak, rainforest, rainbow hues",
        5: "a colorful picture of a cell under a microscope, tentacles on the call wall, dendrites, neuron",
        6: "a 3-dimensional image of an intricate fractal landscape of cells connected through tentacles that pulse and are wet and pulse with life, photo-realistic, movie image",
        7: "a heart shaped object with a rainbow colored pattern on it's side with a reflection of the heart, Benoit B. Mandelbrot, caustics, an abstract sculpture, generative art",
        8: "a heart shaped object with a rainbow colored pattern on it's side with a reflection of the heart, Benoit B. Mandelbrot, caustics, an abstract sculpture, generative art",
        9: "a computer generated image of a spiral of colored ribbons in 3 dimensions, Manessier, c4d, an abstract sculpture, generative art"
    }

promptCount = len(prompts)
mask_count = len(os.listdir(maskFolder))

sdSkip = 5
sd_image_count = FPS * audioSeconds
srcImageCount = len(os.listdir(srcImageFolder))
iterationsThisPrompt = 0
maxIterations = int(np.max(np.array([sd_image_count, mask_count])))
promptSpace = np.linspace(1, promptCount + 1, maxIterations)
maskSpace = np.linspace(1, mask_count, maxIterations)
sdImageSpace = np.linspace(1, sd_image_count, maxIterations)
srcImageSpace = np.linspace(1, srcImageCount, maxIterations)
sdMaskLifetime = (maxIterations // srcImageCount) + 1
invert = 0
print(f'prompt lifetime {sdMaskLifetime}')
lastPrompt = prompts[1]
lastImageName=''
blendedFiles = ['', '']
# higher number gets more 'creative' output
denoise_max = 0.8
denoise_min = 0.6
denoise = denoise_min
denoise_delta = 0.05
framesSinceChange = 0
sdDestFn = None
for x in range(maxIterations):
  # the prompt used for stable diffusion
  # move the prompt before the input image, assuming they have the same cardinality
  promptIxIx = np.max(np.array(promptCount, np.min(np.array([x, maxIterations - 1]))))
  promptIx = int(promptSpace[promptIxIx])
  # the source of original art to create diffusions from
  srcImageIx = int(srcImageSpace[x])
  sdImageIx = int(sdImageSpace[x])
  # the source of masks generated for the sound file.  We assume the mask
  # and sd image size is the same since there is an image generated for every sample.
  # i.e. there could be many frames for a single sd image and mask
  maskIx = int(maskSpace[x])
  # ratRadius is just a number 0-1 that says how close we are to the next input image
  ratRadius =  (np.sin(2 * np.pi * (np.double(iterationsThisPrompt) / np.double(sdMaskLifetime)) + np.pi/2) + 1) / 2

  if x > sdSkip:
    iterationsThisPrompt += 1
  if iterationsThisPrompt > sdMaskLifetime:
    iterationsThisPrompt = 0

  prevMaskIx = maskIx
  maskFn = f'{maskFolder}figure_{maskIx:04d}.png'
  filterImage = Image.open(maskFn)
  blurbuf = io.BytesIO()
  filterImage.save(blurbuf, format='png')
  img_data = base64.b64encode(blurbuf.getvalue())
  filterMask = img_data.decode('utf-8')
  prompt = prompts[promptIx]
  # increase the volatility of the image as we get closer to min
  denoise = denoise_min + (denoise_max - denoise_min) * ratRadius
  if prompt != lastPrompt:
    print(f'prompt is now {prompt}, image is {srcImageIx} frame')

  lastPrompt = prompt
  srcFn = f'{srcImageFolder}f{srcImageIx:04d}.png'
  sdDestFn = f'{destFolder}img{sdImageIx:04d}.png'
  if srcFn != lastImageName:
    print(f'now using image {srcFn}')
    lastImageName = srcFn
  radius = np.double(IMAGESIZE) * ratRadius * 2
  radius = np.max(np.array([radius, 8.0]))
  if x % sdSkip == 0 or len(blendedFiles[0]) == 0 or len(blendedFiles[1]) == 0:
    if len(blendedFiles[0]) > 0 and len(blendedFiles[1]) > 0:
      srcImage = blendImages(blendedFiles[0], blendedFiles[1], 0.5)
    else:
      srcImage = getImageEncoding(srcFn)
      blendedFiles[0] = srcFn
    print(f'mask radius is {radius} denoise is {denoise} x is {x}')
    callApi(srcImage, prompt, sdDestFn, denoise, filterMask, invert, seed=x)
    invert = (1 + invert) % 2
    blendedFiles[1] = blendedFiles[0]
    blendedFiles[0] = sdDestFn
  else:
    fn1 = blendedFiles[0]
    fn2 = blendedFiles[1]
    blended = getBlendedImage(fn1, fn2, 0.5)
    blended.save(sdDestFn)
    blendedFiles[1] = sdDestFn
    # blendedFiles[0] = sdDestFn





