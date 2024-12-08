import os
import requests
import base64
import io
import numpy as np
from PIL import Image, PngImagePlugin


def getImageEncoding(fn):
  with open(fn, "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
  return encoded_string

def blendImages(fn1, fn2):
  global testnum
  with Image.open(fn2) as image2:
    if fn1 in convertedImages:
      image1 = convertedImages[fn1]
    else:
      with Image.open(fn1) as image1:
        imgsize = int(np.min(np.array([image1.width, image1.height])))
        image1 = image1.crop((0,0,imgsize, imgsize))
        image1 = image1.resize((512, 512)).convert("RGB")
        convertedImages[fn1] = image1

    blended = Image.blend(image1, image2, 0.2)
    buffer = io.BytesIO()
    blended.save(buffer, format="PNG")
    img_data = base64.b64encode(buffer.getvalue())
    return img_data.decode('utf-8')
  
def callApi(imgfile, prompt, destImage, denoise):
  url="http://127.0.0.1:7860/sdapi/v1/img2img"
  payload = {
    "prompt": prompt,
    "steps": 20,
    "denoising_strength": denoise,
    "cfg_scale": 12,
    "seed": 2611919718,
    "sampler_name": "DPM++ 2M Karras",
    "inpaint_full_res": False,
    "inpainting_mask_invert":0, 
    "inpainting_fill": 0,
    "mask_blur": 4
  }

  payload['init_images']=[imgfile]

  response = requests.post(url=url, json=payload)
  jr = response.json()
  for i in jr['images']:
    image = Image.open(io.BytesIO(base64.b64decode(i.split(",",1)[0])))
    image.save(destImage)
  print(f"status is {response.status_code}")

def clearDir(dir):
  for fn in os.listdir(dir):
    filepath = os.path.join(dir, fn)
    if os.path.isfile(filepath):
        os.unlink(filepath)

createSd = False
destFolder = './output/'
sdDestFolder = './sdOutput/'
framesFolder = './frames/'
srcImageFolder = './input/'
maskFolder = './masks/'
convertedImages = {}
clearDir(destFolder)
if (createSd):
  clearDir(sdDestFolder)
      
prompts = {
        1: " oil painting portrait of a man with a grey beard and blue eyes, is looking at the camera with a serious look on his face, grey hair, wearing a cotton sweater  --neg ",
        2: " a man with white hair and green eyes wearing a black shirt and a white wig with green eyes and a red and orange hair, Cedric Peyravernay, ultra realistic faces, a character portrait, computer art --neg ",
        3: " a picture of a demon with green eyes and a demon face with flames around it's eyes and a demon's head, Cedric Seaut (Keos Masons), blender and photoshop, a character portrait, shock art --neg ",
        4: " abstract watercolor painting portrait of a man with a grey beard and thin mustache, blue eyes, is looking at the camera with a serious look on his face, whispy grey hair, wearing a sweater with many colors a blue shirt underneath, computer painting --neg ",
        5: " a picture of a demon with green eyes and a demon face with flames around it's eyes and a demon's head, Cedric Seaut (Keos Masons), blender and photoshop, a character portrait, shock art --neg ",
        6: " a drawing of a demon with its mouth open and tongue out in front of a black background with red and orange streaks, Edvard Munch, horror theme, a cave painting, shock art --neg ",
        7: " a horror movie still of a man with his mouth open, glowing, green eyes with bloody red pupils, an elongated, stretched face, head with green and red eyes, bright orange and black fire background, horror, alien, realistic, 3-d animation --neg ",
        8: " a drawing of a creepy clown with with a hideously swollen head and green eyes and a red nose and a green and white hair with a red eye, Adam Manyoki, face enhance, concept art, sots art --neg ",
        9: " a drawing of a man with a scary face, in a fire, green and red eyes, long mouth and chin, horror, ultra-realistic face --neg ",
        10: " a movie still of a demon with a hideous long mouth, its mouth open,  green glowing eyes, red pupils, in front of a black background surrounded by red and orange flame, realistic faces --neg ",
        11: " a horror movie still of a man with his mouth open, glowing, green eyes with bloody red pupils, an elongated, stretched face, head with green and red eyes, bright orange and black fire background, horror, alien, realistic, 3-d animation --neg ",
        12: " oil painting portrait of a man with a grey beard and blue eyes, is looking at the camera with a serious look on his face, grey hair, wearing a cotton sweater  --neg ",
        13: " a horror movie still of a man with his mouth open, glowing, green eyes with bloody red pupils, an elongated, stretched face, head with green and red eyes, bright orange and black fire background, horror, alien, realistic, 3-d animation --neg "
    }
sourceImageCount = 363
promptCount = 13
mask_count = 356
srcImageCount = 13
maxIterations = int(np.max(np.array([sourceImageCount, mask_count, promptCount ])))
frameSpace = np.linspace(1, sourceImageCount, maxIterations)
promptSpace = np.linspace(1, promptCount, maxIterations)
maskSpace = np.linspace(1, mask_count, maxIterations)
srcImageSpace = np.linspace(1, srcImageCount, maxIterations)
lastPrompt = prompts[1]
denoise = 0.05
denoise_delta = 0.05
denoise_max = 0.6
framesSinceChange = 0
framesPerImage = int(1 + (356 / 13))
sdDestFn = None
for x in range(mask_count):
  promptIx = int(promptSpace[x])
  frameIx = int(frameSpace[x])
  srcImageIx = int(srcImageSpace[x])
  maskIx = int(maskSpace[x])
  prompt = prompts[promptIx]
  if denoise_delta + denoise > denoise_max or denoise_delta + denoise < 0.05:
    denoise_delta = -1 * denoise_delta
  denoise = denoise + denoise_delta
  if prompt != lastPrompt:
    print(f'prompt is now {prompt}')

  lastPrompt = prompt
  srcFn = f'{srcImageFolder}c{srcImageIx}.png'
  maskFn = f'{maskFolder}figure_{maskIx}.png'
  srcFrameFn = f'{framesFolder}filename{frameIx:04d}.png'  
  if createSd:
    if sdDestFn:
      srcImage = blendImages(srcFn, sdDestFn)
    else:
      srcImage = getImageEncoding(srcFn)

  sdDestFn = f'{sdDestFolder}img{x:04d}.png'
  if createSd:
    print(f'image {x} denoise {denoise}')
    callApi(srcImage, prompt, sdDestFn, denoise)
  
  with Image.open(maskFn) as maskImage, Image.open(sdDestFn) as sdImage, Image.open(srcFrameFn) as frameImage:
      imgsize = int(np.min(np.array([frameImage.width, frameImage.height])))
      frameImage = frameImage.resize((512, 512)).convert("RGB")
      combined = Image.composite(sdImage, frameImage, maskImage)
      destFn = f'{destFolder}image{x:04d}.png'
      combined.save(destFn)


