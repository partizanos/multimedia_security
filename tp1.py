import matplotlib.image as mpimg; import matplotlib.pyplot as plt; import numpy as np; import requests; from PIL import Image; from io import BytesIO; import math; from skimage import data


url = "https://github.com/partizanos/multimedia_security/raw/master/TP/TP1/peacock.jpg"
response = requests.get(url)
im = Image.open(BytesIO(response.content))

import PIL
from PIL import Image
from matplotlib import pyplot as plt

# im = Image.open('./color_gradient.png')  
w, h = im.size  
colors = im.getcolors(w*h)

def hexencode(rgb):
    r=rgb[0]
    g=rgb[1]
    b=rgb[2]
    return '#%02x%02x%02x' % (r,g,b)

for idx, c in enumerate(colors):
    plt.bar(idx, c[0], color=hexencode(c[1]))

plt.show()


