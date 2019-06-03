# Read in an image and convert it to grayscale.
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt;
im = Image.open("/home/dimitris/github/multimedia_security/TP/TP1/dct_db/t1_a.tif")

X  = np.array(im)
# plt.imshow(X); plt.show()
# Determine the global mean of the image.

def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

X = rgb2gray(X)
global_mean = np.mean(X)


# For each 32 × 32 subimage non-overlapping block, determine the local mean.
def hash(X):
    k = int(X.shape[0]/32)
    l = int(X.shape[1]/32)
    # print("X.shape")
    # print(X.shape)
    blocks = X.reshape((k,l, 32,32))
    blocks.shape

    print("1st block")
    print(blocks[0].shape)
    print(blocks[0][0][:3,:3])
    local_means = []
    for i, _ in enumerate(blocks):
        for j, el in enumerate(blocks):
            local_means.append(np.mean(blocks[i][j]))

    #For each subimage block determine if the local mean is larger than the global mean. If so, the
    #hash value for this particular block is 1 and 0 otherwise. The resulting descriptor is thus a 64 bit
    #binary vector.

    hash = [1 if local > global_mean else 0 for local in local_means ]
    print(hash[10:25])
    return hash

digest = hash(X)
########
hamming = lambda a,b: np.sum(np.bitwise_xor(np.array(a),np.array(b)) )
assert hamming([1,0,1],[1,1,0]) == 2
prob_err = lambda N,h: h/N

########
# I need to read thing from network because i implemented the TP in cloud environmnent

def getImagesList(folder_name: str) -> list:
    
  imagesFilenames = sorted([folder_name + 'code_dm_' + str(i).zfill(4) + '_imag.bmp' for i in range(1, 151)])
  result = {}
  # Parallel images request
  result = multiprocessing.Manager().dict()

  procs = []
  for i, im in enumerate(imagesFilenames):
    procs.append(multiprocessing.Process(target=getImageProc, args=(result, i+1, im,)))
  [p.start() for p in procs]
  [p.join() for p in procs]
  return np.array([result[i] for i in range(1, 151)])