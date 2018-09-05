import cv2
import numpy as np
from sklearn.utils import shuffle

# Array outputs are np.array-type. Individual elements of imgs
# are np.uint8-type; angles, np.float32-type
def generator(gen_entries, batch_sz):
  '''
  Output `batch_sz` number of images and corresponding angles from `gen_entries`
  '''
  num_entries = len(gen_entries)
  while 1:
    entries = shuffle(gen_entries)  # shuffle so each epoch sees different ordering
    for offset in range(0, num_entries, batch_sz): # 0, batch_sz, 2*batch_sz, ...
      batch_entries = entries[offset:offset+batch_sz]
      imgs, angles = [], []
      
      # Append raw images to `imgs` and corresponding steering angles to `angles`
      for entry in batch_entries:
        img = cv2.imread(entry[0])[:,:,::-1]
        imgs.append(img)

        angle = np.float32(entry[1])
        angles.append(angle)          

      yield np.array(imgs), np.array(angles)