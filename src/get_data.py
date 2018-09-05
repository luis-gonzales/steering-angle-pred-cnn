import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.switch_backend('agg')           # for AWS

def get_data(f_path):
  
  # Seed the random number generator
  np.random.seed(0)

  # Read in CSV
  data = pd.read_csv(f_path, header=0, dtype={'steering': np.float32})

  angle_adj = 0.25
  csv_entries = []   # Final output where each list element contains path and corresponding angle
  angles_all = []    # Store angles with offset for L/R camera images (implied recovery data) after downsampling
  angles_center = [] # Only store center angles after downsampling

  for k in range(data.shape[0]):
    angle = data.steering[k]
    path_center = 'data/' + data.center[k][33:]
    path_right  = 'data/' + data.right[k][33:]
    path_left   = 'data/' + data.left[k][33:]

    # Downsample (if necessary) and append to appropriate lists
    if angle == 0:
      if np.random.rand() < 0.15:
        csv_entries.append([ path_center, angle ])
        csv_entries.append([ path_left, angle+angle_adj ])
        csv_entries.append([ path_right, angle-angle_adj ])
        angles_center.append(angle)
        angles_all.append(angle)
        angles_all.append(angle+angle_adj)
        angles_all.append(angle-angle_adj)
    else:
  	  csv_entries.append([ path_center, angle ])
  	  csv_entries.append([ path_left, angle+angle_adj ])
  	  csv_entries.append([ path_right, angle-angle_adj ])
  	  angles_center.append(angle)
  	  angles_all.append(angle)
  	  angles_all.append(angle+angle_adj)
  	  angles_all.append(angle-angle_adj)
  
  font_sz = 8.5
  
  # Plot histogram of original angles and downsampled angles
  f = plt.figure(figsize=(2.75,4.5))

  ax = f.add_subplot(2,1,1)
  ax.grid(axis='y')
  ax.set_axisbelow(True) # push axis to background
  ax.hist(data.steering, bins=90, edgecolor='black', color=(0.254, 0.808, 0.98))
  ax.tick_params(axis='both', which='major', labelsize=font_sz)
  ax.text(-1.63, 4000,'(a)', fontsize=font_sz)
  ax.set_xlim(-1, 1)

  ax = f.add_subplot(2,1,2)
  ax.grid(axis='y')
  ax.set_axisbelow(True) # push axis to background
  ax.hist(angles_center, bins=60, edgecolor='black', color=(0.254, 0.808, 0.98))
  ax.tick_params(axis='both', which='major', labelsize=font_sz)
  ax.text(-1.63, 1000, '(b)', fontsize=font_sz)
  ax.set_xlim(-1, 1)
  plt.xlabel('Steering angle', fontsize=font_sz)
  plt.savefig('plot_outputs/before.svg', dpi = 300, transparent=True, bbox_inches='tight', pad_inches=0.01)

  # Plot histogram of final angles (angles_all), including L/R adj and downsampling
  f = plt.figure(figsize=(2.75,2))

  ax = f.add_subplot(1,1,1)
  ax.grid(axis='y')
  ax.set_axisbelow(True) # push axis to background
  ax.hist(angles_all, bins=60, edgecolor='black', color=(0.254, 0.808, 0.98))
  ax.tick_params(axis='both', which='major', labelsize=font_sz)
  ax.set_xlim(-1-angle_adj, 1+angle_adj)
  ax.set_ylim(0, 5000)
  plt.xlabel('Steering angle', fontsize=font_sz)
  plt.savefig('plot_outputs/after.svg', dpi = 300, transparent=True, bbox_inches='tight', pad_inches=0.01)

  
  return csv_entries
