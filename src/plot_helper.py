import numpy as np
import matplotlib.pyplot as plt

def plot_helper(history, epochs, alpha, batch_sz):
  fig = plt.figure(figsize = (2.5,2.5))
  
  font_sz = 8.5
  ax = fig.add_subplot(1,1,1)
  
  ax.set_xlabel('Epochs', fontsize=font_sz)
  ax.set_ylabel('Loss', fontsize=font_sz)
  ax.plot(np.arange(1,epochs+1), history.history['loss'], 'b', label='train')
  ax.plot(np.arange(1,epochs+1), history.history['val_loss'], 'r', label='val')
  ax.axis([0, epochs, 0, 0.02])
  ax.tick_params(labelsize=font_sz)
  ax.legend(fontsize=font_sz-1)
  ax.set_yticks([0, 0.005, 0.01, 0.015, 0.02])
  ax.set_yticks([0.0025, 0.0075, 0.0125, 0.0175], minor=True)
  ax.yaxis.grid(which='both')

  file_str = 'plot_outputs/val-loss-' + str(history.history['val_loss'][-1])[0:7] + '-alpha-' + str(alpha)[0:7] + '-batch-' + str(batch_sz) + '.svg'
  plt.savefig(file_str, bbox_inches='tight', dpi = 300, transparent=True)