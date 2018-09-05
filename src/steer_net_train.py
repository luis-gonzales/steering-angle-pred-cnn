import numpy as np
from keras import optimizers
from get_data import get_data
from generator import generator
from steer_net import get_model
import model_comma
from plot_helper import plot_helper
from tensorflow import set_random_seed
from sklearn.model_selection import train_test_split

# Seed the random number generator
np.random.seed(1)
set_random_seed(1)


# Get dataset and split for training and validation sets 
data_entries = get_data('data/driving_log.csv') # each element of list contain image path and corresponding angle
train_entries, val_entries = train_test_split(data_entries, train_size=0.8, random_state=0, shuffle=True)


# Define model and print summary
model = model_comma.model()
#model = get_model()
model.summary()


# Perform training
alpha, batch_sz, epochs = 0.0004, 64, 7                       # hyperparameters

train_generator = generator(train_entries, batch_sz=batch_sz) # define generator for training data
val_generator = generator(val_entries, batch_sz=batch_sz)     # define generator for validation data

train_steps = np.uint32(np.ceil(len(train_entries)/batch_sz)) # steps per epoch (training set)
val_steps = np.uint32(np.ceil(len(val_entries)/batch_sz))     # steps per epoch (validation set)

adam = optimizers.Adam(lr=alpha)
model.compile(loss='mse', optimizer=adam)
history = model.fit_generator(generator=train_generator,
	                            steps_per_epoch=train_steps,
	      			                epochs=epochs,
	      			                validation_data=val_generator,
	      			                validation_steps=val_steps)
print('Training complete!')

# Plot learning curves and save model
plot_helper(history, epochs, alpha, batch_sz)
model.save('model.h5')
print('Save complete!')