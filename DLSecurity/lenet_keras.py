from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import numpy as np
import random
#import h5py

from data_op_lenet import *

#preprocess data
def data_preprocess(data):
  #Form batches of data
  #shuffle
  indices=list(range(len(data)))
  random.shuffle(indices)
  return indices

#split train test data
def split_data(indices, pcnt):
  splt = int(round(pcnt*len(indices)))
  return indices[0:splt], indices[splt:len(data)]


#batch generation
def get_batch(batch_size, itr, data, labels, flag):
  indices = data_preprocess(data)
  train_ind, test_ind = split_data(indices, .8)
  '''
  print "indices shape: ", 
  #train_test split 80/20
  splt = int(round(.8*len(data))) 
  train_ind = indices[0:splt]
  test_ind = indices[splt:len(data)]  
  '''
  if flag==1:
    batch_ind = train_ind#train_ind[(itr*batch_size)%len(train_ind):((itr+1)*batch_size)%len(train_ind)]
    print ("train_index length:", len(batch_ind))
  else:
    batch_ind = test_ind
    print ("test_index length: ", len(batch_ind))

  t_data = np.asarray([data[ind] for ind in batch_ind])
  t_labels = np.asarray([labels[ind] for ind in batch_ind])
  return t_data, t_labels


##########################################################
data = []
labels = []
data_dim = 500
sampleL = 21000
num_classes = get_data(data, labels, sampleL)

'''
#add noise
psnr = -2
print data[0][0][0:5]
add_noise(data, psnr)
print data[0][0][0:5]
'''

print(" num_classes: ", num_classes)
batch_size = 32

# Expected input batch shape: (batch_size, timesteps, data_dim)
# Note that we have to provide the full batch_input_shape since the network is stateful.
# the sample of index i in batch k is the follow-up for the sample i in batch k-1.
model = Sequential()
model.add(Dense(512, input_shape=(2*sampleL, ), init='uniform', activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(512, init='uniform', activation='relu'))
model.add(Dropout(0.5))
'''
model.add(Dense(1024, init='uniform', activation='relu'))
model.add(Dropout(0.5))
'''
model.add(Dense(1024, init='uniform', activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes,  activation='softmax'))


model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

#train-test data
# training data
x_train, y_train = get_batch(batch_size, 0, data, labels, 1) #np.random.random((batch_size * 10, timesteps, data_dim))
print ("Shape of train x -> ", x_train.shape)
# validation data
x_val, y_val = get_batch(batch_size, 0, data, labels, 0)#np.random.random((batch_size * 3, timesteps, data_dim))

model.fit(x_train, y_train,
          batch_size=batch_size, epochs=400000, shuffle=False,
          validation_data=(x_val, y_val))
          
          
#save model
model_json = model.to_json()
with open("/home/rajshekd/P rojects/IOT/trained_models/IOT_lstm.model", "w") as json_file:
  json_file.write(model_json)
model.save_weights("model.h5")
print ("saved model")