import numpy as np

def get_data(data, labels, lstm_units, sampleL):
 
  fin = open("/home/rajshekd/Projects/IOT/data/ShortDistanceLOSDataCentered")
  lines = fin.readlines()
  lines = [line.split('\n')[0] for line in lines]
  num_lines = len(lines) 
  print "no. of lines = ", num_lines
  
  numDevices = int(lines[0])
  deviceID = lines[1].split(',') 
  #print "device IDs = ", deviceID
  
  #compute the Permutation Mstrices
  splitL = sampleL/lstm_units
  realP = np.zeros([2*splitL, splitL])
  imagP = realP  
  for c in range(splitL):
    realP[2*c][ c] = 1
    imagP[2*c + 1][ c] = 1
  
  
  for i in range(num_lines):    
    if i<3:
      continue
    elem = lines[i].split(',')
    real = np.zeros(sampleL)
    imag = np.zeros(sampleL)
    
    #fill the real/imag parts in the padded seq
    real[0:20736] = elem[3:20739]
    imag[0:20736] = elem[20739:len(elem)]
    #print "real: ", real[0], real[20735], real[20736], real[20999]
    #print "imag: ", imag[0], imag[20735], imag[20736], imag[20999]
    
    '''
    #normalize data 
    real = real/np.max(np.abs(real))
    imag = imag/np.max(np.abs(imag))
    '''
    
    #divide into chunks 
    real_split = np.split(real, lstm_units)
    imag_split = np.split(imag, lstm_units)
    
    
    #form 2D tensor
    list_iter = []
    for j in range(lstm_units):
      '''
      print real_split[j][0:3]
      print imag_split[j][0:3]      
      list_iter.append((np.matmul(realP, real_split[j]) + np.matmul(imagP, imag_split[j])).tolist())
      print list_iter[len(list_iter)-1][0:6]
      exit(1)
      '''
      list_iter.append(real_split[j].tolist() + imag_split[j].tolist()) 
    
      
    #print "dimensions of list_iter: ", len(list_iter), " ", len(list_iter[0])     
    # correct dimensions
    #np.array(list_iter).transpose().tolist()

    #create device label
    vec = np.zeros(numDevices).astype(int).tolist()
    vec[deviceID.index(elem[0])]= 1
    #print "labels: ", vec
    labels.append(vec)
    #create signal vector
    data.append(list_iter)    
    #break
    
    if i%100==0: 
      print i
    
  #print "data after : ", len(data), len(labels)  
  fin.close()
  return numDevices
  
def add_noise(data, psnr):
  factor = np.power(10,np.divide(psnr,10.0)) 
  print psnr, np.divide(psnr,10.0), factor   
  for c in range(len(data)):
    for i in range(len(data[c])):
      arr = np.array(data[c][i])
      power = np.sum(np.square(arr))/arr.shape[0]
      #print power              
      var = np.divide(power, factor)
      #print i, ': ',power, ': ', var
      noise = np.random.normal(0, np.sqrt(var), len(arr))
      #print noise.shape
      data[c][i] = (arr + noise).tolist()
  