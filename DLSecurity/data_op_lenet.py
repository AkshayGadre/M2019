import numpy as np

def get_data(data, labels, sampleL):
 
  fin = open("/home/rajshekd/Projects/IOT/data/ShortDistanceLOSDataCentered")
  lines = fin.readlines()
  lines = [line.split('\n')[0] for line in lines]
  num_lines = len(lines) 
  print "no. of lines = ", num_lines
  
  numDevices = int(lines[0])
  deviceID = lines[1].split(',') 
  #print "device IDs = ", deviceID
  
  
  
  for i in range(num_lines):    
    if i<3:
      continue
    elem = lines[i].split(',')
    real = np.zeros(sampleL)
    imag = np.zeros(sampleL)
    
    #fill the real/imag parts in the padded seq
    real[0:20736] = elem[3:20739]
    imag[0:20736] = elem[20739:len(elem)]

    #form 2D tensor
    list_iter = real.tolist() + imag.tolist() 
    
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
  