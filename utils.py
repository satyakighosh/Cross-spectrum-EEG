from sklearn.model_selection import train_test_split
import numpy as np
from constants import NUM_SLEEP_STAGES

def remove_nan_tuples(ref_label):

  import math
  import array

  e=np.load(f'/content/drive/My Drive/cross/Cross-spectrum-EEG-master/datasets/uncleaned_training_data/clf{ref_label}.npy',allow_pickle=True)

  d=e.tolist()
  count=0
  no_of_features=len(d[0][1])
  print(np.shape(d))
  nan_tuples=[]

  for i in range(np.shape(d)[0]):

    c=d[i][1]
    try:
      if math.isnan(c[0]):
        print(f"NAN found at {i}th tuple, feature number: {j}")
    except IndexError: 
        print(f"nan found at {i}th tuple")
        nan_tuples.append(i)
        count+=1
        print(f"Shape of data_list: {np.shape(d)}")
  print(f"Total nan tuples encountered:{count}:")
  print(np.unique(nan_tuples))

  for i in range(np.shape(d)[0]-count):

    c=d[i][1]
    try:
      if math.isnan(c[0]):
        print(i)
    except IndexError: 
        count+=1
        d.pop(i)
        i=i-1
        print(f"Shape of data_list: {np.shape(d)}")
  return d


def correntropy(x, y):
    #N = len(x)
    X=preprocess(x)
    Y=preprocess(y)
    s=np.std(X, axis=0)
    #print(f"std dev: {s}")
    V =  np.exp(-0.5*np.square(X - Y)/s**2)
    #CIP = 0.0 # mean in feature space should be subtracted!!
    #for i in range(0, N):
        #CIP += np.average(np.exp(-0.5*(x- y[i])**2/s**2))/N
    return V


def get_sums(W):
  path = '/content/drive/My Drive/cross/Cross-spectrum-EEG-master/datasets/matrix_masks/'
  
  row_mask = np.load(path + 'row_mask2.npy', allow_pickle=True)  #mask matrices have fixed shape for same scale and time i.shape/j.shape=(263,3750)
  column_mask = np.load(path + 'column_mask2.npy', allow_pickle=True)  #for dj=1/24 and 30 second segments
  
  accum = np.multiply(W, np.multiply(row_mask, column_mask))
  accum = np.sum(accum)
  accum_sq = np.multiply(W, np.multiply(row_mask**2, column_mask**2))
  accum_sq = np.sum(accum_sq)
  
  return accum, accum_sq


def get_sums2(total_scales, total_time, W):

  ones = np.ones((total_scales, total_time))
  x = np.arange(total_scales).reshape(-1, 1)
  y = np.arange(total_time).reshape(1,-1)
  i = np.multiply(ones, x).astype(int)
  j = np.multiply(ones, y).astype(int)
  accum = np.multiply(W, np.multiply(i, j))  #kind of masking
  accum = np.sum(accum)
  accum_sq = np.multiply(W, np.multiply(i**2, j**2))
  accum_sq = np.sum(accum_sq)
  
  return accum, accum_sq



def softmax(z):
  assert len(z.shape) == 2
  s = np.max(z, axis=1)
  s = s[:, np.newaxis] # necessary step to do broadcasting
  e_x = np.exp(z - s)
  div = np.sum(e_x, axis=1)
  div = div[:, np.newaxis] # dito
  return e_x / div



def get_len_dict(eeg_dict):  
  len_dict = {}
  for i in eeg_dict.keys():
    len_dict[i] = len(eeg_dict[i])
  print("{" + "\n".join("{!r}: {!r},".format(k, v) for k, v in len_dict.items()) + "}")



def get_X_test(dic):
  X_0 = []
  X_1 = []
  X_2 = []
  X_3 = []
  X_4 = []
  X_5 = []

  for tup in dic[0]:   
    X_0.append(tup[1])
  for tup in dic[1]:   
    X_1.append(tup[1]) 
  for tup in dic[2]:   
    X_2.append(tup[1]) 
  for tup in dic[3]:   
    X_3.append(tup[1]) 
  for tup in dic[4]:   
    X_4.append(tup[1]) 
  for tup in dic[5]:   
    X_5.append(tup[1])  

  X = [X_0, X_1, X_2, X_3, X_4, X_5]
  return X


def get_Y_test(dic):
  svm_id = np.random.randint(NUM_SLEEP_STAGES) #emphasizing that it doesn't  matter which ref label we'll use because the label of the randomly selected sample will be same for all keys in the dict
  Y = []
  for tup in dic[svm_id]:   
    Y.append(tup[0]) 
  return Y



def split_dataset(dic, svm_id):     
  """dic -> ref_label wise list of 
  (selected_seg_label, avg feature_vector of ref_label: selected_seg X ref_seg)
  
  svm_id -> signifies which SVM this data is meant for
  """
  X = []
  Y = []
  for tup in dic[svm_id]:   
    Y.append(tup[0])
    X.append(tup[1])

  X = np.array(X)
  Y = np.array(Y)
  print("Original labels:")
  print(np.unique(Y, return_counts=True))
  # print(f"svm_id:{svm_id}")
  pos_indices = np.where(Y == svm_id)[0]
  Y[np.where(Y != svm_id)[0].tolist()] = -1
  Y[np.where(Y == svm_id)[0].tolist()] = 1
  Y[np.where(Y == -1)[0].tolist()] = 0
  assert np.all(np.where(Y == 1)[0] == pos_indices)
  print("Binarized labels:")
  print(np.unique(Y, return_counts=True))
  return X, Y



def split_datalist(data_list, svm_id):     
  """
  np.ndarray -> list of (selected_seg_label, avg feature_vector of ref_label: selected_seg X ref_seg)
  svm_id -> signifies which SVM this data is meant for
  """
  #X = np.array(data_list[:, 1])
  X = np.array(list(data_list[:, 1]), dtype=np.float)
  Y = np.array(data_list[:, 0]).astype('int')
  # for tup in data_list:   
  #   Y.append(tup[0])
  #   X.append(tup[1])

  print("Original labels:")
  print(np.unique(Y, return_counts=True))

  # print(f"svm_id:{svm_id}")
  pos_indices = np.where(Y == svm_id)[0]
  Y[np.where(Y != svm_id)[0].tolist()] = -1
  Y[np.where(Y == svm_id)[0].tolist()] = 1
  Y[np.where(Y == -1)[0].tolist()] = 0

  assert np.all(np.where(Y == 1)[0] == pos_indices)

  print("Binarized labels:")
  print(np.unique(Y, return_counts=True))

  print(X.shape)
  print(Y.shape)
  
  return X, Y




def preprocess(X):
  #data = (X - np.min(X, axis=0))/(np.max(X, axis=0) - np.min(X, axis=0))
  #data = X/np.max(X, axis=0)
  data = (X - np.mean(X, axis=0))/np.std(X, axis=0)
  return data



def check_all_segments_for_given_patient(patient_no):
  
  current_patient = patient_list[patient_no]  
  patient_ann = current_patient[:-4] + '-nsrr.xml'
  path=TRAIN_DATA_PATH + current_patient

  raw = mne.io.read_raw_edf(path, verbose=False)
  data = raw.get_data(picks=['EEG(sec)'])    #taking 3rd channel(EEG)
  
  x = data.tolist()[0]

  for i in range(len(x)):
    if math.isnan(x[i]):
      print(f"patient{patient_no},seg no{i},{x[i]}")

