from sklearn.model_selection import train_test_split
import numpy as np
from constants import NUM_SLEEP_STAGES

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