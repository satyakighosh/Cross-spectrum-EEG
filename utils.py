from sklearn.model_selection import train_test_split
import numpy as np
from constants import NUM_SLEEP_STAGES
import pandas as pd

def describe(df: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([df.mean().rename('mean'),
                      df.median().rename('median'),
                      df.max().rename('max'),
                      df.min().rename('min')
                     ], axis=1).T

                     
def out_std(s, nstd=3.0, return_thresholds=False):
    """
    Return a boolean mask of outliers for a series
    using standard deviation, works column-wise.
    param nstd:
        Set number of standard deviations from the mean
        to consider an outlier
    :type nstd: ``float``
    param return_thresholds:
        True returns the lower and upper bounds, good for plotting.
        False returns the masked array 
    :type return_thresholds: ``bool``
    """
    data_mean, data_std = s.mean(), s.std()
    cut_off = data_std * nstd
    lower, upper = data_mean - cut_off, data_mean + cut_off
    if return_thresholds:
        return lower, upper
    else:
        return [True if x < lower or x > upper else False for x in s]


def out_iqr(s, k=1.5, return_thresholds=False):
    """
    Return a boolean mask of outliers for a series
    using interquartile range, works column-wise.
    param k:
        some cutoff to multiply by the iqr
    :type k: ``float``
    param return_thresholds:
        True returns the lower and upper bounds, good for plotting.
        False returns the masked array 
    :type return_thresholds: ``bool``
    """
    # calculate interquartile range
    q25, q75 = np.percentile(s, 25), np.percentile(s, 75)
    iqr = q75 - q25
    # calculate the outlier cutoff
    cut_off = iqr * k
    lower, upper = q25 - cut_off, q75 + cut_off
    if return_thresholds:
        return lower, upper
    else: # identify outliers
        return [True if x < lower or x > upper else False for x in s]


def remove_nan(df: pd.DataFrame) -> pd.DataFrame:

  if df.isnull().values.any():
    print(f'Data not OK, removing nan values..')
    print()
    nan_values = []
    indices = list(np.arange(NUM_FEATURES))
    for j in range(df.shape[1]):
      nan_values.append(df[j].isnull().sum().sum())
    
    print(f'Before:')
    print(f"Indices:    {indices}")      #index of feature   
    print(f"NaN values: {nan_values}")   #number of nan values corresponding to each feature
    print()

    df = df.fillna(df.median())  #replacing nan with median

    nan_values = []
    indices = list(np.arange(NUM_FEATURES))
    for j in range(df.shape[1]):
      nan_values.append(df[j].isnull().sum().sum())

    print(f'After:')
    print(f"Indices:    {indices}")        #index of feature
    print(f"NaN values: {nan_values}")     #number of nan values corresponding to each feature
    print()

  else:
    print(f"Data has no NaN values")
  
  return df




def remove_nan_tuples():

  import math
  import array
  cleaned_testset = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[]}
  testdict=np.load(f'/content/drive/My Drive/Cross-spectrum-EEG/datasets/test_set_balanced.npy',allow_pickle=True)
  e=testdict.item()

  for ref in range(6):
    d=e[ref]
    #d=e[ref].tolist()
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
    cleaned_testset[ref].extend(d)
  np.save(f"/content/drive/My Drive/Cross-spectrum-EEG/datasets/test_set_cleaned.npy", cleaned_testset)

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


#@profile
def get_sums(W):
  path = '/content/matrix_masks/'
  
  row_mask = np.load(path + 'row_mask.npy', allow_pickle=True)  #mask matrices have fixed shape for same scale and time i.shape/j.shape=(263,3750)
  column_mask = np.load(path + 'column_mask.npy', allow_pickle=True)  #for dj=1/24 and 30 second segments
  
  accum = np.multiply(W, np.multiply(row_mask, column_mask))
  accum = np.sum(accum)
  accum_sq = np.multiply(W, np.multiply(row_mask**2, column_mask**2))
  accum_sq = np.sum(accum_sq)
  
  return accum, accum_sq


#@profile
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



def split_dataset(data_dict: dict) -> [np.ndarray, list]:
  """
  data_dict -> labelwise dict of (selected_seg_label, avg feature_vector of ref_label: selected_seg X ref_seg)
  """
  X_0, X_1, X_2, X_3, X_4, X_5 = ([] for _ in range(NUM_SLEEP_STAGES)) #initializing 6 empty strings
  X = [X_0, X_1, X_2, X_3, X_4, X_5]
  
  for i in range(NUM_SLEEP_STAGES):
    for tup in data_dict[i]:   
      X[i].append(tup[1]) 

  Y = []
  clf_id = np.random.randint(NUM_SLEEP_STAGES) #emphasizing that it doesn't  matter which ref label we'll use because the label of the randomly selected sample will be same for all keys in the dict
  for tup in data_dict[clf_id]:   
    Y.append(tup[0]) 

  return np.array(X), Y         #(num_sleep_stages, total_samples, num_features), num_samples



def split_datalist(data_list: np.ndarray, clf_id: int) -> [np.ndarray, np.ndarray]:    
  """
  data_list -> list of (selected_seg_label, avg feature_vector of ref_label: selected_seg X ref_seg)
  clf_id -> signifies which SVM this data is meant for
  """
  # print(f"clf_id:{clf_id}")

  X = np.array(list(data_list[:, 1]), dtype=np.float)
  Y = np.array(data_list[:, 0]).astype('int')

  # print("Original labels:")
  # print(Y)

  pos_indices = np.where(Y == clf_id)[0]
  Y[np.where(Y != clf_id)[0].tolist()] = -1
  Y[np.where(Y == clf_id)[0].tolist()] = 1
  Y[np.where(Y == -1)[0].tolist()] = 0

  assert np.all(np.where(Y == 1)[0] == pos_indices)

  # print("Binarized labels:")
  # print(Y)
  # print(X.shape)
  # print(Y.shape)
  
  return X, Y                #(total_samples, num_featues), (total_samples,) 




#used for training and in correntropy calculation
def preprocess(X: np.ndarray) -> np.ndarray:
  #data = (X - np.min(X, axis=0))/(np.max(X, axis=0) - np.min(X, axis=0))
  m = np.mean(X, axis=0)
  s = np.std(X, axis=0)

  data = (X - m)/s
  return data               #(total_samples, num_featues)


def preprocess_test(X: np.ndarray) -> np.ndarray:
  #data = (X - np.min(X, axis=0))/(np.max(X, axis=0) - np.min(X, axis=0))
  m = np.mean(X, axis=1)[:, np.newaxis, :]
  s = np.std(X, axis=1)[:, np.newaxis, :]
  
  data = (X - m)/s
  return data                   #(num_sleep_stages, total_samples, num_features)



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


def average_over_ten_references():
  
  test_set=np.load('/content/drive/My Drive/Cross-spectrum-EEG/datasets/test_set.npy',allow_pickle=True)
  length=np.shape(test_set.item()[0])[0]
  testset_dict = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[]}
  for ref in range(6):
    #F_avg=[]
    for row in range(length):

      #F_avg.append(F)
      if (row+1)%10==0:
        F=test_set.item()[ref][row][1]
        label=test_set.item()[ref][row][0]
        print(label)
        testset_dict[ref].append((label,F))
        #F_avg=[]

  np.save(f"/content/drive/My Drive/Cross-spectrum-EEG/datasets/test_set_balanced_returns.npy", testset_dict)



def avg_no_of_segments_per_patient():

  import os,mne,fnmatch
  import numpy as np

  TRAIN_DATA_PATH = r"/content/drive/My Drive/NSRR/Data/train/"

  TEST_DATA_PATH = r"/content/drive/My Drive/NSRR/Data/test/"

  patient_list=sorted(os.listdir(TRAIN_DATA_PATH))
  total=0
  for patient_no in range(len(patient_list)):
    current_patient_ = patient_list[patient_no]  
    patient_ann_ = current_patient_[:-4] + '-nsrr.xml'
    path=TRAIN_DATA_PATH + current_patient_
    raw = mne.io.read_raw_edf(path, verbose=False)
    channel_names=raw.ch_names
    eeg_names='*EEG*'
    for name in channel_names:
      if fnmatch.fnmatch(name,eeg_names):
        break
    #print(name)
    data = raw.get_data(picks=[name])
    print(f"No. of segments in patient{patient_no} is {np.shape(data)[1]/3750}")
    total+=np.shape(data)[1]/3750

  print(f"Total: {total}")
  print(f"Avg  no. of segments per patient: {total/len(patient_list)}")



remove_nan_tuples()