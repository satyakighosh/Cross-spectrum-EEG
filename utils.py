from sklearn.model_selection import train_test_split
import numpy as np

def get_len_dict(eeg_dict):  
  len_dict = {}
  for i in eeg_dict.keys():
    len_dict[i] = len(eeg_dict[i])
  print("{" + "\n".join("{!r}: {!r},".format(k, v) for k, v in len_dict.items()) + "}")


def split_dataset(dic, svm_id, test_size):     
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
  print(Y)
  pos_indices = np.where(Y == svm_id)[0]
  Y[np.where(Y != svm_id)[0].tolist()] = -1
  Y[np.where(Y == svm_id)[0].tolist()] = 1
  Y[np.where(Y == -1)[0].tolist()] = 0
  assert np.all(np.where(Y == 1)[0] == pos_indices)
  print(Y)
  return train_test_split(X, Y, test_size=test_size)