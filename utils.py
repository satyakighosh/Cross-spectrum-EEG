from sklearn.model_selection import train_test_split

def get_len_dict(eeg_dict):  
  len_dict = {}
  for i in eeg_dict.keys():
    len_dict[i] = len(eeg_dict[i])
  print("{" + "\n".join("{!r}: {!r},".format(k, v) for k, v in len_dict.items()) + "}")


def split_dataset(dic, svm_id, test_size)      
'''dic -> ref_label wise list of 
  (selected_seg_label, avg feature_vector of ref_label: selected_seg X ref_seg)
  
  svm_id -> signifies which SVM this data is meant for
  '''
  for tup in dic[svm_id]:   
    Y.append(tup[0])
    X.append(tup[1])
  return train_test_split(X, Y, test_size)