import numpy as np
import time
import os
import random
from collections import Counter
from numpy import savez_compressed
import tables


from extract import extract_anns, extract_data
from reference_builder import ReferenceBuilder
from constants import *
#from features import feature_gen
from utils import get_len_dict


start = time.time()

patient_list = sorted(os.listdir(TRAIN_DATA_PATH))

def specific_patient_dataset(patient_no):
  current_patient=patient_list[patient_no]
  patient_ann = current_patient[:-4] + '-nsrr.xml'
  ann, onset, duration = extract_anns(TRAIN_ANN_PATH + patient_ann)
  eeg_dict, info_dict = extract_data(TRAIN_DATA_PATH + current_patient, ann, onset)
  len_dict = {}
  eeg_dict_list=[]

  for i in eeg_dict.keys(): 
    len_dict[i] = len(eeg_dict[i])
    if i==4 and len_dict[i]!=0:
      print(f"{len_dict[i]} keys of segment 4 in patient no: {patient_no}")
    random.shuffle(eeg_dict[i])
  eeg_dict_list.append(eeg_dict)
  return eeg_dict_list,len_dict

for p in range(len(patient_list)):
  _,_ = specific_patient_dataset(p)

'''
print(len(patient_list))
#all_patient_dataset={'eeg':[], 'lengths'=[]}
NUM_PATIENT=2
all_patient_dataset=[]
for p in range(NUM_PATIENT):
  eeg,lengths=specific_patient_dataset(p)
  all_patient_dataset.append(eeg)
  print(p)



#print(all_patient_dataset[1][1][1])
#print("done")

filename='all_patients_eeg_data.h5'
f = tables.open_file(filename, mode='w')
atom = tables.Float64Atom()

array_c = f.create_earray(f.root, 'data', atom, (0,1))
eeg_npy=np.array(all_patient_dataset)
print(np.shape(eeg_npy))
#print(eeg_npy[0])

for p in range(NUM_PATIENT):
  array_c.append(eeg_npy[p])

f.close()

print(f"Shape of the whole dataset: {np.shape(all_patient_dataset)}")
print("Saving dataset")
#np.savez_compressed('/content/drive/My Drive/Cross-spectrum-EEG/compressed_all_patients_eeg_data', all_patient_dataset)
'''