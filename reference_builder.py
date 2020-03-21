import os
import time
import numpy as np


start = time.time()

class ReferenceBuilder:

  def __init__(self, size, data_path, ann_path):
    self.size = size
    self.data_path = data_path
    self.ann_path = ann_path
    self.reference_segments = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[]}        #contains 5 reference segments for each label

  def is_dict_ok(self, eeg_dict):         #whether the patient has segments of all types of sleep stages
    len_dict = {}
    for label in eeg_dict.keys():
      if len(eeg_dict[label]) == 0: return False
      len_dict[i] = len(eeg_dict[label])
    #print(len_dict)
    return True

  def get_ref_dicts(self):
    fit_eeg_dicts = []
    refs = []
    ann_refs = []

    while len(fit_eeg_dicts) < self.size:
      ref = list(np.random.choice(os.listdir(self.data_path), size=1, replace=False))
      ref = ref[0]
      #print(ref)
      ann_ref = ref[:-4] + '-nsrr.xml'
      ann, onset, duration = extract_annots(self.ann_path + ann_ref)
      eeg_dict, info_dict = extract_data(self.data_path + ref, ann, onset)
      if self.is_dict_ok(eeg_dict):               #guarantees selection of dict with all labels present
        fit_eeg_dicts.append(eeg_dict)
        refs.append(ref)
        ann_refs.append(ann_ref)
    #print(refs)
    #print(ann_refs)
    return fit_eeg_dicts      

  def each_reference_data(self, eeg_dict):
    ref_data = {}
    ref_info = {}

    for label in range(NUM_SLEEP_STAGES):

      #print(f"label:{label}")
      #print(f"len_eeg[label]: {len(eeg_dict[label])}")
      seg_no = int(np.random.choice(len(eeg_dict[label]), size=1, replace=False))
      #print(f"seg_no selected:{seg_no}")
      #print("*****************************")
      #print(f"eeg_dict[label][seg_no]:{eeg_dict[label][seg_no]}")
      ref_data[label] = eeg_dict[label][seg_no]
      ref_info[SLEEP_STAGES_INV[label]] = len(ref_data[label])
    #print("*****************************")
    return ref_data, ref_info


  def build_refs(self):
    
    info_dicts = []
    fit_eeg_dicts = self.get_ref_dicts()

    for ref_patient in range(self.size):
      ref_data, ref_info = self.each_reference_data(fit_eeg_dicts[ref_patient])
      info_dicts.append(ref_info)
      for label in range(6):
        self.reference_segments[label].append(ref_data[label])



data_path = r"/content/drive/My Drive/NSRR/Data/"
ann_path = r"/content/drive/My Drive/NSRR/Annotations/"
size=5
ref_builder = ReferenceBuilder(size, data_path, ann_path)
ref_builder.build_refs()
print(f"Reference building took {time.time()-start} seconds")
