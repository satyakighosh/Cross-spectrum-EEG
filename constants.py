SLEEP_STAGES = {'Wake|0':0, 'Stage 1 sleep|1':1, 'Stage 2 sleep|2':2, 'Stage 3 sleep|3':3, 
                'Stage 4 sleep|4':4, 'REM sleep|5':5}
SLEEP_STAGES_INV = {v: k for k, v in SLEEP_STAGES.items()}
SAMPLE_RATE = 125    #fixed for shhs1
NUM_SLEEP_STAGES = 6   #fixed
NUM_SEG_PROCESSED_PER_PATIENT = 50

DATA_PATH = r"/content/drive/My Drive/NSRR/Data/"
ANN_PATH = r"/content/drive/My Drive/NSRR/Annotations/"

REF_SEG_PER_CLASS = 2