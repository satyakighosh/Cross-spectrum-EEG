import numpy as np
import pycwt as wavelet
import cmath, math
from utils import get_sums,get_sums2,correntropy
from sklearn.metrics.pairwise import cosine_similarity
from line_profiler import LineProfiler
from scipy.stats import kurtosis,skew,moment,trim_mean
'''
pycwt.xwt()
Parameters:	
y2 (y1,) – Input signal array to calculate cross wavelet transform.
dt (float) – Sample spacing.
dj (float, optional) – Spacing between discrete scales. Default value is 1/12. Smaller values will result in better scale resolution, but slower calculation and plot.
s0 (float, optional) – Smallest scale of the wavelet. Default value is 2*dt.
J (float, optional) – Number of scales less one. Scales range from s0 up to s0 * 2**(J * dj), which gives a total of (J + 1) scales. Default is J = (log2(N*dt/s0))/dj.
wavelet (instance of a wavelet class, optional) – Mother wavelet class. Default is Morlet wavelet.
significance_level (float, optional) – Significance level to use. Default is 0.95.
normalize (bool, optional) – If set to true, normalizes CWT by the standard deviation of the signals.
Returns:	
xwt (array like) – Cross wavelet transform according to the selected mother wavelet.
x (array like) – Intersected independent variable.
coi (array like) – Cone of influence, which is a vector of N points containing the maximum Fourier period of useful information at that particular time. Periods greater than those are subject to edge effects.
freqs (array like) – Vector of Fourier equivalent frequencies (in 1 / time units) that correspond to the wavelet scales.
signif (array like) – Significance levels as a function of scale.
'''
ref_segments = np.load('/content/drive/My Drive/Cross-spectrum-EEG/reference_segments.npy', allow_pickle=True).reshape(-1, 1)[0][0]

profile=LineProfiler()

@profile
def feature_gen(s1, s2, ref_label):   
  
  #print(f"t1={t1}; s1={s1};")
  #print(f"t2={t2}; s2={s2}")
  #print(f"Length of s1={len(s1)}")
  #print(f"Length of s2={len(s2)}")
  #dt = np.diff(t1)[0]
  dt=1/125
  #W12_complex, cross_coi, freq, signif= wavelet.xwt(s1, s2, dt, dj=1/24, normalize=True)
  W12_complex, _, _, _= wavelet.xwt(s1, s2, dt, dj=1/12, normalize=True)                  #TAKING TOO MUCH TIME
  #R12, aWCT, corr_coi, freq, sig = wavelet.wct(s1, s2, dt, dj=1/24, cache=True)
  #find_phase=np.vectorize(cmath.phase)
  W12_phase=np.abs(np.angle(W12_complex))                   #TAKING TOO MUCH TIME
  W12 = np.abs(W12_complex)   #row->scale, col->time
  assert(W12_phase.shape == W12_complex.shape)  
  #print("****************************************")
  # print(W12)

  total_scales = W12.shape[0]
  total_time = W12.shape[1]
  
  accum, accum_sq = get_sums(W12)                             #TAKING TOO MUCH TIME
  accum_phase, accum_sq_phase=get_sums(W12_phase)             #TAKING TOO MUCH TIME

  W12_phase_sum=np.sum(W12_phase)
  W12_sum = np.sum(W12)
  #print(f"W12_sum:{W12_sum}")
  f1 = accum/W12_sum
  f2 = np.sqrt(accum_sq/W12_sum)
  f3 = W12_sum/np.max(W12)

  s_min, t_min = np.unravel_index(W12.argmin(), W12.shape)
  s_max, t_max = np.unravel_index(W12.argmax(), W12.shape)
  x = np.absolute((s_max - s_min) * (t_max - t_min))  #doubt

  eps = 1e-5
  f4 = W12_sum/(x+eps)
  f5 = np.sqrt((np.sum((np.square(f4 - W12))))/(x+eps))

  f6 = s_max     #doubt
  f7 = t_max      #doubt
  f8 = s_min      #doubt

  f9 = 0.5*W12_sum/(x+eps)
  f10 = np.sum(np.square(W12 ))
  f11 = f10/(x+eps)
  f12 = np.sqrt(f11)
  #f13 = np.exp(min(W12_sum/((x+eps)*10),700))
  #print(f"x is{W12_sum/(x*10)}")
  w=np.array(W12)
  s=0
  for i in range(1,total_scales):
    s+=np.sum(np.absolute(w[i,:]-w[i-1,:]))
    #for j in range(total_time):
      #s+=np.square(W12[i][j]-W12[i-1][j])
  consec_scale_diff=s
  '''
  print(f"consec_scale_diff:{consec_scale_diff}")
  s=0
  for j in range(1,total_time):                            
    s+=np.sum(np.absolute(w[:,j]-w[:,j-1]))             #TAKING TOO MUCH TIME
  consec_time_diff=s
  print(f"consec_time_diff:{consec_time_diff}")
  '''
  f14 = consec_scale_diff
  f15 = f14/(x+eps)
  f16 = np.sqrt((f14**2)/(x+eps))
  f17 = np.log10(f14)
  f18 = W12_sum
  f19 = accum_phase/W12_phase_sum
  f20 = np.sqrt(accum_sq_phase/W12_phase_sum)
  f21 = W12_phase_sum
  f22 = W12_phase_sum/(np.max(W12_phase))
  f23 = W12_phase_sum/(x+eps)
  f24 = np.sqrt((np.sum(np.square((f22-W12_phase))))/(x+eps))
  f25 = 0.5*(W12_phase_sum)/(x+eps)
  mean_class_vector=[]
  #for ref_segment in ref_segments[ref_label]:
    #mean_class_vector.append(np.array(ref_segment))
  #print(np.shape(ref_segment))
  #print(f"shape of mean class vector: {np.shape(mean_class_vector)}")  
  #mcv=np.mean(mean_class_vector,axis=0)
  f26 = np.sqrt(np.sum(np.square(s1-s2)))     #euclidean distance between s1 and mcv ?
  f27 = cosine_similarity(s1.reshape(1,-1),s2.reshape(1,-1), dense_output=True).reshape(1,1)[0][0]

  corr=correntropy(s1,s2)
  f28=np.mean(corr)
  f29=kurtosis(corr)
  f30=skew(corr)
  f31=moment(corr,moment=2)
  f32=moment(corr,moment=3)
  f33=trim_mean(corr,0.1)


  f = [f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f14,f15,f16,f17,f18,f19,f20,f21,f22,f23,f24,f25,f26,f27,f28,f29,f30,f31,f32,f33]
  F = []
  for i in f:
    F.append(i)
  # for i in range(len(F)):
  #   print(f"F{i+1}: {F[i]}")
  return np.array(F)