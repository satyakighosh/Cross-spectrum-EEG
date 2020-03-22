import numpy as np
import pycwt as wavelet


def feature_gen(W12, R12, total_scales, total_time):   #takes abs W12 as input
  accum = 0
  accum_sq = 0
  for i in range(total_scales):
    for j in range(total_time):
      accum += i * j * W12[i, j]
      accum_sq += i**2 * j**2 * W12[i, j]

  W12_sum = np.sum(W12)
  print(f"W12_sum:{W12_sum}")
  f1 = accum/W12_sum
  f2 = np.sqrt(accum_sq/W12_sum)
  f3 = W12_sum/np.max(W12)

  s_min, t_min = np.unravel_index(W12.argmin(), W12.shape)
  s_max, t_max = np.unravel_index(W12.argmax(), W12.shape)
  x = np.absolute((s_max - s_min) * (t_max - t_min))   #doubt
  print(s_min, s_max)
  print(t_min, t_max)
  print(f"x1:{x}")

  f4 = W12_sum/x
  f5 = np.sqrt((np.sum((f4 - W12) ** 2))/x)

  f6 = s_max     #doubt
  f7 = t_max      #doubt
  f8 = s_min      #doubt

  f9 = 5*W12_sum/x
  f10 = np.sum(W12 ** 2)
  f11 = f10/x
  f12 = np.sqrt(f11)
  f13 = W12_sum/np.exp(x)
  #f14 = 
  #f15 = f14/x
  #f16 = np.sqrt((f14**2)/x)
  #f17 = np.log10(f14)
  f14 = "N/A"
  f15 = "N/A"
  f16 = "N/A"
  f17 = "N/A"
  f18 = W12_sum
  f19 = "N/A"
  f20 = "N/A"
  f21 = "N/A"
  f22 = "N/A"
  f23 = "N/A"
  f24 = "N/A"
  f25 = "N/A"

  accum = 0
  accum_sq = 0
  for i in range(total_scales):
    for j in range(total_time):
      accum += i * j * R12[i, j]
      accum_sq += i**2 * j**2 * R12[i, j]

  R12_sum = np.sum(R12)
  print(f"R12_sum:{R12_sum}")
  f26 = accum/R12_sum
  f27 = np.sqrt(accum_sq/R12_sum)

  s_min, t_min = np.unravel_index(R12.argmin(), R12.shape)
  s_max, t_max = np.unravel_index(R12.argmax(), R12.shape)
  x = np.absolute((s_max - s_min) * (t_max - t_min))   #doubt

  print(s_min, s_max)
  print(t_min, t_max)
  print(f"x2:{x}")
  f28 = t_max     #doubt
  f29 = s_min      #doubt
  f30 = R12_sum/np.max(R12) #doubt
  f31 = R12_sum/x
  f32 = np.sqrt((np.sum((f30 - R12)))) #doubt
  f33 = 5*R12_sum/x
  f34 = np.sum(R12 ** 2)

  f35 = f34/x
  f36 = np.sqrt(np.absolute(f35))
  f37 = R12_sum/np.exp(x)
  f38 = "N/A"
  f39 = "N/A"
  f40 = "N/A"
  f41 = "N/A"
  #f39 = f14/x
  #f40 = np.sqrt((f14**2)/x)
  #f41 = np.log10(f14)

  f42 = R12_sum

  f = [f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15,f16,f17,f18,f19,f20,f21,f22,f23,f24,f25,f26,f27,f28,
       f29,f30,f31,f32,f33,f34,f35,f36,f37,f38,f39,f40,f41,f42]
  for i in range(len(f)):
    print(f"f{i+1}: {f[i]}")


dt = np.diff(t1)[0]
W12, cross_coi, freq, signif= wavelet.xwt(s1, s2, dt, dj=1/24)
#R12, aWCT, corr_coi, freq, sig = wavelet.wct(s1, s2, dt, dj=1/24, cache=True)
#assert(W12.shape == R12.shape)  
print(W12)
print("****************************************")
W12 = np.abs(W12)
print(W12)

total_scales = W12.shape[0]
total_time = W12.shape[1]
#print(R12.shape)  #row->scale, col->time

feature_gen(W12, R12, total_scales, total_time)
