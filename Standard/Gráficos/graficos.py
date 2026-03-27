import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO

dados1 = np.loadtxt('1_1e-17.txt', dtype=int, delimiter='\t', unpack=True)
dados2 = np.loadtxt('2_5e-17.txt', dtype=int, delimiter='\t', unpack=True)
dados3 = np.loadtxt('3_1e-16.txt', dtype=int, delimiter='\t', unpack=True)
dados4 = np.loadtxt('4_5e-16.txt', dtype=int, delimiter='\t', unpack=True)
dados5 = np.loadtxt('5_7.5e-16.txt', dtype=int, delimiter='\t', unpack=True)
dados6 = np.loadtxt('6_1e-15.txt', dtype=int, delimiter='\t', unpack=True)
dados7 = np.loadtxt('7_2.5e-15.txt', dtype=int, delimiter='\t', unpack=True)
dados8 = np.loadtxt('8_5e-15.txt', dtype=int, delimiter='\t', unpack=True)
dados9 = np.loadtxt('9_7.5e-15.txt', dtype=int, delimiter='\t', unpack=True)

def dist_MWD(dados):
    n = np.shape(dados)
    M = np.zeros([n[0]-1, n[1]], dtype=np.float64)

    for i in range(1, n[0]):
        for j in range(0, n[1]):
            M[i-1, j] = j**2*dados[i, j]


    soma = np.zeros(n[0]-1)
    soma2 = np.zeros(n[0]-1)
    for i in range(1, n[0]):
        for j in range(0, n[1]-1):
            soma[i-1] += (dados[i, j+1]+dados[i, j])*(dados[0, j+1]-dados[0, j])/2
            soma2[i-1] += (M[i-1, j+1]+M[i-1, j])*(dados[0, j+1]-dados[0, j])/2

    delta = 25
    max = int(n[1]/delta)
    L = np.zeros([n[0]-1, max])
    x = np.zeros([n[0]-1, max])
    W = np.zeros([n[0]-1, max])


    for k in range(1, n[0]):
        ii = 0
        for i in range(max):
            for j in range(ii, ii+delta):
                L[k-1, i] += dados[k, j]/soma[k-1]
                W[k-1, i] += M[k-1, j]/soma2[k-1]
                x[k-1, i] += dados[0, j]/delta
            ii += delta
    return x, W

x1, W1 = dist_MWD(dados2)
x2, W2 = dist_MWD(dados3)
x3, W3 = dist_MWD(dados4)
x4, W4 = dist_MWD(dados6)

plt.figure(1, figsize=(12.8, 9.6))
plt.subplot(221)
plt.plot(x1[0,:], W1[0,:], 'ko', ms=3)
plt.plot(x1[1,:], W1[1,:], 'rs', ms=3)
plt.plot(x1[2,:], W1[2,:], 'bv', ms=3)
plt.ylim(bottom=0, top=1e-2)
plt.xlim(left=0, right=12000)
plt.ylabel('$W_{log(n_i)}$', FontSize=14)
plt.xlabel('$n_i$', FontSize=14)
plt.text(8000, 0.008, '$V=5x10^{-17}$', FontSize=14)
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

plt.subplot(222)
plt.plot(x2[0,:], W2[0,:], 'ko', ms=3)
plt.plot(x2[1,:], W2[1,:], 'rs', ms=3)
plt.plot(x2[2,:], W2[2,:], 'bv', ms=3)
plt.ylim(bottom=0, top=1e-2)
plt.xlim(left=0, right=12000)
plt.ylabel('$W_{log(n_i)}$', FontSize=14)
plt.xlabel('$n_i$', FontSize=14)
plt.text(8000, 0.008, '$V=1x10^{-16}$', FontSize=14)
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

plt.subplot(223)
plt.plot(x3[0,:], W3[0,:], 'ko', ms=3)
plt.plot(x3[1,:], W3[1,:], 'rs', ms=3)
plt.plot(x3[2,:], W3[2,:], 'bv', ms=3)
plt.ylim(bottom=0, top=1e-2)
plt.xlim(left=0, right=12000)
plt.ylabel('$W_{log(n_i)}$', FontSize=14)
plt.xlabel('$n_i$', FontSize=14)
plt.text(8000, 0.008, '$V=5x10^{-16}$', FontSize=14)
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

plt.subplot(224)
plt.plot(x4[0,:], W4[0,:], 'ko', ms=3)
plt.plot(x4[1,:], W4[1,:], 'rs', ms=3)
plt.plot(x4[2,:], W4[2,:], 'bv', ms=3)
plt.ylim(bottom=0, top=1e-2)
plt.xlim(left=0, right=12000)
plt.ylabel('$W_{log(n_i)}$', FontSize=14)
plt.xlabel('$n_i$', FontSize=14)
plt.text(8000, 0.008, '$V=1x10^{-15}$', FontSize=14)
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

plt.savefig('dist_Standard.eps', format='eps', dpi=1000)
plt.show()