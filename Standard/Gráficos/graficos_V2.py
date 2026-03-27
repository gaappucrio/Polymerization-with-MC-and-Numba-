import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO

dados1 = np.loadtxt('1_1e-17.txt', dtype='int64', delimiter='\t', unpack=True)
dados2 = np.loadtxt('2_5e-17.txt', dtype='int64', delimiter='\t', unpack=True)
dados3 = np.loadtxt('3_1e-16.txt', dtype='int64', delimiter='\t', unpack=True)
dados4 = np.loadtxt('4_5e-16.txt', dtype='int64', delimiter='\t', unpack=True)
dados5 = np.loadtxt('5_7.5e-16.txt', dtype='int64', delimiter='\t', unpack=True)
dados6 = np.loadtxt('6_1e-15.txt', dtype='int64', delimiter='\t', unpack=True)
dados7 = np.loadtxt('7_2.5e-15.txt', dtype='int64', delimiter='\t', unpack=True)
dados8 = np.loadtxt('8_5e-15.txt', dtype='int64', delimiter='\t', unpack=True)
dados9 = np.loadtxt('9_7.5e-15.txt', dtype='int64', delimiter='\t', unpack=True)

def dist_MWD(dados):
    n = np.shape(dados)
    #M = np.zeros([n[0]-1, n[1]], dtype=np.float64)
    M = []
    shape = (n[0]-1, n[1])
    for i in range(1, n[0]):
        for j in range(0, n[1]):
            M.append(j**2*dados[i, j])

    M = np.array(M)
    M = M.reshape(shape)
    
    dados [0, :] = dados [0, :] * 104.15
    soma = []
    soma2 = []
    for i in range(1, n[0]):
        a, b = 0, 0
        for j in range(0, n[1]-1):
            a += (dados[i, j+1]+dados[i, j])*(dados[0, j+1]-dados[0, j])/2
            b += (M[i-1, j+1]+M[i-1, j])*(dados[0, j+1]-dados[0, j])/2
        soma.append(a)
        soma2.append(b)
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

x1, W1 = dist_MWD(dados1)
x2, W2 = dist_MWD(dados2)
x3, W3 = dist_MWD(dados3)
x4, W4 = dist_MWD(dados4)
x5, W5 = dist_MWD(dados5)
x6, W6 = dist_MWD(dados6)
x7, W7 = dist_MWD(dados7)
x8, W8 = dist_MWD(dados8)
x9, W9 = dist_MWD(dados9)

np.savetxt('dados1.txt', np.transpose(W1))
np.savetxt('dados2.txt', np.transpose(W2))
np.savetxt('dados3.txt', np.transpose(W3))
np.savetxt('dados4.txt', np.transpose(W4))
np.savetxt('dados5.txt', np.transpose(W5))
np.savetxt('dados6.txt', np.transpose(W6))
np.savetxt('dados7.txt', np.transpose(W7))
np.savetxt('dados8.txt', np.transpose(W8))
np.savetxt('dados9.txt', np.transpose(W9))


plt.figure(1, figsize=(12.8, 9.6))
plt.subplot(221)
plt.plot(np.log10(x3[0,:]), W3[0,:], 'ko', ms=3)
plt.plot(np.log10(x3[1,:]), W3[1,:], 'rs', ms=3)
plt.plot(np.log10(x3[2,:]), W3[2,:], 'bv', ms=3)
plt.ylim(bottom=0, top=1e-4)
plt.xlim(left=3, right=7)
plt.ylabel('$W_{log(MW)}$', FontSize=14)
plt.xlabel('$log(MW)$', FontSize=12)
plt.text(3.5, 0.00008, '$V=1x10^{-16}$ L', FontSize=14)
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

plt.subplot(222)
plt.plot(np.log10(x4[0,:]), W4[0,:], 'ko', ms=3)
plt.plot(np.log10(x4[1,:]), W4[1,:], 'rs', ms=3)
plt.plot(np.log10(x4[2,:]), W4[2,:], 'bv', ms=3)
plt.ylim(bottom=0, top=1e-4)
plt.xlim(left=3, right=7)
plt.ylabel('$W_{log(MW)}$', FontSize=14)
plt.xlabel('$log(MW)$', FontSize=12)
plt.text(3.5, 0.00008, '$V=5x10^{-16}$ L', FontSize=14)
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

plt.subplot(223)
plt.plot(np.log10(x6[0,:]), W6[0,:], 'ko', ms=3)
plt.plot(np.log10(x6[1,:]), W6[1,:], 'rs', ms=3)
plt.plot(np.log10(x6[2,:]), W6[2,:], 'bv', ms=3)
plt.ylim(bottom=0, top=1e-4)
plt.xlim(left=3, right=7)
plt.ylabel('$W_{log(MW)}$', FontSize=14)
plt.xlabel('$log(MW)$', FontSize=12)
plt.text(3.5, 0.00008, '$V=1x10^{-15}$ L', FontSize=14)
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

plt.subplot(224)
plt.plot(np.log10(x8[0,:]), W8[0,:], 'ko', ms=3)
plt.plot(np.log10(x8[1,:]), W8[1,:], 'rs', ms=3)
plt.plot(np.log10(x8[2,:]), W8[2,:], 'bv', ms=3)
plt.ylim(bottom=0, top=1e-4)
plt.xlim(left=3, right=7)
plt.ylabel('$W_{log(MW)}$', FontSize=14)
plt.xlabel('$log(MW)$', FontSize=12)
plt.text(3.5, 0.00008, '$V=5x10^{-15}$ L', FontSize=14)
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

plt.savefig('dist_standard.eps', format='eps', dpi=1000)
plt.show()