import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO

dados1 = np.loadtxt('1_1.txt', dtype='int64', delimiter='\t', unpack=True)
dados2 = np.loadtxt('2_2.txt', dtype='int64', delimiter='\t', unpack=True)
dados3 = np.loadtxt('3_10.txt', dtype='int64', delimiter='\t', unpack=True)
dados4 = np.loadtxt('4_15.txt', dtype='int64', delimiter='\t', unpack=True)
dados5 = np.loadtxt('5_20.txt', dtype='int64', delimiter='\t', unpack=True)
dados6 = np.loadtxt('6_50.txt', dtype='int64', delimiter='\t', unpack=True)
dados7 = np.loadtxt('7_100.txt', dtype='int64', delimiter='\t', unpack=True)
dados8 = np.loadtxt('8_150.txt', dtype='int64', delimiter='\t', unpack=True)


def dist_MWD(dados):
    n = np.shape(dados)
    #M = np.zeros([n[0]-1, n[1]], dtype=np.float64)
    M = []
    for i in range(1, n[0]):
        for j in range(0, n[1]):
            M.append(j**2*dados[i, j])

    M = np.array(M)
    M = M.reshape(n[0]-1, n[1])
    
    soma = []#np.zeros(n[0]-1)
    soma2 = []#np.zeros(n[0]-1)
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


x1, W1 = dist_MWD(dados2)
x2, W2 = dist_MWD(dados3)
x3, W3 = dist_MWD(dados5)
x4, W4 = dist_MWD(dados7)

np.savetxt('dados2.txt', np.transpose(W1))
np.savetxt('dados3.txt', np.transpose(W2))
np.savetxt('dados5.txt', np.transpose(W3))
np.savetxt('dados7.txt', np.transpose(W4))

plt.figure(1, figsize=(12.8, 9.6))
plt.subplot(221)
plt.plot(np.log10(x1[0,:]), W1[0,:], 'ko', ms=3)
plt.plot(np.log10(x1[1,:]), W1[1,:], 'rs', ms=3)
plt.plot(np.log10(x1[2,:]), W1[2,:], 'bv', ms=3)
plt.ylim(bottom=0, top=1e-2)
plt.xlim(left=1, right=5)
plt.ylabel('$W_{log(n_i)}$', FontSize=14)
plt.xlabel('$log(n_i)$', FontSize=12)
plt.text(1.5, 0.008, '$2\,Loops$', FontSize=14)
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

plt.subplot(222)
plt.plot(np.log10(x2[0,:]), W2[0,:], 'ko', ms=3)
plt.plot(np.log10(x2[1,:]), W2[1,:], 'rs', ms=3)
plt.plot(np.log10(x2[2,:]), W2[2,:], 'bv', ms=3)
plt.ylim(bottom=0, top=1e-2)
plt.xlim(left=1, right=5)
plt.ylabel('$W_{log(n_i)}$', FontSize=14)
plt.xlabel('$log(n_i)$', FontSize=12)
plt.text(1.5, 0.008, '$10\,Loops$', FontSize=14)
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

plt.subplot(223)
plt.plot(np.log10(x3[0,:]), W3[0,:], 'ko', ms=3)
plt.plot(np.log10(x3[1,:]), W3[1,:], 'rs', ms=3)
plt.plot(np.log10(x3[2,:]), W3[2,:], 'bv', ms=3)
plt.ylim(bottom=0, top=1e-2)
plt.xlim(left=1, right=5)
plt.ylabel('$W_{log(n_i)}$', FontSize=14)
plt.xlabel('$log(n_i)$', FontSize=12)
plt.text(1.5, 0.008, '$20\,Loops$', FontSize=14)
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

plt.subplot(224)
plt.plot(np.log10(x4[0,:]), W4[0,:], 'ko', ms=3)
plt.plot(np.log10(x4[1,:]), W4[1,:], 'rs', ms=3)
plt.plot(np.log10(x4[2,:]), W4[2,:], 'bv', ms=3)
plt.ylim(bottom=0, top=1e-2)
plt.xlim(left=1, right=5)
plt.ylabel('$W_{log(n_i)}$', FontSize=14)
plt.xlabel('$log(n_i)$', FontSize=12)
plt.text(1.5, 0.008, '$100\,Loops$', FontSize=14)
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

plt.savefig('dist_Numba-Loop.eps', format='eps', dpi=1000)
plt.show()