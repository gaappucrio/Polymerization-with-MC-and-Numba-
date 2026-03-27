'''
Programa de distribuição de Monte Carlo para polimerização
'''
#################################################################################################
#               Importação de bibliotecas                                                       #
#################################################################################################

import time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
import lib_MC
from numba import jit

#################################################################################################
#               Constantes                                                                      #
#################################################################################################

MW_1, MW_IB = 104.15, 260
rho1 = 0.906
Rcte, Temp, N = 1.987, 273.15+100, 6.022e23
V = 4.e-18
start = time.time()
A1, A2, A3 = 2.57 - 5.05e-3 * Temp, 9.56 - \
    1.76e-2 * Temp, -3.03 + 7.85e-3 * Temp
#################################################################################################
#               Definição das constantes cinéticas                                              #
#################################################################################################

# Iniciação térmica
# L²/(mol².min¹)
kth = 1.314e7 * np.exp(-27440.5 / (Rcte * Temp))

# Propagação
# L/(mol.min)
kp = 6.128e8 * np.exp(-7067.8 / (Rcte * Temp))

# Iniciação
# L/(mol.min)
ki1 = ki2 = kp

# Transferência para monômero
# L/(mol.min)
kfm = 2.319e8 * np.exp(-12000 / (Rcte * Temp))

# Terminação por combinação
# L/(mol.min)
ktc0 = 7.53e10 * np.exp(-1680 / (Rcte * Temp))

# Decomposição
kd1 = 1.269e18 * np.exp(-35662.08 / (Rcte * Temp))                # 1/min
kd2 = 1.09e21 * np.exp(-42445 / (Rcte * Temp))                    # 1/min

#################################################################################################
#               Definição das constantes cinéticas de Monte Carlo                               #
#################################################################################################

# Iniciação térmica
kthMC = 6 * kth / (V * N) ** 2

# Propagação
kpMC = kp / (V * N)

# Iniciação
ki1MC = ki1 / (V * N)
ki2MC = ki2 / (V * N)

# Transferência para monômero
kfmMC = kfm / (V * N)

# Terminação por combinação
ktc0MC = 2 * ktc0 / (V * N)

# Decomposição
kd1MC = kd1
kd2MC = kd2

#################################################################################################
#               Condições iniciais                                                              #
#################################################################################################

n = 9
X = np.zeros(n, dtype='int64')
X[0] = 0.01 * (N * V)                           # Iniciador
X[1] = 0.e0                                     # Radical primário
X[2] = 0.e0                                     # Radical primário com um grupo peróxido não decomposto
X[3] = 8.0785 * (N * V)                         # Monômero
X[4] = 0.e0                                     # Radicais
X[5] = 0.e0                                     # Radicais com um grupo peróxido não decomposto
X[6] = 0.e0                                     # Polimero morto sem grupo peróxido
X[7] = 0.e0                                     # Polimero morto com um grupo peróxido
X[8] = 0.e0                                     # Polímero morto com dois grupos peróxido
f = 0.7                                         # Eficiência do iniciador

t, xi, xf, M0 = 0.0, 0.0, 0.9, X[3]

#################################################################################################
#               Taxas das reações                                                               #
#################################################################################################

n_reac = 13
Reac = np.zeros(n_reac, dtype=float)

# Cálculo para efeito gel
x = 1 - X[3] / M0
g = np.exp(-2 * (A1 * x + A2 * x ** 2 + A3 * x ** 3))
ktcMC = ktc0MC * g

# Propagação
Reac[0] = kpMC * X[3] * X[4]
Reac[1] = kpMC * X[3] * X[5]

# Decomposição do iniciador
Reac[2] = 2 * kd1MC * X[0]

# Iniciação química
Reac[3] = ki1MC * X[1] * X[3]
Reac[4] = ki2MC * X[2] * X[3]

# Terminação por combinação entre moléculas diferentes
Reac[5] = 1 / 2 * ktcMC * X[4] * X[5]

# Transferência para monômero
Reac[6] = kfmMC * X[3] * X[4]

# Terminação por combinação entre moléculas iguais
Reac[7] = 1 / 4 * ktcMC * X[4] * (X[4] - 1)

# Transferência para monômero
Reac[8] = kfmMC * X[3] * X[5]

# Segunda decomposição
Reac[9] = kd2MC * X[7]

# Terminação por combinação entre moléculas iguais
Reac[10] = 1 / 4 * ktcMC * X[5] * (X[5] - 1)

# Iniciação térmica
Reac[11] = kthMC * X[3] * (X[3] - 1) * (X[3] - 2) / 6

# Segunda decomposição
Reac[12] = 2 * kd2MC * X[8]

#################################################################################################
#               Monte Carlo                                                                     #
#################################################################################################
@jit(nopython=True)
def MC_Estireno(x, xf, t, X, f, Reac, M0, ktc0MC, kd1MC, kd2MC, kfmMC, ki1MC, ki2MC, kpMC, kthMC):
  P1 = [np.int64(x) for x in range(0)]
  P2 = [np.int64(x) for x in range(0)]
  L1 = [np.int64(x) for x in range(0)]
  L2 = [np.int64(x) for x in range(0)]
  L3 = [np.int64(x) for x in range(0)]
  lista_t = [np.float64(x) for x in range(0)]
  lista_x = [np.float64(x) for x in range(0)]
  while x < xf:


    R_soma, r1 = np.sum(Reac), np.random.rand()
    dt = np.log(1 / r1) / R_soma
    t += dt
    
    r2 = np.random.rand()
    R = r2 * R_soma

    if R <= Reac[0]:
      m = int(np.random.rand()*X[4])
      X[3] -= 1
      P1[m] += 1

      x = 1 - X[3] / M0
      Reac[0] = kpMC * X[3] * X[4]
      Reac[1] = kpMC * X[3] * X[5]
      Reac[3] = ki1MC * X[1] * X[3]
      Reac[4] = ki2MC * X[2] * X[3]
      Reac[6] = kfmMC * X[3] * X[4]
      Reac[8] = kfmMC * X[3] * X[5]
      Reac[11] = kthMC * X[3] * (X[3] - 1) * (X[3] - 2) / 6

    elif R <= np.sum(Reac[0:2]):
      m = int(np.random.rand()*X[5])
      X[3] -= 1
      P2[m] += 1

      x = 1 - X[3] / M0
      Reac[0] = kpMC * X[3] * X[4]
      Reac[1] = kpMC * X[3] * X[5]
      Reac[3] = ki1MC * X[1] * X[3]
      Reac[4] = ki2MC * X[2] * X[3]
      Reac[6] = kfmMC * X[3] * X[4]
      Reac[8] = kfmMC * X[3] * X[5]
      Reac[11] = kthMC * X[3] * (X[3] - 1) * (X[3] - 2) / 6

    elif R <= np.sum(Reac[0:3]):
      X[0] -= 1

      Reac[2] = 2 * kd1MC * X[0]
      r3 = np.random.rand()
      if r3 <= f:
        X[1] += 1
        X[2] += 1

        Reac[1] = kpMC * X[3] * X[5]
        Reac[3] = ki1MC * X[1] * X[3]
    elif R <= np.sum(Reac[0:4]):
      X[1] -= 1
      X[3] -= 1
      X[4] += 1
      P1.append(1)
      # Cálculo para efeito gel
      x = 1 - X[3] / M0
      g = np.exp(-2 * (A1 * x + A2 * x ** 2 + A3 * x ** 3))
      ktcMC = ktc0MC * g
      
      Reac[0] = kpMC * X[3] * X[4]
      Reac[1] = kpMC * X[3] * X[5]
      Reac[3] = ki1MC * X[1] * X[3]
      Reac[4] = ki2MC * X[2] * X[3]
      Reac[5] = 1 / 2 * ktcMC * X[4] * X[5]
      Reac[6] = kfmMC * X[3] * X[4]
      Reac[7] = 1 / 4 * ktcMC * X[4] * (X[4] - 1)
      Reac[8] = kfmMC * X[3] * X[5]
      Reac[11] = kthMC * X[3] * (X[3] - 1) * (X[3] - 2) / 6

    elif R <= np.sum(Reac[0:5]):
      X[2] -= 1
      X[3] -= 1
      X[5] += 1
      P2.append(1)

      # Cálculo para efeito gel
      x = 1 - X[3] / M0
      g = np.exp(-2 * (A1 * x + A2 * x ** 2 + A3 * x ** 3))
      ktcMC = ktc0MC * g
      
      Reac[0] = kpMC * X[3] * X[4]
      Reac[1] = kpMC * X[3] * X[5]
      Reac[3] = ki1MC * X[1] * X[3]
      Reac[4] = ki2MC * X[2] * X[3]
      Reac[5] = 1 / 2 * ktcMC * X[4] * X[5]
      Reac[6] = kfmMC * X[3] * X[4]
      Reac[8] = kfmMC * X[3] * X[5]
      Reac[10] = 1 / 4 * ktcMC * X[5] * (X[5] - 1)
      Reac[11] = kthMC * X[3] * (X[3] - 1) * (X[3] - 2) / 6

    elif R <= np.sum(Reac[0:6]):
      m, n = int(np.random.rand()*X[4]), int(np.random.rand()*X[5])
      X[4] -= 1
      X[5] -= 1
      X[7] += 1
      L2.append(P1[m] + P2[n])
      P1.pop(m)
      P2.pop(n)

      # Cálculo para efeito gel
      x = 1 - X[3] / M0
      g = np.exp(-2 * (A1 * x + A2 * x ** 2 + A3 * x ** 3))
      ktcMC = ktc0MC * g
      
      Reac[0] = kpMC * X[3] * X[4]
      Reac[1] = kpMC * X[3] * X[5]
      Reac[5] = 1 / 2 * ktcMC * X[4] * X[5]
      Reac[6] = kfmMC * X[3] * X[4]
      Reac[7] = 1 / 4 * ktcMC * X[4] * (X[4] - 1)
      Reac[8] = kfmMC * X[3] * X[5]
      Reac[9] = kd2MC * X[7]
      Reac[10] = 1 / 4 * ktcMC * X[5] * (X[5] - 1)

    elif R <= np.sum(Reac[0:7]):
      m = int(np.random.rand()*X[4])
      X[3] -= 1
      X[6] += 1
      L1.append(P1[m])
      P1[m] = 1

      x = 1 - X[3] / M0
      Reac[0] = kpMC * X[3] * X[4]
      Reac[1] = kpMC * X[3] * X[5]
      Reac[3] = ki1MC * X[1] * X[3]
      Reac[4] = ki2MC * X[2] * X[3]
      Reac[6] = kfmMC * X[3] * X[4]
      Reac[8] = kfmMC * X[3] * X[5]
      Reac[11] = kthMC * X[3] * (X[3] - 1) * (X[3] - 2) / 6

    elif R <= np.sum(Reac[0:8]):
      m, n = int(np.random.rand()*X[4]), int(np.random.rand()*X[4])
      if m == n:
        if m == 0:
          n += 1
        else:
          n -= 1
      X[4] -= 2
      X[6] += 1
      L1.append(P1[m] + P1[n])

      if m > n:
        ind = [m, n]
      else:
        ind = [n, m]

      for index in ind:
        P1.pop(index)
      # Cálculo para efeito gel
      x = 1 - X[3] / M0
      g = np.exp(-2 * (A1 * x + A2 * x ** 2 + A3 * x ** 3))
      ktcMC = ktc0MC * g
      

      Reac[0] = kpMC * X[3] * X[4]
      Reac[5] = 1 / 2 * ktcMC * X[4] * X[5]
      Reac[6] = kfmMC * X[3] * X[4]
      Reac[7] = 1 / 4 * ktcMC * X[4] * (X[4] - 1)

    elif R <= np.sum(Reac[0:9]):
      m = int(np.random.rand()*X[5])
      X[3] -= 1
      X[7] += 1
      L2.append(P2[m])
      X[5] -= 1
      X[4] += 1

      P1.append(1)
      P2.pop(m)
      # Cálculo para efeito gel
      x = 1 - X[3] / M0
      g = np.exp(-2 * (A1 * x + A2 * x ** 2 + A3 * x ** 3))
      ktcMC = ktc0MC * g
      
      Reac[0] = kpMC * X[3] * X[4]
      Reac[1] = kpMC * X[3] * X[5]
      Reac[3] = ki1MC * X[1] * X[3]
      Reac[4] = ki2MC * X[2] * X[3]
      Reac[5] = 1 / 2 * ktcMC * X[4] * X[5]
      Reac[6] = kfmMC * X[3] * X[4]
      Reac[7] = 1 / 4 * ktcMC * X[4] * (X[4] - 1)
      Reac[8] = kfmMC * X[3] * X[5]
      Reac[9] = kd2MC * X[7]
      Reac[10] = 1 / 4 * ktcMC * X[5] * (X[5] - 1)
      Reac[11] = kthMC * X[3] * (X[3] - 1) * (X[3] - 2) / 6

    elif R <= np.sum(Reac[0:10]):
      m = int(np.random.rand()*X[7])
      r4 = np.random.rand()
      X[7] -= 1
      X[1] += 1
      X[4] += 1

      P1.append(L2[m])
      L2.pop(m)

      # Cálculo para efeito gel
      x = 1 - X[3] / M0
      g = np.exp(-2 * (A1 * x + A2 * x ** 2 + A3 * x ** 3))
      ktcMC = ktc0MC * g
      
      Reac[0] = kpMC * X[3] * X[4]
      Reac[3] = ki1MC * X[1] * X[3]
      Reac[5] = 1 / 2 * ktcMC * X[4] * X[5]
      Reac[6] = kfmMC * X[3] * X[4]
      Reac[7] = 1 / 4 * ktcMC * X[4] * (X[4] - 1)
      Reac[9] = kd2MC * X[7]

      if r4 > f:
        L1.append(P1[X[4] - 1])
        P1.pop(X[4] - 1)
        X[1] -= 1
        X[4] -= 1
        X[6] += 1

        Reac[0] = kpMC * X[3] * X[4]
        Reac[3] = ki1MC * X[1] * X[3]
        Reac[4] = ki2MC * X[2] * X[3]
        Reac[5] = 1 / 2 * ktcMC * X[4] * X[5]
        Reac[6] = kfmMC * X[3] * X[4]
        Reac[7] = 1 / 4 * ktcMC * X[4] * (X[4] - 1)

    elif R <= np.sum(Reac[0:11]):
      m, n = int(np.random.rand()*X[5]), int(np.random.rand()*X[5])
      if m == n:
        if m == 0:
          n += 1
        else:
          n -= 1
      X[5] -= 2
      X[8] += 1

      if m > n:
        ind = [m, n]
      else:
        ind = [n, m]
      L3.append(P2[m] + P2[n])
      for index in ind:
        P2.pop(index) 

      # Cálculo para efeito gel
      x = 1 - X[3] / M0
      g = np.exp(-2 * (A1 * x + A2 * x ** 2 + A3 * x ** 3))
      ktcMC = ktc0MC * g
      
      Reac[1] = kpMC * X[3] * X[5]
      Reac[5] = 1 / 2 * ktcMC * X[4] * X[5]
      Reac[8] = kfmMC * X[3] * X[5]
      Reac[10] = 1 / 4 * ktcMC * X[5] * (X[5] - 1)
      Reac[12] = 2 * kd2MC * X[8]

    elif R <= np.sum(Reac[0:12]):
      X[3] -= 3
      X[4] += 2
      P1.append(1)
      P1.append(1)

      # Cálculo para efeito gel
      x = 1 - X[3] / M0
      g = np.exp(-2 * (A1 * x + A2 * x ** 2 + A3 * x ** 3))
      ktcMC = ktc0MC * g
      
      Reac[0] = kpMC * X[3] * X[4]
      Reac[1] = kpMC * X[3] * X[5]
      Reac[3] = ki1MC * X[1] * X[3]
      Reac[4] = ki2MC * X[2] * X[3]
      Reac[5] = 1 / 2 * ktcMC * X[4] * X[5]
      Reac[6] = kfmMC * X[3] * X[4]
      Reac[7] = 1 / 4 * ktcMC * X[4] * (X[4] - 1)
      Reac[8] = kfmMC * X[3] * X[5]
      Reac[11] = kthMC * X[3] * (X[3] - 1) * (X[3] - 2) / 6

    else:
      m = int(np.random.rand()*X[8])
      r5 = np.random.rand()
      X[8] -= 1
      X[1] += 1
      X[5] += 1
      P2.append(L3[m])
      L3.pop(m)

      # Cálculo para efeito gel
      x = 1 - X[3] / M0
      g = np.exp(-2 * (A1 * x + A2 * x ** 2 + A3 * x ** 3))
      ktcMC = ktc0MC * g
      
      Reac[1] = kpMC * X[3] * X[5]
      Reac[3] = ki1MC * X[1] * X[3]
      Reac[5] = 1 / 2 * ktcMC * X[4] * X[5]
      Reac[8] = kfmMC * X[3] * X[5]
      Reac[10] = 1 / 4 * ktcMC * X[5] * (X[5] - 1)
      Reac[12] = 2 * kd2MC * X[8]

      lista_t.append(t)
      lista_x.append(x)
      if r5 > f:
        X[7] += 1
        L2.append(P2[X[5] - 1])
        P2.pop(X[5] - 1)
        X[1] -= 1
        X[5] -= 1


        Reac[1] = kpMC * X[3] * X[5]
        Reac[3] = ki1MC * X[1] * X[3]
        Reac[5] = 1 / 2 * ktcMC * X[4] * X[5]
        Reac[8] = kfmMC * X[3] * X[5]
        Reac[9] = kd2MC * X[7]
        Reac[10] = 1 / 4 * ktcMC * X[5] * (X[5] - 1)

  return L1, L2, L3, lista_t, lista_x

start = time.time()
L1, L2, L3, lista_t, lista_x = MC_Estireno(x, xf, t, X, f, Reac, M0, ktc0MC, kd1MC, kd2MC, kfmMC, ki1MC, ki2MC, kpMC, kthMC)
tempo_total = time.time() - start
print("Tempo total de simulação: %f" % (tempo_total))

L1, L2, L3 = lib_MC.counting_sort(L1, int(max(L1))), lib_MC.counting_sort(
    L2, int(max(L2))), lib_MC.counting_sort(L3, int(max(L3)))

abc, max_L = [], max(len(L1), len(L2), len(L3))

for i in range(max_L):
    abc.append(i)

for i in range(len(L1), max_L):
  L1.append(0)
for i in range(len(L2), max_L):
  L2.append(0)
for i in range(len(L3), max_L):
  L3.append(0)
abc = np.array(abc)

L_soma = []
for i in range(len(L1)):
  L_soma.append(L1[i] + L2[i] + L3[i])

file1 = open('results-Numba.txt', 'w')
file1.write(
    '###########################################################################################\n')
file1.write('log\n')
file1.write(
    '###########################################################################################\n\n')  
file1.write('Time = %.2f\nVolume = %.2e\n\n' %
            (tempo_total, V))
file1.write(
    '###########################################################################################\n')
file1.write('conversão\n')
file1.write(
    '###########################################################################################\n\n')  
for i in range(len(lista_t)):
  file1.write('%.6f;%.6f\n' % (lista_t[i], lista_x[i]))
file1.write(
    '###########################################################################################\n')
file1.write('distribuições\n')
file1.write(
    '###########################################################################################\n\n')  
for i in range(len(L1)-1):
    file1.write('%i;%i;%i;%i;%i\n' % (abc[i], L1[i], L2[i], L3[i], L_soma[i]))
file1.close()

plt.figure(1)
plt.plot(abc, L1, 'ko', ms=3)

plt.figure(2)
plt.plot(abc, L2, 'rs', ms=3)

plt.figure(3)
plt.plot(abc, L3, 'b*', ms=3)

plt.figure(4)
plt.plot(abc, L_soma, 'ko', ms=3)

plt.figure(5)
plt.plot(lista_t, lista_x)
plt.show()