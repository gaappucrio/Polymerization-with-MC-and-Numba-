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

#################################################################################################
#               Constantes                                                                      #
#################################################################################################

MW_1, MW_IB = 104.15, 260
rho1 = 0.906
Rcte, Temp, N = 1.987, 273.15+100, 6.022e23
V = 1e-17
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

L1, L2, L3, P1, P2 = [], [], [], np.zeros(0, dtype=int), np.zeros(0, dtype=int)
lista_reac = []
t, xi, xf, M0 = 0.0, 0.0, 0.9, X[3]

#################################################################################################
#               Taxas das reações                                                               #
#################################################################################################

n_reac = 13
Reac = np.zeros(n_reac, dtype=float)

# Iniciação térmica
Reac[0] = kthMC * X[3] * (X[3] - 1) * (X[3] - 2) / 6

# Decomposição do iniciador
Reac[1] = 2 * kd1MC * X[0]

# Iniciação química
Reac[2] = ki1MC * X[1] * X[3]
Reac[3] = ki2MC * X[2] * X[3]

# Segunda decomposição
Reac[4] = kd2MC * X[7]
Reac[5] = 2 * kd2MC * X[8]

# Propagação
Reac[6] = kpMC * X[3] * X[4]
Reac[7] = kpMC * X[3] * X[5]

# Transferência para monômero
Reac[8] = kfmMC * X[3] * X[4]
Reac[9] = kfmMC * X[3] * X[5]

# Cálculo para efeito gel
x = 1 - X[3] / M0
g = np.exp(-2 * (A1 * x + A2 * x ** 2 + A3 * x ** 3))
ktcMC = ktc0MC * g

# Terminação por combinação entre moléculas iguais
Reac[10] = 1 / 4 * ktcMC * X[4] * (X[4] - 1)
Reac[11] = 1 / 4 * ktcMC * X[5] * (X[5] - 1)

# Terminação por combinação entre moléculas diferentes
Reac[12] = 1 / 2 * ktcMC * X[4] * X[5]

#################################################################################################
#               Monte Carlo                                                                     #
#################################################################################################


while x < xf:


    R_soma, r1 = sum(Reac), np.random.rand()
    dt = np.log(1 / r1) / R_soma

    r2 = np.random.rand()
    R = r2 * R_soma

    if R <= Reac[0]:
        X[3] -= 3 # monomero
        X[4] += 2 # radical sem grupo peroxido
        P1 = np.append(P1, [1, 1])
        lista_reac.append(1)
    elif R <= sum(Reac[0:2]):
        X[0] -= 1
        r3 = np.random.rand()
        lista_reac.append(2)
        if r3 <= f:
            X[1] += 1
            X[2] += 1
    elif R <= sum(Reac[0:3]):
        lista_reac.append(3)
        X[1] -= 1
        X[3] -= 1
        X[4] += 1
        P1 = np.append(P1, 1)
    elif R <= sum(Reac[0:4]):
        lista_reac.append(4)
        X[2] -= 1
        X[3] -= 1
        X[5] += 1
        P2 = np.append(P2, 1)
    elif R <= sum(Reac[0:5]):
        lista_reac.append(5)
        m = int(np.random.rand()*X[7])
        r4 = np.random.rand()
        X[7] -= 1
        X[1] += 1
        X[4] += 1
        P1 = np.append(P1, L2[m])
        L2.pop(m)
        if r4 > f:
            L1.append(P1[X[4] - 1])
            P1 = np.delete(P1, X[4] - 1)
            X[1] -= 1
            X[4] -= 1
            X[6] += 1
    elif R <= sum(Reac[0:6]):
        lista_reac.append(6)
        m = int(np.random.rand()*X[8])
        r5 = np.random.rand()
        X[8] -= 1
        X[1] += 1
        X[5] += 1
        P2 = np.append(P2, L3[m])
        L3.pop(m)
        if r5 > f:
            X[7] += 1
            L2.append(P2[X[5] - 1])
            P2 = np.delete(P2, X[5] - 1)
            X[1] -= 1
            X[5] -= 1
    elif R <= sum(Reac[0:7]):
        lista_reac.append(7)
        m = int(np.random.rand()*X[4])
        X[3] -= 1
        P1[m] += 1
    elif R <= sum(Reac[0:8]):
        lista_reac.append(8)
        m = int(np.random.rand()*X[5])
        X[3] -= 1
        P2[m] += 1
    elif R <= sum(Reac[0:9]):
        lista_reac.append(9)
        m = int(np.random.rand()*X[4])
        X[3] -= 1
        X[6] += 1
        L1.append(P1[m])
        P1[m] = 1
    elif R <= sum(Reac[0:10]):
        lista_reac.append(10)
        m = int(np.random.rand()*X[5])
        X[3] -= 1
        X[7] += 1
        L2.append(P2[m])
        X[5] -= 1
        X[4] += 1
        P1 = np.append(P1, 1)
        P2 = np.delete(P2, m)
    elif R <= sum(Reac[0:11]):
        lista_reac.append(11)
        m, n = int(np.random.rand()*X[4]), int(np.random.rand()*X[4])
        if m == n:
            if m == 0:
                n += 1
            else:
                n -= 1
        X[4] -= 2
        X[6] += 1
        L1.append(P1[m] + P1[n])
        P1 = np.delete(P1, [m, n])
    elif R <= sum(Reac[0:12]):
        lista_reac.append(12)
        m, n = int(np.random.rand()*X[5]), int(np.random.rand()*X[5])
        if m == n:
            if m == 0:
                n += 1
            else:
                n -= 1
        X[5] -= 2
        X[8] += 1
        L3.append(P2[m] + P2[n])
        P2 = np.delete(P2, [m, n])
    else:
        lista_reac.append(13)
        m, n = int(np.random.rand()*X[4]), int(np.random.rand()*X[5])
        X[4] -= 1
        X[5] -= 1
        X[7] += 1
        L2.append(P1[m] + P2[n])
        P1 = np.delete(P1, m)
        P2 = np.delete(P2, n)

    # Iniciação térmica
    Reac[0] = kthMC * X[3] * (X[3] - 1) * (X[3] - 2) / 6

    # Decomposição do iniciador
    Reac[1] = 2 * kd1MC * X[0]

    # Iniciação química
    Reac[2] = ki1MC * X[1] * X[3]
    Reac[3] = ki2MC * X[2] * X[3]

    # Segunda decomposição
    Reac[4] = kd2MC * X[7]
    Reac[5] = 2 * kd2MC * X[8]

    # Propagação
    Reac[6] = kpMC * X[3] * X[4]
    Reac[7] = kpMC * X[3] * X[5]

    # Transferência para monômero
    Reac[8] = kfmMC * X[3] * X[4]
    Reac[9] = kfmMC * X[3] * X[5]

    # Cálculo para efeito gel
    x = 1 - X[3] / M0
    g = np.exp(-2 * (A1 * x + A2 * x ** 2 + A3 * x ** 3))
    ktcMC = ktc0MC * g

    # Terminação por combinação entre moléculas iguais
    Reac[10] = 1 / 4 * ktcMC * X[4] * (X[4] - 1)
    Reac[11] = 1 / 4 * ktcMC * X[5] * (X[5] - 1)

    # Terminação por combinação entre moléculas diferentes
    Reac[12] = 1 / 2 * ktcMC * X[4] * X[5]

    t += dt


tempo_total = time.time() - start
print("Tempo total de simulação: %f" % (tempo_total))
print(lib_MC.counting_sort(lista_reac, int(max(lista_reac))))

L1, L2, L3 = lib_MC.counting_sort(L1, int(max(L1))), lib_MC.counting_sort(
    L2, int(max(L2))), lib_MC.counting_sort(L3, int(max(L3)))
abc1, abc2, abc3 = [], [], []
for i in range(len(L1)):
    abc1.append(i)

for i in range(len(L2)):
    abc2.append(i)

for i in range(len(L3)):
    abc3.append(i)

abc1, abc2, abc3 = np.array(abc1), np.array(abc2), np.array(abc3)
file1 = open('results.txt', 'w')
file1.write(
    '###########################################################################################\n\n')
file1.write('Time = %.2f\nVolume = %.2e\n\n' %
            (tempo_total, V))
file1.write(
    '###########################################################################################\n\n')

for i in range(len(L1)-1):
    file1.write('%i;%i\n' % (abc1[i], L1[i]))

file1.write(
    '\n###########################################################################################\n\n')
for i in range(len(L2)-1):
    file1.write('%i;%i\n' % (abc2[i], L2[i]))

file1.write(
    '\n###########################################################################################\n\n')
for i in range(len(L3)-1):
    file1.write('%i;%i\n' % (abc3[i], L3[i]))

file1.close()

print(t)

plt.figure(1)
plt.plot(abc1, L1, 'ko', ms=3)

plt.figure(2)
plt.plot(abc2, L2, 'rs', ms=3)

plt.figure(3)
plt.plot(abc3, L3, 'b*', ms=3)

plt.show()


'''
a = np.bincount(P) # formando o vetor a que ja conta quantas cadeias de certo tamanho existem (tira direto do histograma)
tamanho = np.linspace(0, len(a), len(a)) 



# Construcao do grafico
arquivo = 'CLD.tiff'

plt.figure(0)
plt.plot(tamanho,a, 'r*', color = 'red')
plt.ylabel("Quantidade de cadeias")
plt.xlabel("tamanho da cadeia")
#plt.savefig(caminho_final, format = "tiff", dpi = 100)
plt.savefig(arquivo, format = "tiff", dpi = 100)
axes = plt.gca()
axes.set_xlim([0.0,len(a)])
plt.show()
'''