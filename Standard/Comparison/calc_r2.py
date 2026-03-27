'''
Programa para calcular o valor de R2
'''

import numpy as np
import statistics

def R2(exp,calc):

    n=len(exp)

    media_exp=np.mean(exp)              #Média dos valores experimentais
    media_calc=np.mean(calc)
    delta_exp=exp-media_exp       #Soma total dos quadrados
    delta_calc=calc-media_calc            #Soma dos quadrados dos resíduos

    desvio_exp=statistics.stdev(delta_exp)#np.std(delta_exp)
    desvio_calc=statistics.stdev(delta_calc)#np.std(delta_calc)
    
    num=np.matmul(delta_exp,delta_calc)/(n-1)
    dem=desvio_exp*desvio_calc
    
    print(num,dem)
    R2 = num/dem

    return R2

a=np.array([3,-0.5,2,7])
b=np.array([2.5,0,2,8])

r=R2(a,b)

print(r)