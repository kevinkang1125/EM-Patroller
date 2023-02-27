import math
import numpy as np
robot_1=[6724, 12300, 7697, 21871, 11766, 6614, 12268, 9410, 11350]
robot_2=[12053, 11065, 5007, 17053, 7144, 11886, 13792, 14532, 7468]
entropy=[]
for i in range(9):
    entro = 1-(1-robot_1[i]/100000)*(1-robot_2[i]/100000)
    entropy.append(entro)
print(entropy)
entropy_max = 0
for m in range(9):
    entropy_max += -entropy[m]*math.log(entropy[m],math.e)
    print(entropy_max)
    
print(entropy_max)
c = np.savetxt("500epoch.txt",entropy,delimiter=',')