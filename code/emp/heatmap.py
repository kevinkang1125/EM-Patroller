import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

# vegetables = ["cucumber", "tomato", "lettuce", "asparagus",
#               "potato", "wheat", "barley"]
# farmers = ["Farmer Joe", "Upland Bros.", "Smith Gardening",
#            "Agrifun", "Organiculture", "BioGoods Ltd.", "Cornylee Corp."]
test = np.loadtxt("2000epoch.txt")
test = np.reshape(test,(10,7))
# test_ = np.zeros((10,7))
# for m in range(7):
#     for n in range(10):
#         test_[m,n] = test[n,m]
# print(test_)
# harvest = np.array([[8.891325686998019506e-03, 2.4, 2.5, 3.9, 0.0, 4.0, 0.0],
#                     [6.633811807999157040e-03, 0.0, 4.0, 1.0, 2.7, 0.0, 0.0],
#                     [2.038472098084387341e-03, 2.4, 0.8, 4.3, 1.9, 4.4, 0.0],
#                     [2.357954969034792292e-03, 0.0, 0.3, 0.0, 3.1, 0.0, 0.0],
#                     [2.288156909780125758e-03, 1.7, 0.6, 2.6, 2.2, 6.2, 0.0],
#                     [1.3, 1.2, 0.0, 0.0, 0.0, 3.2, 5.1],
#                     [0.1, 2.0, 0.0, 1.4, 0.0, 1.9, 6.3],
#                     [0.8, 2.4, 2.5, 3.9, 0.0, 4.0, 0.0],
#                     [0.1, 2.0, 0.0, 1.4, 0.0, 1.9, 6.3],
#                     [0.8, 2.4, 2.5, 3.9, 0.0, 4.0, 0.0]
# ])

# plt.xticks(np.arange(len(farmers)), labels=farmers, 
#                      rotation=45, rotation_mode="anchor", ha="right")
# plt.yticks(np.arange(len(vegetables)), labels=vegetables)
plt.figure(figsize=(8, 8))


plt.xticks([])  # 去掉x轴
plt.yticks([])
plt.axis('off')    
#plt.title("Harvest of local farmers (in tons/year)")
for i in range(10):
    for j in range(7):
    
        text = plt.text(j, i, (7*i+j+1), ha="center", va="center", color="k")
norm = (0,0.2)
map_vir = cm.get_cmap('RdYlBu_r')
#sm = cm.ScalarMappable(cmap=map_vir,norm=norm)

plt.imshow(test,cmap = map_vir)
plt.colorbar()
plt.clim(0,0.1)
plt.tight_layout()

plt.savefig("./2000.png")
plt.show()