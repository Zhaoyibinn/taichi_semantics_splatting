import matplotlib.pyplot as plt
import numpy as np

i = np.array(range(-1000,1000))
i = i/100
y = []
for onei in i:
    y.append(float(max(onei+0.5,0)))

# for onei in i:
#     y.append(float(int(onei >= (-0.5))))

plt.plot(i,y)
plt.show()
