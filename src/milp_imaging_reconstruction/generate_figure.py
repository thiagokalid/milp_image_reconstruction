import matplotlib.pyplot as plt
import numpy as np

A = np.zeros((4,5))

A[2 ,2] = .15
A[1 ,3] = .15
A[2 ,3] = .70

plt.figure(figsize=(3,2))
plt.imshow(A, extent=[.5, 5.5, .5, 4.5], aspect='equal', vmin=0, vmax=1)
plt.yticks([1, 2, 3, 4])
plt.xticks([1, 2, 3, 4, 5])
plt.colorbar()
plt.show()