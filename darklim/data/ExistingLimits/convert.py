import numpy as np

m, x = np.loadtxt('Knapen_Scaled_2018.txt', unpack=True)

f = open('Knapen_Scaled_2018_2.txt', 'w')
for i in range(len(m)):
    f.write('%f %e\n' % (m[i] * 1e3, x[i]))
f.close()
