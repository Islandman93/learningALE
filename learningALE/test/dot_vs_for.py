import numpy as np
import time
a = np.random.random((10,128))

w = np.random.random((128, 4))

dst = time.time()
for x in range(1000):
    _ = np.dot(a, w)
det = time.time()

print(det-dst)

fst = time.time()
for x in range(1000):
    for node in range(4):
        for weight in range(128):
            _ = a[0,weight] * w[weight, node]
fet = time.time()
print(fet-fst)