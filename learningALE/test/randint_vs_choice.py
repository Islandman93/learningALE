import numpy as np
import time


st = time.time()
for x in range(1000):
    np.random.randint(x+1, size=32)
et = time.time()
print('randint {0}'.format(et-st))

st = time.time()
for x in range(1000):
    np.random.choice(x+1, size=32)
et = time.time()
print('choice {0}'.format(et-st))

st = time.time()
for x in range(1000):
    np.random.permutation(x+1)[0:32]
et = time.time()
print('perm {0}'.format(et-st))