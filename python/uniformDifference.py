import random
import math

err = 0.0
for i in range(1000):
    x = random.uniform(0,1)
    y = random.uniform(0,1)
    err += math.fabs(x-y)

print err/float(1000)

