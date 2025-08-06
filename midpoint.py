import numpy as np
import matplotlib.pyplot as plt

N = 3

r = np.arange(0,N)
points = np.exp(2.0*np.pi*1j*r/N)

res = 100
w = np.arange(0,res)
unit_circle = np.exp(2.0*np.pi*w*1j/res)

start = 0.1 + 0.5j

def compute_new_rand_location(startLoc):
    rand_location = np.random.randint(0, N)
    vector = (points[rand_location] - startLoc) / 2.0
    next_point = startLoc + vector
    return next_point, rand_location

iterations = 10000

plt.plot(np.real(unit_circle), np.imag(unit_circle), "b-")
plt.plot(np.real(points), np.imag(points), "r.")
plt.plot(np.real(start), np.imag(start), "y.")

next_point = start

for iteration in range(iterations):
    next_point, rand_location = compute_new_rand_location(next_point)
    plt.plot(np.real(next_point), np.imag(next_point), "b.")
    

plt.show()