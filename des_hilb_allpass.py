import numpy as np
import scipy.signal as sig

import matplotlib.pyplot as plt


def to_hilb(a):
    a = np.array(a)
    h = np.array([(-1j) ** (-i) for i in range(a.size)])
    return a * h


def tfmul(a, b):
    return np.flip(np.polymul(np.flip(a), np.flip(b)))


# taps of a two-branch, two-section, lowpass half-band filter
alphas = [6.27978583e-01, 3.56629192e-01, 1.07224663e-01, 8.74907424e-01]
alphas = np.array(alphas)

b0 = []
a0 = []
for i in range(0, alphas.size, 2):
    b0.append([alphas[i], 0, 1])
    a0.append([1, 0, alphas[i]])

b1 = []
a1 = []
for i in range(1, alphas.size, 2):
    b1.append([alphas[i], 0, 1])
    a1.append([1, 0, alphas[i]])


b0 = list(map(to_hilb, b0))
a0 = list(map(to_hilb, a0))

b1 = list(map(to_hilb, b1))
a1 = list(map(to_hilb, a1))

h0 = 1
bi = 1
ai = 1
for b, a in zip(b0, a0):
    w, h = sig.freqz(b, a, whole=True)
    h0 *= h
    bi = tfmul(bi, b)
    ai = tfmul(ai, a)

_, h1 = sig.freqz([0, 1], 1, whole=True)
bq = [0, 1]
aq = 1
for b, a in zip(b1, a1):
    w, h = sig.freqz(b, a, whole=True)
    h1 *= h
    bq = tfmul(bq, b)
    aq = tfmul(aq, a)

# h = h0 + 1j * h1
# plt.plot(w, np.unwrap(np.angle(h0)))
# plt.plot(w, np.unwrap(np.angle(h1)))
# plt.grid(True)
# plt.show()

# plt.plot(w, 20 * np.log10(np.abs((h) / 2)))
# plt.ylim(-80, 1)
# plt.ylabel("Magnitude (dB)")
# plt.xlabel("radian/sample")
# plt.grid(True)
# plt.show()

print(list(map(float, np.real(bi))))
print(list(map(float, np.real(bq))))

w, hi = sig.freqz(bi, ai, whole=True)
w, hq = sig.freqz(bq, aq, whole=True)
plt.plot(w, 20 * np.log10(np.abs((hi + 1j * hq) / 2)))
plt.ylim(-80, 1)
plt.ylabel("Magnitude (dB)")
plt.xlabel("radian/sample")
plt.grid(True)
plt.show()

plt.plot(w, np.unwrap(np.angle(hq) - np.angle(hi)))
plt.ylabel("Phase difference (radian)")
plt.xlabel("radian/sample")
plt.grid(True)
plt.show()
