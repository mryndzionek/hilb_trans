import numpy as np
import scipy.signal as sig

import matplotlib.pyplot as plt


def to_hilb(a):
    a = np.array(a)
    h = np.array([(-1j) ** (-i) for i in range(a.size)])
    return a * h


# taps of a two-branch, two-section, lowpass half-band filter
alphas = [6.27978583e-01, 3.56629192e-01, 1.07224663e-01, 8.74907424e-01]
alphas = np.array(alphas)

b00 = [alphas[0], 0, 1]
a00 = [1, 0, alphas[0]]

b01 = [alphas[2], 0, 1]
a01 = [1, 0, alphas[2]]

b10 = [alphas[1], 0, 1]
a10 = [1, 0, alphas[1]]

b11 = [alphas[3], 0, 1]
a11 = [1, 0, alphas[3]]

w, h00 = sig.freqz(b00, a00)
w, h01 = sig.freqz(b01, a01)

w, h10 = sig.freqz(b10, a10)
w, h11 = sig.freqz(b11, a11)

w, hd = sig.freqz([0, 1], 1)
h = h00 * h01 + h10 * h11 * hd

# plt.plot(w, np.unwrap(np.angle(h)))
# plt.show()
# plt.plot(w, 20 * np.log10(np.abs((h) / 2)))
# plt.show()

b00 = to_hilb(b00)
a00 = to_hilb(a00)
b01 = to_hilb(b01)
a01 = to_hilb(a01)
b10 = to_hilb(b10)
a10 = to_hilb(a10)
b11 = to_hilb(b11)
a11 = to_hilb(a11)

w, h00 = sig.freqz(b00, a00, whole=True)
w, h01 = sig.freqz(b01, a01, whole=True)

w, h10 = sig.freqz(b10, a10, whole=True)
w, h11 = sig.freqz(b11, a11, whole=True)

w, hd = sig.freqz([0, 1], 1, whole=True)

h = h00 * h01 + 1j * h10 * h11 * hd

# plt.plot(w, np.unwrap(np.angle(h)))
# plt.show()
# plt.plot(w, 20 * np.log10(np.abs((h) / 2)))
# plt.ylim(-60, 1)
# plt.show()

bi = np.polymul(np.flip(b00), np.flip(b01))
ai = np.polymul(np.flip(a00), np.flip(a01))
ai = np.real(np.flip(ai))
bi = np.real(np.flip(bi))

bq = np.polymul(np.flip(b10), np.flip(b11))
bq = np.polymul(bq, [0] * (bq.size - 2) + [1, 0])
aq = np.polymul(np.flip(a10), np.flip(a11))

aq = np.real(np.flip(aq))
bq = np.real(np.flip(bq))

print(list(map(float, bi)), list(map(float, ai)))
print(list(map(float, bq)), list(map(float, aq)))

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
