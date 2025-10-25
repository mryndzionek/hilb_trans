import numpy as np
import scipy.signal as sig

from functools import partial
from scipy import optimize
import matplotlib.pyplot as plt
from matplotlib import markers


def H(x, z):
    if x.size % 2 == 1:
        raise ValueError("x needs to be even!")

    H0 = 1
    for i in range(0, x.size, 2):
        H0 *= (x[i] + (z ** (-2))) / (1 + x[i] * (z ** (-2)))

    H1 = z ** (-1)
    for i in range(1, x.size, 2):
        H1 *= (x[i] + (z ** (-2))) / (1 + x[i] * (z ** (-2)))

    return np.abs(H0 + H1) / 2


def H_mag(ws, ds, l1, x):
    res = []
    fun = H

    for w, d in zip(ws, ds):
        z = np.exp(1j * w)
        mag = fun(x, z)
        res.append(d - mag)

    if l1:
        return np.sum(np.abs(res))
    else:
        return np.sum(np.square(res))


def des_mag_response(w0, tw, w_step=0.01, As=180, restrict_transition=False):
    w = 0
    ws = []
    ds = []

    while w < np.pi:
        if w < (w0 - tw / 2):
            ds.append(1)
            ws.append(w)
            w += w_step
        elif w < (w0 + tw / 2):
            if restrict_transition:
                a = -(1 / tw)
                b = 1 - (w0 - (tw / 2)) * a
                d = a * w + b
                ds.append(d)
                ws.append(w)
            w += w_step
        else:
            d = 10 ** (-As / 20)
            ds.append(d)
            ws.append(w)
            w += w_step

    return ws, ds


def flatten(xss):
    return [x for xs in xss for x in xs]


def float_array_to_str(arr, line_len=4):
    lines = []
    if isinstance(arr[0], float):
        fmt = lambda x: "{:.8e}".format(x)
    else:
        fmt = lambda x: str(x)

    for i in range(0, len(arr), line_len):
        lines.append("    " + ", ".join(map(fmt, arr[i : i + line_len])) + ",")

    return "{\n" + "\n".join(lines) + "\n};"


def tfmul(a, b):
    return np.flip(np.polymul(np.flip(a), np.flip(b)))


def pad(a, b):
    if a.size > b.size:
        b = np.concat([b, np.zeros(a.size - b.size)])
    elif b.size > a.size:
        a = np.concat([a, np.zeros(b.size - a.size)])
    return a, b


SECTIONS = 2  # Number of sections in each branch
w0 = np.pi * (1.0 / 2.0)
tw = w0 / (2 * SECTIONS)
ws, ds = des_mag_response(w0, tw, 0.01, 60, restrict_transition=False)

res = partial(H_mag, ws, ds, True)
results = []
for i in range(10):
    x0 = np.random.uniform(0, 0.99, SECTIONS * 2)
    res_1 = optimize.minimize(res, x0, bounds=[(0, 0.99)] * (SECTIONS * 2))
    # print(f"{res_1.success}")
    # print(f"Cost: {res_1.fun}")
    if res_1.success:
        results.append((res_1.fun, res_1.x))

results = sorted(results, key=lambda x: x[0])
print(f"Lowest cost: {results[0][0]}")
alphas = list(map(float, results[0][1]))

print("Alphas:")
print(float_array_to_str(alphas))

alphas = np.array(alphas)
h0 = 1
a0 = 1
b0 = 1
for i in range(0, alphas.size, 2):
    b = [alphas[i], 0, 1]
    a = np.flip(b)
    w, h = sig.freqz(b, a)
    a0 = tfmul(a0, a)
    b0 = tfmul(b0, b)
    h0 *= h

a1 = 1
b1 = [0, 1]
w, h1 = sig.freqz(b1, a1)
for i in range(1, alphas.size, 2):
    b = [alphas[i], 0, 1]
    a = np.flip(b)
    w, h = sig.freqz(b, a)
    a1 = tfmul(a1, a)
    b1 = tfmul(b1, b)
    h1 *= h

ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=2)
ax2 = plt.subplot2grid((2, 2), (1, 0), colspan=1)
ax3 = plt.subplot2grid((2, 2), (1, 1), colspan=1)
axs = [ax1, ax2, ax3]

for ax in axs:
    ax.grid(True)

b0, b1 = pad(tfmul(b0, a1), tfmul(b1, a0))
b = b0 + b1
a = tfmul(a0, a1)

print("TF combined:")
print(f"b = {float_array_to_str(b)}")
print(f"a = {float_array_to_str(a)}")

axs[0].plot(ws, 20 * np.log10(ds))
axs[0].plot(w, 20 * np.log10(np.abs(h0 + h1) / 2))
axs[0].set_xlabel("radian/sample")
axs[0].set_ylabel("dB")
axs[0].set_title("Magnitude")

axs[1].plot(w, np.unwrap(np.angle(h0)))
axs[1].plot(w, np.unwrap(np.angle(h1)))
axs[1].set_xlabel("radian/sample")
axs[1].set_ylabel("radian")
axs[1].set_title("Phase")

zs, ps, _ = sig.tf2zpk(b, a)
c = plt.Circle((0, 0), 1, fill=False)
axs[2].add_patch(c)
axs[2].scatter(np.real(ps), np.imag(ps), marker="x")
axs[2].scatter(
    np.real(zs), np.imag(zs), marker=markers.MarkerStyle("o", fillstyle="none")
)
axs[2].set_title("Poles and zeros")
plt.show()
