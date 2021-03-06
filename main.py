import numpy as np
import pandas as pd
from math import pi
import matplotlib.pyplot as plt

file = pd.read_excel('ELR.xlsx')
timp = file.iloc[:, 0]
acc = np.zeros((len(file.iloc[:, 1]), 1))
acc[:, 0] = file.iloc[:, 1]

dt = timp[1] - timp[0]
xi = 0.02 #fractiunea din amortizarea critica

T = np.zeros((300, 1))#perioada oscilatiilor
T[:, 0] = np.linspace(0.01, 3, 300)[:]
omega_n = 2*np.pi/T #frecventa unghiulara


def SD(xi, omega_n, dt, E):
    u0 = np.zeros((len(omega_n), 1))
    for j in range(len(omega_n)):
        wn = omega_n[j, 0]
        A = np.array([[0, 1], [-wn**2, -2*xi*wn]])
        (D, V) = np.linalg.eig(A)
        ep = np.array([[np.exp(D[0]*dt), 0], [0, np.exp(D[1]*dt)]])
        Ad = V.dot(ep).dot(np.linalg.inv(V))
        Bd = np.linalg.inv(A).dot(Ad - np.eye((len(A))))
        z = [[0], [0]]
        d = np.zeros((len(E), 1))
        v = np.zeros((len(E), 1))
        for i in range(len(E)):
            z = np.real(Ad).dot(z) + np.real(Bd).dot([[0], [-E[i]*9.81]])
            d[i] = z[0, 0]
            v[i] = z[1, 0]
        u0[j] = max(abs(d))
    return u0


u0_1 = SD(xi, omega_n, dt, acc)
v0_1 = omega_n*u0_1
a0_1 = omega_n*omega_n*u0_1

#plt.subplot(3, 1, 1)
plt.plot(T, u0_1, 'r', linewidth=2)
plt.title('Spectrul deplasărilor relative', fontsize=12, fontweight = 'bold')
plt.ylabel('u0', fontsize=10)
plt.xlabel('timp', fontsize=10)
plt.xlim(0, 3)
plt.tight_layout()
plt.show()

#plt.subplot(3, 1, 2)
plt.plot(T, v0_1, 'g', linewidth=2)
plt.title('Spectrul vitezelor relative', fontsize=12, fontweight = 'bold')
plt.ylabel('v0', fontsize=10)
plt.xlabel('timp', fontsize=10)
plt.xlim(0, 3)
plt.tight_layout()
plt.show()

#plt.subplot(3, 1, 3)
plt.plot(T, a0_1, 'y', linewidth=2)
plt.title('Spectrul acceleratiilor relative', fontsize=12, fontweight = 'bold')
plt.ylabel('a0', fontsize=10)
plt.xlabel('timp', fontsize=10)
plt.xlim(0, 3)
plt.tight_layout()
plt.show()

#plt.tight_layout()
#plt.show()

k = [2*197.392, 197.392]
m = [2*2500, 2500]
A = np.zeros((2, 2))

for i in range(2):
    if i != 1:
        A[i][i] = (k[i] + k[i + 1]) / m[i]
        A[i][i + 1] = k[i] / m[i]
    else:
        A[i][i] = k[i] / m[i]

    if i != 0:
        A[i][i - 1] = -k[i] / m[i]

    (d, v) = np.linalg.eig(A)
    wn = np.sqrt(d) / 2 / pi
    t = np.array(range(0, 3000, 2)).astype(float)
    t[:] /= 10

    aux1 = wn[:, np.newaxis]
    aux2 = t[np.newaxis, :]
    x = v @ np.sin(aux1 @ aux2)
    x[:] /= 10**8
    x = x.real

    amp = np.empty(len(x))
    for i in range(len(x)):
        amp[i] = max(abs(x[i]))


    print("Matricea sistemului:\n{}\n\nValori proprii:\n{}\nVectori proprii:\n{}\nFrecventele:\n{}\n\nMagnitudinea:\n{}\n\n\n". \
                    format(A, np.round(d, 2), np.round(v, 4), np.round(wn.real, 2), amp))

    (fig, ax) = plt.subplots(1)
    colors = ["lightcoral", "tomato", "saddlebrown", "bisque", "gold", "lawngreen", "deepskyblue", "rebeccapurple", "gray"]

    for i in range(len(x)):
        ax.plot(t, x[i], label = "Out{}".format(i + 1), color = colors[i])

    ax.set_xlabel("Timp", color = "white")
    ax.tick_params(axis = 'x', colors = "white")
    ax.set_ylabel("Amplitudine", color = "white")
    ax.tick_params(axis = 'y', colors = "white")

    leg0 = ax.legend(facecolor = "dimgray")
    for text in leg0.get_texts():
            text.set_color("white")

    fig.set_facecolor("0.15")
    ax.set_facecolor("k")

    fig.tight_layout()

plt.show()



