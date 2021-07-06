import math
import random
import numpy as np
from fractions import Fraction
from qiskit import Aer, QuantumCircuit, execute
import matplotlib.pyplot as plt

def factorize4(N):
    trial = 0
    while (True):
        trial += 1
        print('trial =', trial)
        a = random.randint(2, N - 1)
        if a not in [2, 7, 8, 11, 13]:
            continue
        r, qc = findPeriodByQuantumCircuit(N, a)
        print('\ta =', a, 'r =', r)
        if (r % 2 != 0):
            continue
        gcd1 = math.gcd(N, a ** (r // 2) + 1)
        gcd2 = math.gcd(N, a ** (r // 2) - 1)
        print('\tgcd1 =', gcd1, 'gcd2 =', gcd2)
        if (gcd1 == 1 or gcd2 == 1):
            continue
        return gcd1, gcd2, qc

def findPeriodByQuantumCircuit(N, a):
    phase, qc = qpe_amod15(a)
    frac = Fraction(phase).limit_denominator(15)
    return frac.denominator, qc

def qpe_amod15(a):
    n_count = 3
    qc = QuantumCircuit(4 + n_count, n_count)
    for q in range(n_count):
        qc.h(q) 
    qc.x(3 + n_count) 
    for q in range(n_count): 
        qc.append(c_amod15(a, 2 ** q), [q] + [i + n_count for i in range(4)])
    qc.append(qft_dagger(n_count), range(n_count)) 
    qc.measure(range(n_count), range(n_count))
    backend = Aer.get_backend('qasm_simulator')
    result = execute(qc, backend, shots=1, memory=True).result()
    readings = result.get_memory()
    phase = int(readings[0], 2) / (2 ** n_count)
    return phase, qc

def c_amod15(a, power):
    if a not in [2, 7, 8, 11, 13]:
        raise ValueError("'a' must be in [2, 7, 8, 11, 13]")
    U = QuantumCircuit(4)
    for iteration in range(power):
        if a in [2, 13]:
            U.swap(0, 1)
            U.swap(1, 2)
            U.swap(2, 3)
        if a in [7, 8]:
            U.swap(2, 3)
            U.swap(1, 2)
            U.swap(0, 1)
        if a == 11:
            U.swap(1, 3)
            U.swap(0, 2)
        if a in [7, 11, 13]:
            for q in range(4):
                U.x(q)
    U = U.to_gate()
    U.name = " %i^%i mod 15" % (a, power)
    c_U = U.control()
    return c_U

def qft_dagger(n):
    qc = QuantumCircuit(n)
    for qubit in range(n // 2):
        qc.swap(qubit, n - qubit - 1)
    for j in range(n):
        for m in range(j):
            qc.cu1(-np.pi / float(2**(j-m)), m, j)
        qc.h(j)
    qc.name = " QFT† (I-QFT)"
    return qc

def main():
    N = 3 * 5
    p, q, qc = factorize4(N)
    print(N, '=', p, '*', q)
    qc.draw('mpl') # 'mpl' added
    plt.show() # added
        
main()

# QPE 회로 그리기
a = 7
phase, qc = qpe_amod15(a)
qc.draw('mpl') # added
plt.show() # added

# Modular Power 회로 그리기
a = 2
power = 4
U = QuantumCircuit(4)
for iteration in range(power):
    if a in [2, 13]:
        U.swap(0, 1)
        U.swap(1, 2)
        U.swap(2, 3)
    if a in [7, 8]:
        U.swap(2, 3)
        U.swap(1, 2)
        U.swap(0, 1)
    if a == 11:
        U.swap(1, 3)
        U.swap(0, 2)
    if a in [7, 11, 13]:
        for q in range(4):
            U.x(q)
U.draw('mpl')
plt.show() # added


# I-QFT 회로 그리기

qc = qft_dagger(3)
qc.draw('mpl') # 'mpl' added
plt.show() # added

