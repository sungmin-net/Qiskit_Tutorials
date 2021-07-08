 # https://qiskit.org/documentation/tutorials/circuits/3_summary_of_quantum_operations.html

import matplotlib.pyplot as plt
import numpy as np
from math import pi

from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, execute
from qiskit.tools.visualization import circuit_drawer
from qiskit.quantum_info import state_fidelity
from qiskit import BasicAer

from qiskit.visualization import plot_bloch_multivector # added

# Single Qubit Quantum states :  |φ> = α|0> + β|1>
backend = BasicAer.get_backend('unitary_simulator')

# Single Qubit Gates -----------------------------------------------
q = QuantumRegister(1)
qc = QuantumCircuit(q)
plot_bloch_multivector(qc)
plt.show()

# U(unitary) gates
q = QuantumRegister(1)
qc = QuantumCircuit(q)
qc.u3(pi / 2, pi / 2, pi / 2, q) # deprecation warning

qc.draw('mpl') # added 'mpl'
plt.show() # added

plot_bloch_multivector(qc)
plt.show()

job = execute(qc, backend)
print(job.result().get_unitary(qc, decimals = 3)) # print() added


qc = QuantumCircuit(q)

qc.u2(pi / 2, pi / 2, q)
qc.draw('mpl') # 'mpl' added
plt.show() # added

plot_bloch_multivector(qc)
plt.show()

job = execute(qc, backend)
print(job.result().get_unitary(qc, decimals = 3)) # print() added

qc = QuantumCircuit(q)
qc.u1(pi/2, q)
qc.draw('mpl') # 'mpl' added
plt.show() # added

plot_bloch_multivector(qc)
plt.show()

job = execute(qc, backend)
print(job.result().get_unitary(qc, decimals = 3)) # print() added

# I(identity) gate
qc = QuantumCircuit(q)
qc.id(q)
qc.draw('mpl') # 'mpl' added 
plt.show() # added

plot_bloch_multivector(qc)
plt.show()

job = execute(qc, backend)
print(job.result().get_unitary(qc, decimals = 3))


# Pauli Gates --------------
# X : bit-flip

qc = QuantumCircuit(q)
qc.x(q)
qc.draw('mpl') # 'mpl' added 
plt.show() # added

plot_bloch_multivector(qc)
plt.show()

job = execute(qc, backend)
print(job.result().get_unitary(qc, decimals = 3)) # print() added

# Y : bit and phase flip
qc = QuantumCircuit(q)
qc.y(q)
qc.draw('mpl') # 'mpl' added
plt.show() # added

plot_bloch_multivector(qc)
plt.show()

job = execute(qc, backend)
print(job.result().get_unitary(qc, decimals = 3)) # print() added


# Z : phaze-flip
qc = QuantumCircuit(q)
qc.z(q)
qc.draw('mpl') # 'mpl' added
plt.show() # added

plot_bloch_multivector(qc)
plt.show()

job = execute(qc, backend)
print(job.result().get_unitary(qc, decimals = 3)) # print() added

# Clifford Gates --------------
# Hadamard gate
qc = QuantumCircuit(q)
qc.h(q)
qc.draw('mpl') # 'mpl' added
plt.show() # added

plot_bloch_multivector(qc) # added
plt.show() # added

job = execute(qc, backend)
print(job.result(). get_unitary(qc, decimals = 3)) # print() added

# S (or sqrt(Z) phase) gate
qc = QuantumCircuit(q)
qc.s(q)
qc.draw('mpl') # 'mpl' added
plt.show()

plot_bloch_multivector(qc) # added
plt.show() # added

job = execute(qc, backend)
print(job.result().get_unitary(qc, decimals = 3)) # print() added


# S† (or conjugate of √Z phase) gate
qc = QuantumCircuit(q)
qc.sdg(q)
qc.draw('mpl')
plt.show()

plot_bloch_multivector(qc) # added
plt.show() # added

job = execute(qc, backend)
print(job.result().get_unitary(qc, decimals = 3)) # print() added


# C3 gates --------------
# T(or sqrt(S) phase) gate
qc = QuantumCircuit(q)
qc.t(q)
qc.draw('mpl') # 'mpl' added
plt.show()

plot_bloch_multivector(qc) # added
plt.show() # added

job = execute(qc, backend)
print(job.result().get_unitary(qc, decimals = 3)) # print() added

# T† (Or, conjugate of or sqrt(S) phase) gate
qc = QuantumCircuit(q)
qc.tdg(q)
qc.draw('mpl')
plt.show()

plot_bloch_multivector(qc) # added
plt.show() # added

job = execute(qc, backend)
job.result().get_unitary(qc, decimals = 3)
print(job.result().get_unitary(qc, decimals = 3)) # print() added


# Standard Rotations --------------
# Rotation araound X-axis
qc = QuantumCircuit(q)
qc.rx(pi / 2, q)
qc.draw('mpl') # 'mpl' added
plt.show() # added

plot_bloch_multivector(qc) # added
plt.show() # added

job = execute(qc, backend)
print(job.result().get_unitary(qc, decimals = 3)) # print() added


# Rotation around Y-axis
qc = QuantumCircuit(q)
qc.ry(pi / 2, q)
qc.draw('mpl') # 'mpl' added
plt.show() # added

plot_bloch_multivector(qc) # added
plt.show() # added

job = execute(qc, backend)
print(job.result().get_unitary(qc, decimals = 3)) # print() added


# Rotation around Z-axis
qc = QuantumCircuit(q)
qc.rz(pi / 2, q)
qc.draw('mpl') # 'mpl' added
plt.show() # added

plot_bloch_multivector(qc) # added
plt.show() # added

job = execute(qc, backend)
print(job.result().get_unitary(qc, decimals = 3)) # print() added


# Multi-Qubit Gates -----------------------------------------------
# Two-qubit gates------------------------
q = QuantumRegister(2) 

# Controlled Pauli Gates------
# Controlled-X (Or, controlled-NOT) gate 
qc = QuantumCircuit(q)
#qc.x(q[0])
qc.cx(q[0], q[1])
qc.draw('mpl') # 'mpl' added
plt.show() # added

plot_bloch_multivector(qc) # added
plt.show() # added

job = execute(qc, backend)
print('Controlled-X\n', job.result().get_unitary(qc, decimals = 3)) # print('Controlled-X\n',) added

# Controlled-Y gate
qc = QuantumCircuit(q)
#qc.x(q[0])
qc.cy(q[0], q[1])
qc.draw('mpl') # 'mpl' added
plt.show() # added

plot_bloch_multivector(qc) # added
plt.show() # added

job = execute(qc, backend)
print('Controlled-Y\n', job.result().get_unitary(qc, decimals = 3)) # print('Controlled-Y\n',) added

# Controlled-Z (Or, controlled phase) gate
qc = QuantumCircuit(q)
#qc.x(q[0])
qc.cz(q[0], q[1])
qc.draw('mpl') # 'mpl' added
plt.show() # added

plot_bloch_multivector(qc) # added
plt.show() # added

job = execute(qc, backend)
print('Controlled-Z\n', job.result().get_unitary(qc, decimals = 3)) # print('Controlled-Z\n',) added


# Controlled-Hadamard gate
qc = QuantumCircuit(q)
#qc.x(q[0])
qc.ch(q[0], q[1])
qc.draw('mpl') # 'mpl' added
plt.show() # added

plot_bloch_multivector(qc) # added
plt.show() # added

job = execute(qc, backend)
print('Controlled-Hadamard\n', job.result().get_unitary(qc, decimals = 3)) # print('~~~',) added

# Controlled-rotation gates ------
# Controlled rotation around Z-axis
qc = QuantumCircuit(q)
#qc.x(q[0])
qc.crz(pi / 2, q[0], [1])
qc.draw('mpl') # 'mpl' added
plt.show() # added

plot_bloch_multivector(qc) # added
plt.show() # added

job = execute(qc, backend)
print('Controlled-Rotation of Z axis\n', job.result().get_unitary(qc, decimals = 3)) # print('~~~',) added


# Controlled phase rotation
qc = QuantumCircuit(q)
#qc.x(q[0])
qc.cu1(pi / 2, q[0], q[1]) # deprecated

qc.draw('mpl') # 'mpl' added
plt.show() # added

plot_bloch_multivector(qc) # added
plt.show() # added

job = execute(qc, backend)
print('Controlled phase rotation\n', job.result().get_unitary(qc, decimals = 3)) # print('~~~',) added


# Controlled u3 rotation
qc = QuantumCircuit(q)
#qc.x(q[0])
qc.cu3(pi / 2, pi / 2, pi / 2, q[0], q[1]) # deprecated
qc.draw('mpl') # 'mpl' added
plt.show() # added

plot_bloch_multivector(qc) # added
plt.show() # added

job = execute(qc, backend)
print('Controlled u3 rotation\n', job.result().get_unitary(qc, decimals = 3)) # print('~~',) added


# Swap gate
qc = QuantumCircuit(q)
#qc.x(q[0])
qc.swap(q[0], q[1])
qc.draw('mpl') # 'mpl' added
plt.show() # added

plot_bloch_multivector(qc) # added
plt.show() # added

job = execute(qc, backend)
print('Swap\n', job.result().get_unitary(qc, decimals = 3)) # print('~~',) added


# Three-qubit gates------------------------
q = QuantumRegister(3)

# Toffoli gate(ccx gate): flips the third qubit if the first two 
# qubits (LSB) are both |1>.
qc = QuantumCircuit(q)
#qc.x(q[0])
#qc.x(q[1])     
qc.ccx(q[0], q[1], q[2])
qc.draw('mpl') # 'mpl' added
plt.show() # added

plot_bloch_multivector(qc) # added
plt.show() # added

job = execute(qc, backend)
print('Toffoli(CCX)\n', job.result().get_unitary(qc, decimals = 3)) # print('~~',) added


# Fredkin(Controlled swap) gate: exchanges the second and third 
# qubits if the first qubit (LSB) is |1>
qc = QuantumCircuit(q)
#qc.x(q[0])
#qc.x(q[1])
qc.cswap(q[0], q[1], q[2])
qc.draw('mpl') # 'mpl' added
plt.show() # added

plot_bloch_multivector(qc) # added
plt.show() # added

job = execute(qc, backend)
print('Fredkin(CSWAP)\n', job.result().get_unitary(qc, decimals = 3)) # print('~~',) added

# Non-unitary operations
q = QuantumRegister(1)
c = ClassicalRegister(1)
backend = BasicAer.get_backend('qasm_simulator')

# Measurements
qc = QuantumCircuit(q, c)
qc.measure(q, c)
qc.draw('mpl') # 'mpl' added
plt.show() # added

job = execute(qc, backend, shots = 1024)
print(job.result().get_counts(qc)) # print() added

qc = QuantumCircuit(q, c)
qc.h(q) # 1-qubit
qc.measure(q, c)
qc.draw('mpl') # 'mpl' added
plt.show() # added

job = execute(qc, backend, shots=1024)
print(job.result().get_counts(qc)) # print() added


# Reset
qc = QuantumCircuit(q, c)
qc.reset(q[0])
qc.measure(q, c)
qc.draw('mpl') # 'mpl' added
plt.show() # added

job = execute(qc, backend, shots = 1024)
print(job.result().get_counts(qc)) # print() added, {'0': 1024}

qc = QuantumCircuit(q, c)
qc.h(q)
qc.reset(q[0])
qc.measure(q, c)
qc.draw('mpl') # 'mpl' added
plt.show() # added

job = execute(qc, backend, shots = 1024)
print(job.result().get_counts(qc)) # print() added, {'0': 1024}

# Conditional operations
# operations can be conditioned on the state of the classical register

qc = QuantumCircuit(q, c)
qc.x(q[0]).c_if(c, 0)
qc.measure(q, c)
qc.draw('mpl') # 'mpl' added
plt.show() # added

job = execute(qc, backend, shots = 1024)
print(job.result().get_counts(qc)) # print() added, {'1': 1024}

qc = QuantumCircuit(q, c)
qc.h(q)
qc.measure(q, c)
qc.x(q[0]).c_if(c, 0) # if q is 1, this line does not flip q[0]
qc.measure(q, c)
qc.draw('mpl') # 'mpl' added
plt.show() # added

job = execute(qc, backend, shots = 1024)
print(job.result().get_counts(qc)) # print() added


# Arbitrary initialization
# initializaing a three-qubit state
import math
desired_vector = [
    1 / 4 * complex(0, 1), # changed from 1 / math.sqrt(16) * complex(0, 1),
    1 / math.sqrt(8), # chagned from 1 / math.sqrt(8) * complex(1, 0)  
    1 / 4 * complex(1, 1), # changed from 1 / math.sqrt(16) * complex(1, 1),
    0,
    0,
    1 / math.sqrt(8) * complex(1, 2),
    1 / 4, # changed from 1/ math.sqrt(16) * complex(1, 0),
    0
]

q = QuantumRegister(3)
qc = QuantumCircuit(q)
qc.initialize(desired_vector, [ q[0], q[1], q[2] ])
qc.draw('mpl') # 'mpl' added
plt.show() # added

backend = BasicAer.get_backend('statevector_simulator')
job = execute(qc, backend)
qc_state = job.result().get_statevector(qc)
print(qc_state) # print() added

# The fidelity is equal to 1 if and only if two states are equal.
print(state_fidelity(desired_vector, qc_state))

'''
his code is a part of Qiskit
© Copyright IBM 2017, 2021.

This code is licensed under the Apache License, Version 2.0. You may
obtain a copy of this license in the LICENSE.txt file in the root directory
of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.

Any modifications or derivative works of this code must retain this
copyright notice, and modified files need to carry a notice indicating
that they have been altered from the originals.
'''