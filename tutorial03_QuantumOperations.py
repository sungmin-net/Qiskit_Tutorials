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
qc.u3(pi / 2, pi / 2, pi / 2, q)

plot_bloch_multivector(qc)
plt.show()


'''
The QuantumCircuit.u3 method is deprecated as of 0.16.0. It will be removed no earlier than 3 months
after the release date. You should use QuantumCircuit.u instead, which acts identically. 
Alternatively, you can decompose u3 in terms of QuantumCircuit.p and QuantumCircuit.sx: 
u3(ϴ,φ,λ) = p(φ+π) sx p(ϴ+π) sx p(λ) (2 pulses on hardware).
  qc.u3(pi / 2, pi / 2, pi / 2, q)
'''
qc.draw('mpl') # added 'mpl'
#plt.show() # added
  
job = execute(qc, backend)
print(job.result().get_unitary(qc, decimals = 3)) # print() added

qc = QuantumCircuit(q)
qc.draw('mpl')
plt.show()

qc.u2(pi / 2, pi / 2, q)
'''
The QuantumCircuit.u2 method is deprecated as of 0.16.0. It will be removed no earlier than 3 months
after the release date. You can use the general 1-qubit gate QuantumCircuit.u instead: 
u2(φ,λ) = u(π/2, φ, λ). Alternatively, you can decompose it interms of QuantumCircuit.p and
QuantumCircuit.sx: u2(φ,λ) = p(π/2+φ) sx p(π/2+λ) (1 pulse on hardware).
'''
qc.draw('mpl') # 'mpl' added
#plt.show() # added

job = execute(qc, backend)
print(job.result().get_unitary(qc, decimals = 3)) # print() added 

qc = QuantumCircuit(q)
qc.u1(pi/2, q)
'''
The QuantumCircuit.u1 method is deprecated as of 0.16.0. It will be removed no earlier than 3 months
after the release date. You should use the QuantumCircuit.p method instead, which acts identically.
qc.u1(pi/2, q)
'''
qc.draw('mpl') # 'mpl' added
#plt.show() # added
 
# I(identity) gate
qc = QuantumCircuit(q)
qc.id(q)
qc.draw('mpl') # 'mpl' added 
#plt.show() # added

job = execute(qc, backend)
print(job.result().get_unitary(qc, decimals = 3))

# Pauli Gates --------------
# X : bit-flip

qc = QuantumCircuit(q)
qc.x(q)
qc.draw('mpl') # 'mpl' added 
#plt.show() # added

job = execute(qc, backend)
print(job.result().get_unitary(qc, decimals = 3)) # print() added

# Y : bit and phase flip
qc = QuantumCircuit(q)
qc.y(q)
qc.draw('mpl') # 'mpl' added
#plt.show() # added

job = execute(qc, backend)
print(job.result().get_unitary(qc, decimals = 3)) # print() added

# Z : phaze-flip
qc = QuantumCircuit(q)
qc.z(q)
qc.draw('mpl') # 'mpl' added
#plt.show() # added

job = execute(qc, backend)
job.result().get_unitary(qc, decimals = 3)

# Clifford Gates --------------
# Hadamard gate
qc = QuantumCircuit(q)
qc.h(q)
qc.draw('mpl') # 'mpl' added
#plt.show() # added

job = execute(qc, backend)
print(job.result(). get_unitary(qc, decimals = 3)) # print() added

# S (또는 √Z phase) gate
qc = QuantumCircuit(q)
qc.s(q)
qc.draw('mpl') # 'mpl' added
#plt.show()

job = execute(qc, backend)
print(job.result().get_unitary(qc, decimals = 3)) # print() added

# S† (또는 conjugate of √Z phase) 게이트
qc = QuantumCircuit(q)
qc.sdg(q)
qc.draw('mpl')
#plt.show()

# C3 gates --------------
# T(또는 √S phase) gate
qc = QuantumCircuit(q)
qc.t(q)
qc.draw('mpl') # 'mpl' added
#plt.show()

job = execute(qc, backend)
print(job.result().get_unitary(qc, decimals = 3)) # print() added

# T† (Or, conjugate of √S phase) gate
qc = QuantumCircuit(q)
qc.tdg(q)
qc.draw('mpl')
#plt.show()

job = execute(qc, backend)
job.result().get_unitary(qc, decimals = 3)

# Standard Rotations --------------
# Rotation araound X-axis
qc = QuantumCircuit(q)
qc.rx(pi / 2, q)
qc.draw('mpl') # 'mpl' added
#plt.show() # added

job = execute(qc, backend)
print(job.result().get_unitary(qc, decimals = 3)) # print() added

# Rotation around Y-axis
qc = QuantumCircuit(q)
qc.ry(pi / 2, q)
qc.draw('mpl') # 'mpl' added
#plt.show() # added

job = execute(qc, backend)
print(job.result().get_unitary(qc, decimals = 3)) # print() added

# Rotation around Z-axis
qc = QuantumCircuit(q)
qc.rz(pi / 2, q)
qc.draw('mpl') # 'mpl' added
#plt.show() # added

job = execute(qc, backend)
print(job.result().get_unitary(qc, decimals = 3)) # print() added

# Multi-Qubit Gates -----------------------------------------------
# Two-qubit gates------------------------
q = QuantumRegister(2) 

# Controlled Pauli Gates------
# Controlled-X (Or, controlled-NOT) gate 
qc = QuantumCircuit(q)
qc.cx(q[0], q[1])
qc.draw('mpl') # 'mpl' added
#plt.show() # added

job = execute(qc, backend)
print('Controlled-X\n', job.result().get_unitary(qc, decimals = 3)) # print('Controlled-X\n',) added

# Controlled-Y gate
qc = QuantumCircuit(q)
qc.cy(q[0], q[1])
qc.draw('mpl') # 'mpl' added
#plt.show() # added

job = execute(qc, backend)
print('Controlled-Y\n', job.result().get_unitary(qc, decimals = 3)) # print('Controlled-Y\n',) added

# Controlled-Z (Or, controlled phase) gate
qc = QuantumCircuit(q)
qc.cz(q[0], q[1])
qc.draw('mpl') # 'mpl' added
#plt.show() # added

job = execute(qc, backend)
print('Controlled-Z\n', job.result().get_unitary(qc, decimals = 3)) # print('Controlled-Z\n',) added

# Controlled-Hadamard gate
qc = QuantumCircuit(q)
qc.ch(q[0], q[1])
qc.draw('mpl') # 'mpl' added
#plt.show() # added

job = execute(qc, backend)
print('Controlled-Hadamard\n', job.result().get_unitary(qc, decimals = 3)) # print('~~~',) added

# Controlled-rotation gates ------
# Controlled rotation around Z-axis
qc = QuantumCircuit(q)
qc.crz(pi / 2, q[0], [1])
qc.draw('mpl') # 'mpl' added
#plt.show() # added

job = execute(qc, backend)
print('Controlled-Rotation of Z axis\n', job.result().get_unitary(qc, decimals = 3)) # print('~~~',) added

# Controlled phase rotation
qc = QuantumCircuit(q)
qc.cu1(pi / 2, q[0], q[1]) # deprecated
'''
The QuantumCircuit.cu1 method is deprecated as of 0.16.0. It will be removed no earlier than 
3 months after the release date. You should use the QuantumCircuit.cp method instead, 
which acts identically.
qc.cu1(pi / 2, q[0], q[1])
'''
qc.draw('mpl') # 'mpl' added
#plt.show() # added

job = execute(qc, backend)
print('Controlled phase rotation\n', job.result().get_unitary(qc, decimals = 3)) # print('~~~',) added

# Controlled u3 rotation
qc = QuantumCircuit(q)
qc.cu3(pi / 2, pi / 2, pi / 2, q[0], q[1]) # deprecated
qc.draw('mpl') # 'mpl' added
#plt.show() # added
'''
The QuantumCircuit.cu3 method is deprecated as of 0.16.0. It will be removed no earlier than
3 months after the release date. You should use the QuantumCircuit.cu method instead, where
cu3(ϴ,φ,λ) = cu(ϴ,φ,λ,0).
'''

job = execute(qc, backend)
print('Controlled u3 rotation\n', job.result().get_unitary(qc, decimals = 3)) # print('~~',) added

# Swap gate
qc = QuantumCircuit(q)
qc.swap(q[0], q[1])
qc.draw('mpl') # 'mpl' added
#plt.show() # added

job = execute(qc, backend)
print('Swap\n', job.result().get_unitary(qc, decimals = 3)) # print('~~',) added

# Three-qubit gates------------------------
q = QuantumRegister(3)

# Toffi gate(ccx gate) : flips the third qubit if the first two qubits (LSB) are both |1>.
qc = QuantumCircuit(q)
qc.ccx(q[0], q[1], q[2])
qc.draw('mpl') # 'mpl' added
#plt.show() # added

job = execute(qc, backend)
print('Toffoli(CCX)\n', job.result().get_unitary(qc, decimals = 3)) # print('~~',) added

# Fredkin(Controlled swap) gate : exchanges the second and third qubits if the first qubit (LSB) is |1>:
qc = QuantumCircuit(q)
qc.cswap(q[0], q[1], q[2])
qc.draw('mpl') # 'mpl' added
#plt.show() # added

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
#plt.show() # added

job = execute(qc, backend, shots = 1024)
print(job.result().get_counts(qc)) # print() added

qc = QuantumCircuit(q, c)
qc.h(q) # 1-qubit
qc.measure(q, c)
qc.draw('mpl') # 'mpl' added
#plt.show() # added

job = execute(qc, backend, shots=1024)
print(job.result().get_counts(qc)) # print() added

# Reset
qc = QuantumCircuit(q, c)
qc.reset(q[0])
qc.measure(q, c)
qc.draw('mpl') # 'mpl' added
#plt.show() # added

job = execute(qc, backend, shots = 1024)
print(job.result().get_counts(qc)) # print() added, {'0': 1024}

qc = QuantumCircuit(q, c)
qc.h(q)
qc.reset(q[0])
qc.measure(q, c)
qc.draw('mpl') # 'mpl' added
#plt.show() # added

job = execute(qc, backend, shots = 1024)
print(job.result().get_counts(qc)) # print() added, {'0': 1024}

# Conditional operations
# It is also possible to do operations conditioned on the state of the classical register

qc = QuantumCircuit(q, c)
qc.x(q[0]).c_if(c, 0)
qc.measure(q, c)
qc.draw('mpl') # 'mpl' added
#plt.show() # added

job = execute(qc, backend, shots = 1024)
print(job.result().get_counts(qc)) # print() added, {'1': 1024}

qc = QuantumCircuit(q, c)
qc.h(q)
qc.measure(q, c) # 만약 qrk 0으로 측정되면, 다음 라인에서 1로 바뀜(측정 후에도 게이트 연산 가능).
qc.x(q[0]).c_if(c, 0) # q가 1로 측정되면, 다음라인에서 안건드림
qc.measure(q, c)
qc.draw('mpl') # 'mpl' added
#plt.show() # added

job = execute(qc, backend, shots = 1024)
print(job.result().get_counts(qc)) # print() added, 그래서 결론은 항상 |1>. {'1': 1024}

# Arbitrary initialization
# initializaing a three-qubit state
import math
desired_vector = [
    1 / math.sqrt(16) * complex(0, 1),
    1 / math.sqrt(8) * complex(1, 0),
    1 / math.sqrt(16) * complex(1, 1),
    0,
    0,
    1 / math.sqrt(8) * complex(1, 2),
    1 / math.sqrt(16) * complex(1, 0),
    0
] # 8개 = 2^3 개

q = QuantumRegister(3)
qc = QuantumCircuit(q)
qc.initialize(desired_vector, [ q[0], q[1], q[2] ])
qc.draw('mpl') # 'mpl' added
#plt.show() # added

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