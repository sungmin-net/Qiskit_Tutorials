# https://qiskit.org/documentation/tutorials/circuits_advanced/02_operators_overview.html

import numpy as np
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit import execute, BasicAer
from qiskit.compiler import transpile
from qiskit.quantum_info.operators import Operator, Pauli
from qiskit.quantum_info import process_fidelity
from qiskit.extensions import RXGate, XGate, CXGate

import matplotlib.pyplot as plt # added

# Creating Operators
XX = Operator([[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]])
print(XX) # print() added

# Operator Properties
print(XX.data) # print() added

input_dim, output_dim = XX.dim
print(input_dim, output_dim)

# Input and Output Dimensions
op = Operator(np.random.rand(2 ** 1, 2 ** 2))
print('Input dimensions : ', op.input_dims())
print('Output dimensions : ', op.output_dims())

# [6]
op = Operator(np.random.rand(6, 6))
print('Input dimensions: ', op.input_dims())
print('Output dimensions: ', op.output_dims())

# [7]
# Force input dimension to be (4, ) rather than (2, 2)
op = Operator(np.random.rand(2 ** 1, 2 ** 2), input_dims = [4])
print('Input dimensions: ', op.input_dims())
print('Output dimensions: ', op.output_dims())

# [8]
# Specify system is a qubit and qutrit(?)
op = Operator(np.random.rand(6, 6), input_dims = [2, 3], output_dims = [2, 3])
print('Input dimensions: ', op.input_dims())
print('Output dimensions: ', op.output_dims())

print('Dimension of input system 0:', op.input_dims([0]))
print('Dimension of input system 1:', op.input_dims([1]))

# Converting classes to Operators
# Create an Operator from a Pauli object
pauliXX = Pauli(label = 'XX')
print(Operator(pauliXX)) # print() added

# Create an Operator for a Gate object
print(Operator(CXGate())) # print() added

# Create an operator from a parameterized Gate object
print(Operator(RXGate(np.pi / 2))) # print() added

# Create an operator from a QuantumCircuit object
circ = QuantumCircuit(10)
circ.h(0)
for j in range(1, 10):
    circ.cx(j - 1, j)

# Convert circuit to an operator by implicit unitary simulation
print(Operator(circ)) # print() added

# Using Operators in circuits
# Create an operator
XX = Operator(Pauli(label = 'XX'))

# Add to a circuit
circ = QuantumCircuit(2, 2)
circ.append(XX, [0, 1])
circ.measure([0, 1], [0, 1])
circ.draw('mpl')
#plt.show() # added

backend = BasicAer.get_backend('qasm_simulator')
job = execute(circ, backend, basis_gates = ['u1', 'u2', 'u3', 'cx'])
job.result().get_counts(0)

# add to a circuit
circ2 = QuantumCircuit(2, 2)
circ2.append(Pauli(label = 'XX'), [0, 1])
circ2.measure([0, 1], [0, 1])
circ2.draw('mpl') # 'mpl' added
#plt.show() # added

# Combining Operators
# Tensor Product : A.tensor(B) = A⊗B
A = Operator(Pauli(label = 'X'))
B = Operator(Pauli(label = 'Z'))
print(A.tensor(B)) # print() added

# Tensor Expansion : A.expand(B) = B⊗A
A = Operator(Pauli(label = 'X'))
B = Operator(Pauli(label = 'Z'))
print(A.expand(B)) # print() added

# Composition
A = Operator(Pauli(label = 'X'))
B = Operator(Pauli(label = 'Z'))
print(A.compose(B)) # print() added; # matrix = B.A


A = Operator(Pauli(label = 'X'))
B = Operator(Pauli(label = 'Z'))
print(A.compose(B), front = True) # print() added; # compose: A.compose(B, front=True) = A.B


