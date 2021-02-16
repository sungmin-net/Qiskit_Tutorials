# https://qiskit.org/documentation/tutorials/circuits/1_getting_started_with_qiskit.html

import numpy as np
from qiskit import *
# %matplotlib inline # blocked I don't use Conda.
import matplotlib.pyplot as plt # added

# Circuit Basics
# building the circuit

# crate a quantum circuit acting on an quantum register of three qubits
circ = QuantumCircuit(3)

# Add a H gate on qubit 0, putting this qubit in superposition
circ.h(0)

# Add a CX(CNOT) gate on control qubit 0 and target qubit 1, putting the qubits in a Bell state
circ.cx(0, 1)

# Add a CX(CNOT) gate on control qubit 0 and target qubit 2, putting the qubits in a GHZ state
circ.cx(0, 2)

# visualize circuit
circ.draw('mpl')
plt.show() # added

# Simulating circuits using Qiskit Aer

# statevector backend

# Import Aer
from qiskit import Aer

# Run the quantum circuit on a statevector simulator backend
backend = Aer.get_backend('statevector_simulator')

# Create a Quantum Program for execution
job = execute(circ, backend)
result = job.result()

outputstate = result.get_statevector(circ, decimals=3)
print(outputstate)

from qiskit.visualization import plot_state_city
plot_state_city(outputstate)
plt.show() # added

# Unitary backend
# Run the quantum circuit on a unitary simulator backend
backend = Aer.get_backend('unitary_simulator')
job = execute(circ, backend)
result = job.result()

# Show the results
print(result.get_unitary(circ, decimals=3))

# OpenQASM backend
# Create a Quantum Circuit
meas = QuantumCircuit(3, 3)
meas.barrier(range(3))
# map the quantum measurement to the classical bits
meas.measure(range(3), range(3))

# The Qiskit circuit object supports composition using the addition operator.
qc = circ + meas

#drawing the circuit
qc.draw('mpl') # 'mpl' added
plt.show() # added

# Use Aer's qasm_simulator
backend_sim = Aer.get_backend('qasm_simulator')

# Execute the circuit on the qasm simulator. We've set the number of repeats of the circuit
# to be 1024, which is the default.
job_sim = execute(qc, backend_sim, shots=1024)

# Grab the results from the job.
result_sim = job_sim.result()

counts = result_sim.get_counts(qc)
print(counts) # {'000': 483, '111': 541}

from qiskit.visualization import plot_histogram
plot_histogram(counts)
plt.show() # added


'''
This code is a part of Qiskit
© Copyright IBM 2017, 2021.

This code is licensed under the Apache License, Version 2.0. You may
obtain a copy of this license in the LICENSE.txt file in the root directory
of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.

Any modifications or derivative works of this code must retain this
copyright notice, and modified files need to carry a notice indicating
that they have been altered from the originals.
'''