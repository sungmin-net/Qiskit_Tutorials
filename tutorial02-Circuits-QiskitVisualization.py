# https://qiskit.org/documentation/tutorials/circuits/2_plotting_data_in_qiskit.html

# Qiskit visualizations

from qiskit import *
from qiskit.visualization import plot_histogram
from qiskit.tools.monitor import job_monitor
import matplotlib.pyplot as plt # added

# Plot histogram
# quantum circuit to make a Bell state
bell = QuantumCircuit(2, 2)
bell.h(0)
bell.cx(0, 1)

meas = QuantumCircuit(2, 2)
meas.measure([0,1], [0,1])

# execute the quantum circuit
backend = BasicAer.get_backend('qasm_simulator') # the device to run on
circ = bell + meas
result = execute(circ, backend, shots=1000).result()
counts  = result.get_counts(circ)
print(counts)

plot_histogram(counts)
#plt.show() # added

# Execute 2-qubit Bell state again
second_result = execute(circ, backend, shots = 1000).result()
second_counts = second_result.get_counts(circ)

# Plot results with Legend
legend = ['First execution', 'Second execution']
plot_histogram([counts, second_counts], legend = legend)
#plt.show() # added

plot_histogram([counts, second_counts], legend=legend, sort='desc', figsize=(15,12),
        color=['orange', 'black'], bar_labels=False)
#plt.show() #added

# Plot State
from qiskit.visualization import plot_state_city, plot_bloch_multivector
from qiskit.visualization import plot_state_paulivec, plot_state_hinton
from qiskit.visualization import plot_state_qsphere

# execute the quantum circuit
backend = BasicAer.get_backend('statevector_simulator') # the device to run on
result = execute(bell, backend).result()
psi  = result.get_statevector(bell)

plot_state_city(psi)
#plt.show() # added

plot_state_hinton(psi)
#plt.show() # added

plot_state_qsphere(psi)
#plt.show() # added

plot_state_paulivec(psi)
#plt.show()

plot_bloch_multivector(psi)
#plt.show()

# Options when using state plotting functions

plot_state_city(psi, title="My City", color=['black', 'orange'])
#plt.show()

plot_state_hinton(psi, title="My Hinton")
#plt.show()

plot_state_paulivec(psi, title="My Paulivec", color=['purple', 'orange', 'green'])
#plt.show()

plot_bloch_multivector(psi, title="My Bloch Spheres")
#plt.show()

# Plot Bloch Vector
from qiskit.visualization import plot_bloch_vector
plot_bloch_vector([0,1,0])
#plt.show() # added

# Options for plot_bloch_vector()
plot_bloch_vector([0,1,0], title='My Bloch Sphere')
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
