"""
VQE implementation for H2 molecule using Qiskit.
Author: Denis
"""


import numpy as np
from pyscf import ao2mo, gto, mcscf, scf

#Setting up the molecule
distance = 0.735
a = distance / 2

mol = gto.Mole()
mol.build(
    verbose = 0,
    unit="Angstrom",
    atom=[
        ["H", (0, 0, -a)],
        ["H", (0, 0, a)],
    ],
    basis="sto-6g",
    spin=0,
    charge=0,
    symmetry="Dooh",
)

#Mean Field energy (HF)
mf = scf.RHF(mol)
mf.scf()
print (
    "Nuclear interaction energy:",
    mf.energy_nuc())
print ("Full energy from HF:",
    mf.energy_elec()[0])
print ("Full energy from HF:",
    mf.energy_tot())    


#CASCI energy for reference
active_space = range(mol.nelectron // 2 - 1, mol.nelectron // 2 + 1)
#from the set [0,2) we get indices of active orbitals (0,1)

E1 = mf.kernel()
 #calculating the Hartree-Fock energy (mf)

mx = mcscf.CASCI(mf, ncas=2, nelecas=(1,1))
 #Created a complete active space and set its parameters: ncas - amount of active orbitals;
#nelecass - spin distribution

mo = mx.sort_mo(active_space, base=0)
#Sort it so the CASCI gets correct indices of active orbitals

E2 = mx.kernel(mo)[:2]
#Calculating the CASCI (mx) energy using orbitals from mo
#E[0] - full energy; E[1] - electronic energy

print("Electron energy from CASCI", E2[1])
print("Full energy from CASCI: ", E2[0])


#Forming a canonical hamiltonian
from qiskit_nature.second_q.operators.tensor_ordering import to_physicist_ordering
from qiskit_nature.second_q.hamiltonians import ElectronicEnergy

h1e, ecore = mx.get_h1eff() #1-electron integrals matrice and nuclear energy
h2e = ao2mo.restore(1, mx.get_h2eff(), mx.ncas) #4-d tensor of 2-electron integrals

h2e_phys = to_physicist_ordering(h2e) #rearrange indices 
elec_eng = ElectronicEnergy.from_raw_integrals(h1e, h2e_phys, None, None, None)  #packing integrals
second_q_op = elec_eng.second_q_op() #canonical hamiltonian

#Mapping to qubit hamiltonian
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit.quantum_info import SparsePauliOp

mapper = JordanWignerMapper()
qubit_hamiltonian = mapper.map(second_q_op) #qubit hamiltonian without nuclear energy

nq = qubit_hamiltonian.num_qubits
identity = SparsePauliOp.from_list([('I'*nq, 1.0)]) #identity operator
qubit_hamiltonian = qubit_hamiltonian + ecore * identity #qubit hamiltonian with nuclear energy
#print(qubit_hamiltonian)

#Ansatz (1 parameter)
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.circuit import Parameter

theta = Parameter('theta')
qreg_q = QuantumRegister(4, 'q') #4 qubits
creg_c = ClassicalRegister(4, 'c') #4 classical bits
circuit=QuantumCircuit(qreg_q, creg_c)

for i in range(4):
    circuit.ry(theta, qreg_q[i]) # y и z rotations
    circuit.rz(theta, qreg_q[i])

circuit.cx(qreg_q[0], qreg_q[1]) #CNOT gates
circuit.cx(qreg_q[1], qreg_q[2])
circuit.cx(qreg_q[2], qreg_q[3])

diagram = circuit.draw('text')

#Classical optimizer (1 parameter)
from qiskit.primitives import StatevectorEstimator
from scipy.optimize import minimize

estimator = StatevectorEstimator()

def energy_scipy(params):
    theta_val = float(params[0])
    bound = circuit.assign_parameters([theta_val], inplace=False)
    pub = (bound, [qubit_hamiltonian])
    result = estimator.run([pub]).result()[0]
    return float(result.data.evs[0]) + ecore

res = minimize(energy_scipy, x0=np.array([3.0]), method='COBYLA', options={'maxiter': 200, 'disp': False})

print("VQE energy:", res.fun)
print("Theta:", res.x)




#Ansatz (4 parameters)
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.circuit import ParameterVector

thetas = ParameterVector('thetas', 4)
qreg_q = QuantumRegister(4, 'q') #4 qubits
creg_c = ClassicalRegister(4, 'c') #4 classical bits
circuit=QuantumCircuit(qreg_q, creg_c)

for i in range(4):
    circuit.ry(thetas[i], qreg_q[i]) #y и z rotations
    circuit.rz(thetas[i], qreg_q[i])

circuit.cx(qreg_q[0], qreg_q[1]) #CNOT gates
circuit.cx(qreg_q[1], qreg_q[2])
circuit.cx(qreg_q[2], qreg_q[3])

#diagram = circuit.draw('text')
#print(diagram)


#Classical optimizer (4 parameters)
from qiskit.primitives import StatevectorEstimator
from scipy.optimize import minimize

estimator = StatevectorEstimator()

def energy_scipy(params):
    params = np.atleast_2d(params) 
    pub = (circuit, qubit_hamiltonian, params)
    result = estimator.run([pub]).result()[0]
    return float(result.data.evs[0]) + ecore

res = minimize(energy_scipy, x0=np.array([3.0, 3.0, 3.0, 3.0]), method='COBYLA', options={'maxiter': 200, 'disp': False})

print("VQE energy (4 parametrs):", res.fun)
#print("Theta:", res.x)



#SU2 Ansatz
from qiskit.circuit.library import EfficientSU2

ansatz = EfficientSU2(num_qubits=4, reps=1, entanglement='linear')

estimator = StatevectorEstimator()
def energy_scipy(params):
    params = np.atleast_2d(params)  
    pub = (ansatz, qubit_hamiltonian, params)
    result = estimator.run([pub]).result()[0]
    return float(result.data.evs[0]) + ecore

init_params = np.zeros(len(ansatz.parameters))
res = minimize(energy_scipy, x0=init_params, method='COBYLA', options={'maxiter': 200, 'disp': False})
print("VQE energy (SU2):", res.fun)
#print("Theta:", res.x)