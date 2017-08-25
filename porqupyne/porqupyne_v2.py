import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as ssp
from scipy.stats import itemfreq


class StateVector:

    sqrt2 = np.sqrt(2)
    def __init__(self, num_qubits):
        """Initialize a quantum state vector consisting of n qubits
        The state vector space has 2**n bases (since qubits)
        """
        self.state = np.zeros(int(2**num_qubits), dtype='complex64')
        self.num_qubits = num_qubits
        self.size = self.state.size

        # Initialize to a normalized state.
        self.state[0] = 1

    def basis(self, basis_state):

        if type(basis_state) is int:
            try:
                self.state = np.zeros(self.size, dtype='complex64')
                self.state[basis_state - 1] = 1
            except IndexError:
                raise
        elif type(basis_state) is str:
            try:
                basis_state = int(basis_state, 2)
                self.state = np.zeros(self.size, dtype='complex64')
                self.state[basis_state - 1] = 1
            except IndexError:
                raise

    def cat(self):
        self.state[0] = self.state[1] = 1/StateVector.sqrt2

    def randu(self):
        state_size = self.size
        self.state = np.random.rand(state_size) + \
                     1j * np.random.rand(state_size)
        # SPEED: np.linalg.norm(a) is ~10-15x faster than:
        #        np.sum(np.abs(a)**2) or np.sum(np.conjugate(a)*a) at
        #        np.sum(np.abs(a)*np.abs(a))
        self.state = self.state / np.linalg.norm(self.state)


    def cnot(self, control_qubit, target_qubit):

        # The NOT operator for a single qubit.
        op = np.array([[0, 1], [1, 0]])
