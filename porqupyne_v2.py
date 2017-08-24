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