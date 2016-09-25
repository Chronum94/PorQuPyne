# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 02:07:33 2016

@author: Chronum
"""
import numpy as np
import scipy.sparse as ss
from scipy.stats import itemfreq

__all__ = ['StateVector']
class StateVector():
    
    # Initialize a state vector for a given number of qubits.
    def __init__(self, num_bits):
        self.state = np.zeros(int(2**num_bits), dtype='complex64')
        self.bit_count = num_bits
        self.size = self.state.size
    
    # Initialize to a basis state.
    def init_basis(self, basis_state):
        """Initializes a given state vector to the desired basis state.
        Inputs:
            basis_state     : The basis state to which to initialize the 
                            state vector.
        """
        if basis_state > self.size:
            print('Basis state does not exist for '\
            'state vector of given length.')
            return
        self.state = np.zeros(self.size, dtype='complex64')
        self.state[basis_state-1] = 1
        return self.state
    
    # Initialize to a given cat state.
    def init_cat(self):
        """Initializes a given state vector to the cat state.
        Inputs:
            basis_state     : The basis state to which to initialize the 
                            state vector.
        """
        self.state[0], self.state[-1] = 1, 1
        return self.state
    
    # Initialize to a uniform random state.
    def init_rand(self):
        """Initializes a given state vector to the cat state.
        """
        state_size = self.size
        self.state = np.random.rand(state_size)+1j*np.random.rand(state_size)
        
        # SPEED: np.linalg.norm(a) is ~10-15x faster than:
        #        np.sum(np.abs(a)**2) or np.sum(np.conjugate(a)*a) at
        #        np.sum(np.abs(a)*np.abs(a))
        self.state = self.state/np.linalg.norm(self.state)
        return self.state
        
    def measure_state(self, measure_state, n=10000):
        state_vals = np.linspace(0, self.size-1, self.size)
        measure = np.random.choice(state_vals, n, p=np.abs(self.state)**2)
        state_count = np.count_nonzero(measure == measure_state)

        # print(unique, counts)
        return measure_state, state_count/n
        
    def measure_all(self,
                    n=10000,
                    show_first=5,
                    print_pretty=False,
                    sort=False):
        state_vals = np.linspace(0, self.size-1, self.size)
        measure = np.random.choice(state_vals, int(n), p=np.abs(self.state)**2)
        state_count = itemfreq(measure)
        state_count[:, 1] /= n
        if sort:
            state_count = np.take(state_count, 
                                  np.argsort(-state_count[:, 1]), axis=0)
        
        if print_pretty:
            for entry in state_count[:show_first, :]:
                print("|{}>: {:0.4f}".format(
                str(bin(int(entry[0]))[2:]).zfill(self.bit_count),
                    entry[1]))
        return state_count
        
    def print_state(self):
        print(self.state)
        
    ############################ PRIVATE METHODS ##############################
    def __gate_gen1(root_gate, gate_bit_size, total_bits, applied_bit):
        # TODO: Common method to generate arbitrary gates from root gates.
        return 0
    ###########################################################################
        
    ################################## GATES ##################################
    def apply_cnot(self, control_bit, not_bit, f='dia'):
        cnotgen = np.array([[1, 0, 0, 0],
                            [0, 1, 0, 0],\
                            [0, 0, 0, 1],\
                            [0, 0, 1, 0]])
        cnot = ss.kron(
        ss.eye(2**(self.bit_count-bit_num), format=f), 
        ss.kron(hadgen, ss.eye(2**(bit_num-1), format=f)))
        return 0
        
    
    def apply_hadamard(self, bit_num, f='dia'):
        """Apply a Hadamard gate to the quantum state vector.
        Inputs:
            bit_num:    The bit on which to apply the Hadamard operator.
        """
        hadgen = 0.7071067811865476 * np.array([[1, 1], [1, -1]])
        hadamard = ss.kron(
        ss.eye(2**(self.bit_count-bit_num), format=f), 
        ss.kron(hadgen, ss.eye(2**(bit_num-1), format=f)))
        # print(hadamard)
        self.state = hadamard.dot(self.state)
        return 0
        
    def apply_pauli(self, bit_num, axis=0, f='dia'):
        pauligen_store = np.array([[[0,1],[1,0]], 
                                   [[0,-1j],[1j,0]],\
                                   [[1,0],[0,-1]]])
        pauligen = pauligen_store[axis]
        pauli = ss.kron(
        ss.eye(2**(self.bit_count-bit_num), format=f, dtype='complex64'), 
        ss.kron(pauligen, ss.eye(2**(bit_num-1), 
        format=f, dtype='complex64')))
        self.state = pauli.dot(self.state)
        return 0
        
        
    def apply_phase(self, bit_num, theta, f='dia'):
        phasegen = np.array([[1, 0], [0, np.exp(1j*theta)]])   
        # print(np.sum(np.abs(phasegen)**2))
        phase = ss.kron(
        ss.eye(2**(self.bit_count-bit_num), format=f, dtype='complex64'), 
        ss.kron(phasegen, ss.eye(2**(bit_num-1), 
        format=f, dtype='complex64')))
        # print(phase.toarray())
        self.state = phase.dot(self.state)
        return 0
    ###########################################################################