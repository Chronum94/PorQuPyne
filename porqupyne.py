# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 02:07:33 2016

@author: Chronum
"""
# Bunch of imports. The itemfreq import is used for the measure methods.
# The time import is used to measure gate times.
# The jit import is used to speed up stuff involving loops, if any.
import numpy as np
import scipy.sparse as ss
from scipy.stats import itemfreq
import matplotlib.pyplot as plt
import time as t
from numba import jit


def _gate_gen_(n_bits, bit_op, t_bit):
    """Generates any arbitrary gate using monobit-gates and identity matrices:
    Inputs:
        n_bits      : Number of bits
        bit_op      : The single-bit operation. NOT, Hadamard, phase shifts
                    anything qualifies as long as it's on one bit.
        t_bit       : Target bit.
    """
    t_bit = np.array(t_bit)
    # print("Kek", bit_op)
    # A 3D stack of 2x2 arrays, to be Kronecker-producted.
    oparray = np.eye(2, dtype='complex128').reshape([1, 2, 2]).repeat(n_bits, axis=0)
    
    # Replacing appropriate matrices with operator matrices.    
    oparray[t_bit-1, :, :] = bit_op
    
    # Creating an initial 1x1 identity matrix to keep Kroneck-producting to.
    finalop = ss.eye(1)
    
    # Creating the final operator matrix. COOrdinate format or
    # things will end VERY badly if BSR is used.
    for a in range(n_bits):
        finalop = ss.kron(finalop, oparray[a], format='coo')
    # Returning the operator matrix.
    return finalop
  

def _cgate_gen_(n_bits, bit_op, c_bits, t_bits):
    """Generates any arbitrary gate using monobit-gates, control gates,
    and identity matrices:
    Inputs:
        bit_op      :The single-bit operation. NOT, Hadamard, phase shifts
                    anything qualifies as long as it's on one bit.
        c_bits      :The control bits. For a cnot gate, there is only one
                    control gate. Toffoli/Fredkin has 2. And so on.
        t_bit       : Target bit.
    """
    
    # This method generates a control-gate by using the general scheme:
    # For a 5-qubit system, if the 1st qubit is controlling a NOT on the
    # 4th qubit, the matrix is created by:
    # I_size + C X I_2 X I_2 X (NOT-I_2) X I_2
    # Where size = size of the statevector, 2**nbits.
    # I discovered this from a website which I can't trace anymore.
    # I WILL rediscover it. It was a gem.
    
    # List of target bits.
    t_bits = np.array(t_bits)
    oparray = np.eye(2).reshape([1, 2, 2]).repeat(n_bits, axis=0)
    
    # Control bit.
    # |0 0|
    # |0 1|
    oparray[c_bits-1, :, :] = np.array([[0, 0], [0, 1]])
    
    # Operator array.
    oparray[t_bits-1, :, :] = bit_op - np.eye(2)
    
    # Check previous methods.
    finalop = ss.eye(1)
    for a in range(n_bits):
        finalop = ss.kron(finalop, oparray[a], format='coo')
    
    # Add the final identity
    finalop += ss.eye(int(2**n_bits))
    # print(finalop.toarray())
    return finalop
    
# %% State vector and methods.
class StateVector():

    # Initialize a state vector for a given number of qubits.
    # Example:
    # a = StateVector(n)
    # An n-qubit state vector.
    def __init__(self, num_bits):
        self.state = np.zeros(int(2**num_bits), dtype='complex64')
        self.bit_count = num_bits
        self.size = self.state.size
    
    
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
    
    
    def init_cat(self):
        """Initializes a given state vector to the cat state.
        """
        self.state[0], self.state[-1] = 1, 1
        self.state = self.state/np.linalg.norm(self.state)
        return self.state
    
    # Initialize to a uniform random state.
    # This is NOT an equal superposition of all states.
    # It is simply a vector in which all state amplitudes are drawn from
    # a uniform random distribution in [0, 1).
    def init_rand(self):
        """Initializes a given state vector to a random state.
        """
        state_size = self.size
        self.state = np.random.rand(state_size)+1j*np.random.rand(state_size)
        
        # SPEED: np.linalg.norm(a) is ~10-15x faster than:
        #        np.sum(np.abs(a)**2) or np.sum(np.conjugate(a)*a) at
        #        np.sum(np.abs(a)*np.abs(a))
        self.state = self.state/np.linalg.norm(self.state)
        return self.state
        
        
# %% Measurement methods
    def measure_state(self, measure_state, n=100000):
        """Measures the specified state.
        Inputs:
            n   : Number of measurements to conduct on state.
        """
        state_vals = np.linspace(0, self.size-1, self.size)
        measure = np.random.choice(state_vals, n, p=np.abs(self.state)**2)
        state_count = np.count_nonzero(measure == measure_state-1)

        # print(unique, counts)
        return state_count/n
        
    def measure_all(self, n=100000, show_first=5, print_pretty=False,
                    sort=False, plot=False):
        """Measures all states.
        Inputs:
            show_first  : If print_pretty, specifies number of states to print.
            print_pretty: If true, prints states in qubit form.
            sort        : Specifies whether to sort output states.
        """
        # print("Measuring...")
        tbegin = t.time()
        state_vals = np.linspace(0, self.size-1, self.size)
        measure = np.random.choice(state_vals, int(n), p=np.abs(self.state)**2)
        state_count = itemfreq(measure)
        state_count[:, 1] /= n
        # print("Measuring done in {:0.3f} seconds.".format(t.time()-tbegin))
        if sort:
            state_count = np.take(state_count, 
                                  np.argsort(-state_count[:, 1]), axis=0)
        
        if print_pretty:
            bincount = 0
            for entry in state_count[:show_first, :]:
                print("|{}>: {:0.4f}".format(
                str(bin(int(entry[0]))[2:]).zfill(self.bit_count),
                    entry[1]))
        print('\n')
        if plot:
            if self.bit_count <= 10:
                bincount = int(2**self.bit_count)
            else:
                bincount = 100
            plt.hist(measure, bins=bincount)
            plt.show()
        
        return state_count
        
    def print_state(self):
        print(self.state)

    ############################ PRIVATE METHODS ##############################
    
    ###########################################################################
        
# %% Gate methods.
    ################################## GATES ##################################
    def cnot(self, control_bit, target_bit, f='dia'):
        """Applies a CNOT gate.
        Inputs:
            control_bit: The control bit.
            target_bit : The target bit(s).
        """
        # The NOT gate.
        #|0 1|
        #|1 0|
        op = np.array([[0, 1],[1, 0]])
        cnot = _cgate_gen_(self.bit_count, op, control_bit, target_bit)
        self.state = cnot.dot(self.state)
        
    def cphase(self, control_bit, target_bit, theta, f='dia'):
        """Applies a CPHASE gate.
        Inputs:
            control_bit: The control bit.
            target_bit : The target bit(s).
        """
        # The phase gate.
        #|1 0|
        #|0 t|
        op = np.array([[1, 0],[0, np.exp(1j*theta)]])
        cphase = _gate_gen_(self.bit_count, op, control_bit, target_bit)
        self.state = cphase.dot(self.state)
        
    
    def hadamard(self, bit_nums, f='dia'):
        """Applies a Hadamard gate to the quantum state vector.
        Inputs:
            bit_nums:    The bit(s) on which to apply the Hadamard operator.
                         Is a Numpy array of bits.
        """
        # A Hadamard gate.
        # sqrt(1/2)*|1  1|
        #           |1 -1|
        hadgen = 0.7071067811865476 * np.array([[1, 1], [1, -1]])
        bit_nums = np.array(bit_nums)
        stride_size = 4
        
        # The Hadamard gate is built and applied in strides.
        # Since a full Hadamard matrix is completely dense, it would be
        # prohibitive to apply a 20-qubit gate, for example, at once.
        # So we apply a small number of gates at a time. 4 is a good number
        # for even the largest matrices. You'll be running out of memory
        # by then anyway.
        
        # print("Commencing Hadamard...")
        tbegin = t.time()
        # print("Length:", len(bit_nums))
        if len(bit_nums) > stride_size:
            # Number of strides.
            n_strides = int(len(bit_nums)/stride_size)
            # print("Hadamard strides:", n_strides)
            
            # Number of 'leftover' bits.
            last_stride = len(bit_nums)%stride_size
            # print("Last stride:", last_stride)
            
            # Striding...
            for stride in range(n_strides):
                # Indices to apply had gate to.
                indices = np.linspace(stride_size*stride, 
                                      stride_size*(stride+1)-1, 
                                      stride_size, dtype='uint8')
                # print("Steps:", indices)
                # print("Bits:", bit_nums[indices])
                # Generating the had gate for current stride.
                hadamard = _gate_gen_(self.bit_count, hadgen, 
                                       bit_nums[indices])
                self.state = hadamard.dot(self.state)
            # Generating the had gate for the last <4 indices.
            # print("Last few bits:\n", bit_nums[-last_stride:])
            if last_stride != 0:
                hadamard = _gate_gen_(self.bit_count, 
                                      hadgen, 
                                      bit_nums[-last_stride:])
                self.state = hadamard.dot(self.state)
        else:
            hadamard = _gate_gen_(self.bit_count, hadgen, bit_nums)
        # print(bit_nums)
            self.state = hadamard.dot(self.state)
        # print("Hadamard done: {:0.3f} seconds.".format(t.time()-tbegin))
        
    def pauli(self, bit_num, axis=0, f='dia'):
        """Apply a Pauli gate to the quantum state vector.
        Inputs:
            bit_num:    The bit on which to apply the Hadamard operator.
            axis:       Integer for axis. 0=x, 1=y, 2=z
        """
        
        # Stores the 3 Pauli gates.
        # X     Y       Z
        #|0 1|  |0 -i|  |1  0|
        #|1 0|  |i  0|  |0 -1|
        pauligen_store = np.array([[[0,1],[1,0]], 
                                   [[0,-1j],[1j,0]],\
                                   [[1,0],[0,-1]]])
        pauligen = pauligen_store[axis]
        pauli = ss.kron(
        ss.eye(2**(bit_num-1), format=f, dtype='complex64'), 
        ss.kron(pauligen, ss.eye(2**(self.bit_count-bit_num), 
        format=f, dtype='complex64')))
        self.state = pauli.dot(self.state)
        return 0
        
        
    def phase(self, bit_num, theta, f='dia'):
        phasegen = np.array([[1, 0], [0, np.exp(1j*theta)]], dtype='complex64')   
        # print(np.sum(np.abs(phasegen)**2))
        phase = _gate_gen_(self.bit_count, phasegen, bit_num)
        # print(phase.toarray())
        self.state = phase.dot(self.state)
        return 0
        
    ###########################################################################