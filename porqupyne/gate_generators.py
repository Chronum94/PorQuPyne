import numpy as np
import scipy.sparse as ss


def _gate_gen_(n_qbits, qbit_op, target_qbits):
    """Generates any arbitrary gate using monobit-gates and identity matrices:
    Inputs:
        n_qbits: Number of qubits.
        qbit_op: The single-qubit operation. NOT, Hadamard,
                 phase shifts, anything qualifies as long as
                 it's on one bit.
        target_bits: Target qubit(s).
    """

    target_qbits = np.array(target_qbits)
    # A 3D stack of 2x2 arrays, to be Kronecker-producted.
    oparray = np.eye(2, dtype='complex128').reshape([1, 2, 2]).repeat(n_qbits,
                                                                      axis=0)

    # Replacing appropriate matrices with operator matrices.
    oparray[target_qbits - 1, :, :] = qbit_op

    # Creating an initial 1x1 identity matrix to keep Kroneck-producting to.
    finalop = ss.eye(1)

    # Creating the final operator matrix. COOrdinate format or
    # things will end VERY badly if BSR is used.
    for a in range(n_qbits):
        finalop = ss.kron(finalop, oparray[a], format='coo')
    # Returning the operator matrix.
    return finalop.asformat('csr')