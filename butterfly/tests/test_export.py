"""
Tests for quantum circuit export.
"""

import numpy as np

from butterfly import export_qft_qasm, export_unitary_seed_qasm


def run_tests_export(check, rng):
    """Tests for quantum circuit export."""
    print("\n── Quantum Circuit Export ──────────────────────────────────────")
    qasm_qft = export_qft_qasm(2, include_swaps=True)
    check(
        "QFT export generates non‑empty QASM",
        len(qasm_qft) > 0 and "OPENQASM" in qasm_qft,
    )

    H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    qasm_unitary = export_unitary_seed_qasm(H, n_qubits=1)
    check(
        "Unitary seed export generates non‑empty QASM",
        len(qasm_unitary) > 0 and "u3" in qasm_unitary,
    )

    rand_mat = rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))
    Q, R = np.linalg.qr(rand_mat)
    U4 = Q @ np.diag(np.sign(np.diag(R)))
    qasm_unitary4 = export_unitary_seed_qasm(U4, n_qubits=2)
    check(
        "d=4 unitary seed export generates non‑empty QASM",
        len(qasm_unitary4) > 0 and "opaque" in qasm_unitary4,
    )

    rand_mat3 = rng.standard_normal((3, 3)) + 1j * rng.standard_normal((3, 3))
    Q3, R3 = np.linalg.qr(rand_mat3)
    U3 = Q3 @ np.diag(np.sign(np.diag(R3)))
    qasm_unitary3 = export_unitary_seed_qasm(U3, n_qubits=1)
    check(
        "d=3 qudit seed export generates non‑empty QASM",
        len(qasm_unitary3) > 0 and "qudit" in qasm_unitary3,
    )
