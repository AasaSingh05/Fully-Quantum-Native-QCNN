import numpy as np
import pennylane as qml


class QuantumNativePooling:
    @staticmethod
    def quantum_measurement_pooling(measure_qubits: list[int]) -> list[qml.measurements.ExpectationMP]:
        """
        Quantum pooling via partial measurement
        Measures some qubits, traces out others - pure quantum operation
        """
        measurements = []
        for qubit in measure_qubits:
            measurements.append(qml.expval(qml.PauliZ(qubit)))
        return measurements
    
    @staticmethod
    def quantum_unitary_pooling(params: np.ndarray, input_qubits: list[int], 
                                output_qubits: list[int]) -> None:
        """
        Quantum unitary pooling - compresses quantum information
        Uses controlled rotations to aggregate quantum states

        Updated semantics:
        - input_qubits are the KEEP wires (targets).
        - output_qubits are the DISCARD wires (controls).
        - No data-dependent scaling; params are trainable angles.
        - For each (keep <- discard) pair, apply CRY and CRZ from discard to keep,
          then a small RY on keep to consolidate information.
        """
        # Flatten and ensure float
        angles = np.array(params, dtype=float).reshape(-1)
        if angles.size == 0:
            return

        # We consume 3 angles per pair: [cry, crz, post_ry_keep]
        def triple(idx):
            a = angles[(3*idx) % angles.size]
            b = angles[(3*idx + 1) % angles.size]
            c = angles[(3*idx + 2) % angles.size]
            return a, b, c

        n_pairs = min(len(input_qubits), len(output_qubits))
        for i in range(n_pairs):
            keep = input_qubits[i]
            discard = output_qubits[i]

            # Skip accidental self-pairing
            if keep == discard:
                continue

            a, b, c = triple(i)

            # Controlled rotations from discard -> keep
            qml.CRY(a, wires=[discard, keep])
            qml.CRZ(b, wires=[discard, keep])

            # Light disentangle touch on discard (not used after pooling)
            qml.RY(0.02, wires=discard)

            # Consolidate on keep
            qml.RY(c, wires=keep)

        # After this, the model should drop 'output_qubits' from the active set to downsample by half.
