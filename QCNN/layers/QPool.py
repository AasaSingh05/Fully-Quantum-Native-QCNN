import numpy as np
import pennylane as qml


class QuantumNativePooling:
    """
    Quantum-native pooling operations for QCNN architectures.
    Provides both measurement-based pooling and fully unitary
    pooling mechanisms for coherent downsampling of qubit grids.
    """

    @staticmethod
    def quantum_measurement_pooling(measure_qubits: list[int]) -> list[qml.measurements.ExpectationMP]:
        """
        Quantum pooling via partial measurement.
        Measures the selected qubits with expval(Z), enabling
        a measurement-based reduction of quantum information.

        Args:
            measure_qubits: list of qubit indices to measure

        Returns:
            List of expectation value measurement objects.
        """
        measurements = []
        for qubit in measure_qubits:
            measurements.append(qml.expval(qml.PauliZ(qubit)))
        return measurements
    
    @staticmethod
    def quantum_unitary_pooling(params: np.ndarray, input_qubits: list[int], 
                                output_qubits: list[int]) -> None:
        """
        Quantum unitary pooling – coherently compresses local quantum information.
        
        Updated semantics:
        - input_qubits are the KEEP wires (targets).
        - output_qubits are the DISCARD wires (controls).
        - Uses CRY/CRZ controlled rotations from discard → keep.
        - Applies a small disentangling RY on discard.
        - Applies consolidation RY on keep.
        
        This implements a fully unitary downsampling layer
        consistent with quantum-native QCNN designs.

        Args:
            params: flat or vector-like array of trainable pooling angles
            input_qubits: list of qubits to keep after pooling
            output_qubits: list of qubits to discard after pooling
        """
        # Flatten while preserving autograd types (avoid dtype=float casts)
        angles = np.asarray(params).reshape(-1)
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

    @staticmethod
    def make_pairing(active_wires: list[int]) -> list[tuple[int, int]]:
        """
        Default pairing strategy for pooling.
        Groups consecutive active wires into (keep, discard) pairs.
        
        Args:
            active_wires: list of active qubit indices

        Returns:
            List of (keep, discard) tuples.
        """
        pairs = []
        for i in range(0, len(active_wires) - 1, 2):
            keep = active_wires[i]
            discard = active_wires[i + 1]
            pairs.append((keep, discard))
        return pairs
