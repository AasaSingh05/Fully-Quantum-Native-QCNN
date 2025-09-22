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
        """
        param_idx = 0

        # Scale parameters from data range to [0, 2pi]
        params_scaled = (params - np.min(params)) / (np.ptp(params) + 1e-10) * 2 * np.pi
        
        for out_qubit in output_qubits:
            for in_qubit in input_qubits:
                if param_idx >= len(params_scaled):
                    break
                
                # Skip controlled rotation on the same qubit
                if in_qubit == out_qubit:
                    continue
                
                qml.CRY(params_scaled[param_idx], wires=[in_qubit, out_qubit])
                param_idx += 1
                
                if param_idx >= len(params_scaled):
                    break
                
                qml.CRZ(params_scaled[param_idx], wires=[in_qubit, out_qubit])
                param_idx += 1
