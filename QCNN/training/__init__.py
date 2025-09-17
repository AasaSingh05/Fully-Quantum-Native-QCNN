from .Qtrainer import QuantumNativeTrainer

__all__ = ["QuantumNativeTrainer"]

TRAINING_ALGORITHMS = {
    "quantum_native": QuantumNativeTrainer
}

def create_trainer(trainer_type="quantum_native", learning_rate=0.05):
    if trainer_type not in TRAINING_ALGORITHMS:
        raise ValueError(f"Unknown trainer type: {trainer_type}")
    
    return TRAINING_ALGORITHMS[trainer_type](learning_rate=learning_rate)

def get_available_trainers():
    return list(TRAINING_ALGORITHMS.keys())
