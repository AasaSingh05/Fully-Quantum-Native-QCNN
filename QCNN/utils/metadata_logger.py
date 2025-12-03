import json, os, platform
import numpy as np
import pkg_resources

def save_metadata(path, config):
    meta = {
        "config": config.__dict__,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "packages": {p.key: p.version for p in pkg_resources.working_set
                     if p.key in ("pennylane", "numpy", "scikit-learn", "matplotlib")},
        "seed": 42
    }

    with open(path, "w") as f:
        json.dump(meta, f, indent=4)
