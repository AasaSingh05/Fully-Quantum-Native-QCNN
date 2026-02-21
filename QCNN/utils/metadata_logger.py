import json, os, platform
import numpy as np
import importlib.metadata

def save_metadata(path, config):
    packages_to_check = ("pennylane", "numpy", "scikit-learn", "matplotlib")
    package_versions = {}
    
    for pkg in packages_to_check:
        try:
            package_versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            pass

    meta = {
        "config": config.__dict__,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "packages": package_versions,
        "seed": 42
    }

    with open(path, "w") as f:
        json.dump(meta, f, indent=4)
