#!/usr/bin/env python3
"""
Quick Start Example for Quantum QCNN
Simple example to test the installation and basic functionality
Run this first to make sure everything is working before running main.py
"""

import sys
import os
import numpy as np

# Add project to path so we can import our modules
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def test_imports():
    """Test if all quantum QCNN modules can be imported"""
    print("📦 Testing imports...")
    
    try:
        from QCNN.config.Qconfig import QuantumNativeConfig
        print("   ✅ quantum_config imported")
        
        from QCNN.encoding.QEncoder import PureQuantumEncoder
        print("   ✅ quantum_encoder imported")
        
        from QCNN.layers.QConv import QuantumNativeConvolution
        from QCNN.layers.QPool import QuantumNativePooling
        print("   ✅ quantum layers imported")
        
        from QCNN.models.QCNNModel import PureQuantumNativeCNN
        print("   ✅ quantum_cnn imported")
        
        from QCNN.training.Qtrainer import QuantumNativeTrainer
        print("   ✅ quantum_trainer imported")
        
        from QCNN.utils.dataset_generator import generate_quantum_binary_dataset
        print("   ✅ data_generator imported")
        
        return True
        
    except ImportError as e:
        print(f"   ❌ Import failed: {e}")
        return False

def test_dependencies():
    """Test if all required dependencies are installed"""
    print("\n🔧 Testing dependencies...")
    
    dependencies = {
        'pennylane': 'PennyLane quantum computing framework',
        'numpy': 'NumPy for numerical computing',
        'sklearn': 'Scikit-learn for machine learning utilities',
        'matplotlib': 'Matplotlib for plotting (optional)'
    }
    
    all_good = True
    
    for package, description in dependencies.items():
        try:
            if package == 'sklearn':
                import sklearn
            else:
                __import__(package)
            print(f"   ✅ {package} - {description}")
        except ImportError:
            print(f"   ❌ {package} - {description} (MISSING)")
            all_good = False
    
    return all_good

def test_configuration():
    """Test quantum configuration creation"""
    print("\n⚙️  Testing configuration...")
    
    try:
        from QCNN.config.Qconfig import QuantumNativeConfig
        
        config = QuantumNativeConfig()
        print(f"   ✅ Config created successfully")
        print(f"   📊 Image size: {config.image_size}x{config.image_size}")
        print(f"   🔬 Qubits: {config.n_qubits}")
        print(f"   🏗️  Conv layers: {config.n_conv_layers}")
        print(f"   📡 Encoding: {config.encoding_type}")
        print(f"   🎯 Epochs: {config.n_epochs}")
        print(f"   📈 Learning rate: {config.learning_rate}")
        
        return True, config
        
    except Exception as e:
        print(f"   ❌ Configuration failed: {e}")
        return False, None

def test_data_generation():
    """Test quantum dataset generation"""
    print("\n📊 Testing data generation...")
    
    try:
        from QCNN.utils.dataset_generator import generate_quantum_binary_dataset
        
        # Generate small test dataset
        X, y = generate_quantum_binary_dataset(n_samples=20, image_size=4)
        
        print(f"   ✅ Dataset generated successfully")
        print(f"   📋 Shape: {X.shape}")
        print(f"   🏷️  Labels: {len(np.unique(y))} classes")
        print(f"   📊 Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
        print(f"   📏 Feature range: [{X.min():.3f}, {X.max():.3f}]")
        
        return True, X, y
        
    except Exception as e:
        print(f"   ❌ Data generation failed: {e}")
        return False, None, None

def test_model_creation():
    """Test quantum model creation"""
    print("\n🔧 Testing model creation...")
    
    try:
        from QCNN.config.Qconfig import QuantumNativeConfig
        from QCNN.models.QCNNModel import PureQuantumNativeCNN
        
        config = QuantumNativeConfig()
        model = PureQuantumNativeCNN(config)
        
        print(f"   ✅ Quantum model created successfully")
        
        # Count parameters
        total_params = sum(np.prod(p.shape) for p in model.quantum_params.values())
        print(f"   🔢 Total quantum parameters: {total_params}")
        print(f"   🌌 Hilbert space dimension: 2^{config.n_qubits} = {2**config.n_qubits}")
        
        return True, model
        
    except Exception as e:
        print(f"   ❌ Model creation failed: {e}")
        return False, None

def test_single_prediction():
    """Test a single quantum prediction"""
    print("\n🎯 Testing quantum prediction...")
    
    try:
        from QCNN.config.Qconfig import QuantumNativeConfig
        from QCNN.models.QCNNModel import PureQuantumNativeCNN
        from QCNN.utils.dataset_generator import generate_quantum_binary_dataset
        
        # Create small test setup
        config = QuantumNativeConfig()
        model = PureQuantumNativeCNN(config)
        X, y = generate_quantum_binary_dataset(n_samples=5, image_size=4)
        
        # Make a single prediction
        prediction = model.quantum_predict_single(X[0])
        
        print(f"   ✅ Quantum prediction successful")
        print(f"   📊 Input shape: {X[0].shape}")
        print(f"   🔮 Quantum output: {prediction:.6f}")
        print(f"   🏷️  True label: {y[0]}")
        print(f"   ✨ Binary prediction: {1 if prediction > 0 else -1}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Prediction failed: {e}")
        return False

def quick_test():
    """Run all quick tests"""
    
    print("🔬 Quantum QCNN Quick Start Test")
    print("=" * 50)
    print("Testing installation and basic functionality...")
    print("This should take about 30-60 seconds to complete.")
    
    # Run all tests
    tests = [
        ("Dependencies", test_dependencies),
        ("Imports", test_imports),
        ("Configuration", lambda: test_configuration()[0]),
        ("Data Generation", lambda: test_data_generation()[0]),
        ("Model Creation", lambda: test_model_creation()[0]),
        ("Single Prediction", test_single_prediction),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"   ❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 QUICK TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status:8} | {test_name}")
    
    print("-" * 50)
    print(f"Total: {passed}/{total} tests passed ({passed/total*100:.0f}%)")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        print("🚀 Ready to run the full experiment:")
        print("   python main.py")
        return True
    else:
        print("\n⚠️  SOME TESTS FAILED!")
        print("🔧 Please fix the issues above before running main.py")
        
        # Provide specific help
        if not results[0][1]:  # Dependencies failed
            print("\n💡 Install missing dependencies:")
            print("   pip install pennylane numpy scikit-learn matplotlib")
        
        if not results[1][1]:  # Imports failed
            print("\n💡 Check file structure - make sure all files are in place")
        
        return False

if __name__ == "__main__":
    success = quick_test()
    
    if success:
        print("\n🌟 Everything looks good!")
        print("🚀 Next step: python main.py")
    else:
        print("\n🔧 Please resolve the issues above first.")
