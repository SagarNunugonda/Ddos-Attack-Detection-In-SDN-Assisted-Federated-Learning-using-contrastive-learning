#!/usr/bin/env python3
"""
Validation script to verify multi-network DDoS detection system setup.
Run this before deploying to production.
"""

import sys
import os
import importlib.util

def check_file_exists(filepath, description):
    """Check if a file exists"""
    if os.path.exists(filepath):
        print(f"✓ {description}: {filepath}")
        return True
    else:
        print(f"✗ {description}: NOT FOUND - {filepath}")
        return False

def check_directory_exists(dirpath, description):
    """Check if a directory exists"""
    if os.path.isdir(dirpath):
        print(f"✓ {description}: {dirpath}")
        return True
    else:
        print(f"✗ {description}: NOT FOUND - {dirpath}")
        return False

def check_python_module(module_name):
    """Check if a Python module is installed"""
    try:
        __import__(module_name)
        print(f"✓ Python module '{module_name}' is installed")
        return True
    except ImportError:
        print(f"✗ Python module '{module_name}' is NOT installed")
        return False

def check_network_config():
    """Verify network_config.py integrity"""
    try:
        from network_config import (
            NETWORK_TYPES, get_network_config, get_features_for_network,
            get_feature_count, validate_input_data, extract_features_in_order
        )
        
        print(f"\n✓ network_config.py loaded successfully")
        print(f"  Supported networks: {', '.join(NETWORK_TYPES.keys())}")
        
        for net_type, config in NETWORK_TYPES.items():
            feature_count = len(config['features'])
            print(f"  - {net_type}: {feature_count} features ({config['name']})")
        
        return True
    except Exception as e:
        print(f"✗ Error loading network_config.py: {str(e)}")
        return False

def check_models():
    """Check for trained models"""
    models_dir = "models"
    if not os.path.isdir(models_dir):
        print(f"\n⚠ Models directory not found: {models_dir}")
        return False
    
    print(f"\n✓ Models directory found")
    
    model_files = os.listdir(models_dir)
    if not model_files:
        print(f"⚠ No model files found in {models_dir}/")
        print(f"  Run: python train_network_model.py to train models")
        return False
    
    print(f"  Found {len(model_files)} files:")
    for filename in sorted(model_files):
        filepath = os.path.join(models_dir, filename)
        filesize = os.path.getsize(filepath)
        print(f"    - {filename} ({filesize} bytes)")
    
    return True

def check_templates():
    """Check if templates are present"""
    templates = ['base.html', 'dashboard.html', 'login.html', 'signup.html']
    all_exist = True
    
    print(f"\n✓ Checking templates/")
    for template in templates:
        filepath = os.path.join("templates", template)
        if os.path.exists(filepath):
            print(f"  ✓ {template}")
        else:
            print(f"  ✗ {template} NOT FOUND")
            all_exist = False
    
    return all_exist

def check_flask_app():
    """Verify Flask app can be imported"""
    try:
        from app import app, NETWORK_TYPES, AVAILABLE_MODELS
        print(f"\n✓ Flask app loaded successfully")
        print(f"  - Server name: {app.name}")
        print(f"  - Network types available: {list(AVAILABLE_MODELS.keys())}")
        
        for net_type, models in AVAILABLE_MODELS.items():
            models_ready = " & ".join([k for k, v in models.items() if v])
            status = "✓" if models_ready else "✗"
            print(f"  {status} {net_type}: {models_ready or 'NO MODELS'}")
        
        return True
    except Exception as e:
        print(f"\n✗ Error loading Flask app: {str(e)}")
        return False

def main():
    print("=" * 60)
    print("DDoS Detection System - Validation Check")
    print("=" * 60)
    
    results = []
    
    # Check core files
    print("\n--- CORE FILES ---")
    results.append(check_file_exists("app.py", "Flask application"))
    results.append(check_file_exists("network_config.py", "Network configuration"))
    results.append(check_file_exists("requirements.txt", "Requirements"))
    
    # Check utility scripts
    print("\n--- UTILITY SCRIPTS ---")
    results.append(check_file_exists("train_network_model.py", "Model training script"))
    results.append(check_file_exists("generate_sample_datasets.py", "Data generation script"))
    
    # Check directories
    print("\n--- DIRECTORIES ---")
    results.append(check_directory_exists("models", "Models directory"))
    results.append(check_directory_exists("templates", "Templates directory"))
    results.append(check_directory_exists("static", "Static files directory"))
    
    # Check documentation
    print("\n--- DOCUMENTATION ---")
    results.append(check_file_exists("README.md", "README"))
    results.append(check_file_exists("QUICK_REFERENCE.md", "Quick Reference"))
    results.append(check_file_exists("MULTI_NETWORK_SETUP.md", "Setup Guide"))
    results.append(check_file_exists("DEPLOYMENT_GUIDE.md", "Deployment Guide"))
    results.append(check_file_exists("INDEX.md", "Documentation Index"))
    
    # Check Python dependencies
    print("\n--- PYTHON DEPENDENCIES ---")
    dependencies = ['flask', 'torch', 'numpy', 'pandas', 'sklearn', 'joblib', 'matplotlib']
    for dep in dependencies:
        results.append(check_python_module(dep))
    
    # Check configuration
    print("\n--- CONFIGURATION ---")
    results.append(check_network_config())
    results.append(check_templates())
    results.append(check_models())
    
    # Check Flask app
    print("\n--- APPLICATION ---")
    results.append(check_flask_app())
    
    # Summary
    print("\n" + "=" * 60)
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"✓ ALL CHECKS PASSED ({passed}/{total})")
        print("\nSystem is ready to deploy!")
        return 0
    else:
        failed = total - passed
        print(f"⚠ SOME CHECKS FAILED ({passed}/{total})")
        print(f"\nIssues to resolve:")
        
        if not check_models.__wrapped__(None)[1]:
            print("  1. Train models: python train_network_model.py --network-type sdn --data data.csv")
        
        print(f"\nRun this script again to verify fixes.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
