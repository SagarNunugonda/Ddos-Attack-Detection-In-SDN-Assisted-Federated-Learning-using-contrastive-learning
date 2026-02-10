"""
Generate sample datasets for each network type.
This script creates synthetic data following the feature definitions in network_config.

Usage:
    python generate_sample_datasets.py --output data/
"""

import argparse
import pandas as pd
import numpy as np
from network_config import NETWORK_TYPES
import sys


def generate_network_data(network_type, num_samples=1000, attack_ratio=0.3, random_seed=42):
    """
    Generate synthetic network traffic data for a given network type.
    
    Args:
        network_type: Type of network (sdn, traditional, iot, hybrid)
        num_samples: Number of samples to generate
        attack_ratio: Ratio of attack samples to total
        random_seed: Random seed for reproducibility
    
    Returns:
        DataFrame with features and labels
    """
    np.random.seed(random_seed)
    
    config = NETWORK_TYPES[network_type]
    features = config["features"]
    
    # Split samples into normal and attack
    num_attacks = int(num_samples * attack_ratio)
    num_normal = num_samples - num_attacks
    
    data = {}
    
    print(f"Generating {num_samples} samples for {config['name']}...")
    print(f"  - Normal traffic: {num_normal}")
    print(f"  - Attack traffic: {num_attacks}")
    
    for feature in features:
        data[feature] = []
    
    # Generate normal traffic
    for _ in range(num_normal):
        for feature in features:
            if feature in ['protocol', 'routing_type']:
                # Categorical features
                data[feature].append(np.random.randint(0, 6))  # 0-5 protocols
            elif feature in ['device_id']:
                # Device ID (small range)
                data[feature].append(np.random.randint(1, 100))
            elif feature in ['signal_strength']:
                # Signal strength (-100 to 0 dBm)
                data[feature].append(np.random.uniform(-100, 0))
            elif feature in ['battery_level']:
                # Battery percentage (0-100)
                data[feature].append(np.random.uniform(20, 100))
            elif feature in ['error_rate']:
                # Error rate (0-0.1)
                data[feature].append(np.random.uniform(0, 0.1))
            elif feature in ['dt']:
                # Duration timestamp
                data[feature].append(np.random.randint(0, 3600))
            elif feature in ['switch']:
                # Switch ID
                data[feature].append(np.random.randint(1, 16))
            elif 'port_no' in feature or 'src_port' in feature or 'dst_port' in feature:
                # Port numbers
                data[feature].append(np.random.randint(1, 65536))
            elif 'pkt' in feature.lower():
                # Packet counts (normal: lower values)
                data[feature].append(np.random.exponential(100))
            elif 'byte' in feature.lower():
                # Byte counts (normal: lower values)
                data[feature].append(np.random.exponential(5000))
            elif 'dur' in feature.lower():
                # Duration (in seconds)
                data[feature].append(np.random.exponential(1))
            elif 'flow' in feature.lower():
                # Flow count
                data[feature].append(np.random.exponential(10))
            elif 'rate' in feature.lower() or 'kbps' in feature.lower():
                # Rate in kbps (normal: lower values)
                data[feature].append(np.random.exponential(100))
            else:
                # Default: random positive value
                data[feature].append(np.random.exponential(50))
    
    # Generate attack traffic (higher values, different patterns)
    for _ in range(num_attacks):
        for feature in features:
            if feature in ['protocol', 'routing_type']:
                data[feature].append(np.random.randint(0, 6))
            elif feature in ['device_id']:
                data[feature].append(np.random.randint(1, 100))
            elif feature in ['signal_strength']:
                data[feature].append(np.random.uniform(-100, -50))
            elif feature in ['battery_level']:
                data[feature].append(np.random.uniform(0, 30))
            elif feature in ['error_rate']:
                data[feature].append(np.random.uniform(0.3, 0.9))
            elif feature in ['dt']:
                data[feature].append(np.random.randint(0, 3600))
            elif feature in ['switch']:
                data[feature].append(np.random.randint(1, 16))
            elif 'port_no' in feature or 'src_port' in feature or 'dst_port' in feature:
                data[feature].append(np.random.randint(1, 65536))
            elif 'pkt' in feature.lower():
                # Attack: higher packet counts
                data[feature].append(np.random.exponential(2000))
            elif 'byte' in feature.lower():
                # Attack: higher byte counts
                data[feature].append(np.random.exponential(50000))
            elif 'dur' in feature.lower():
                # Attack: longer durations
                data[feature].append(np.random.exponential(10))
            elif 'flow' in feature.lower():
                # Attack: more flows
                data[feature].append(np.random.exponential(100))
            elif 'rate' in feature.lower() or 'kbps' in feature.lower():
                # Attack: higher rates
                data[feature].append(np.random.exponential(1000))
            else:
                data[feature].append(np.random.exponential(500))
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Add labels (0=Normal, 1=Attack)
    labels = np.concatenate([np.zeros(num_normal), np.ones(num_attacks)])
    df['Label'] = labels.astype(int)
    
    # Shuffle
    df = df.sample(frac=1).reset_index(drop=True)
    
    # Ensure all values are non-negative
    df = df.clip(lower=0)
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description='Generate sample datasets for each network type'
    )
    parser.add_argument('--output', default='data/',
                       help='Output directory for sample datasets')
    parser.add_argument('--samples', type=int, default=1000,
                       help='Number of samples per network type')
    parser.add_argument('--attack-ratio', type=float, default=0.3,
                       help='Ratio of attack samples (0-1)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    try:
        # Create output directory
        import os
        os.makedirs(args.output, exist_ok=True)
        
        # Generate data for each network type
        for network_type in NETWORK_TYPES.keys():
            df = generate_network_data(
                network_type,
                num_samples=args.samples,
                attack_ratio=args.attack_ratio,
                random_seed=args.seed
            )
            
            # Save to CSV
            output_file = f"{args.output}{network_type}_data.csv"
            df.to_csv(output_file, index=False)
            print(f"✓ Saved: {output_file} ({len(df)} samples)\n")
        
        print("✓ Sample dataset generation complete!")
        print(f"\nTo train models, use:")
        print(f"  python train_network_model.py --network-type sdn --data {args.output}sdn_data.csv")
        print(f"  python train_network_model.py --network-type traditional --data {args.output}traditional_data.csv")
        print(f"  python train_network_model.py --network-type iot --data {args.output}iot_data.csv")
        print(f"  python train_network_model.py --network-type hybrid --data {args.output}hybrid_data.csv")
        
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
