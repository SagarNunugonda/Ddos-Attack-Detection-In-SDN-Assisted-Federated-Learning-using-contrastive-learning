# Network configuration for multi-network DDoS detection support

NETWORK_TYPES = {
    "sdn": {
        "name": "SDN-Assisted Network",
        "description": "Software-Defined Network with centralized control",
        "features": [
            'dt', 'switch', 'pktcount', 'bytecount', 'dur', 'dur_nsec', 'tot_dur', 
            'flows', 'packetins', 'pktperflow', 'byteperflow', 'pktrate', 
            'Pairflow', 'port_no', 'tx_bytes', 'rx_bytes', 'tx_kbps', 'rx_kbps', 'tot_kbps'
        ],
        "model_prefix": "sdn",
        "feature_count": 19
    },
    "traditional": {
        "name": "Traditional Network",
        "description": "Conventional routed network",
        "features": [
            'pktcount', 'bytecount', 'dur', 'dur_nsec', 'tot_dur', 'flows',
            'pktperflow', 'byteperflow', 'pktrate', 'src_port', 'dst_port',
            'protocol', 'tx_bytes', 'rx_bytes', 'tx_kbps', 'rx_kbps', 'tot_kbps'
        ],
        "model_prefix": "traditional",
        "feature_count": 17
    },
    "iot": {
        "name": "IoT Network",
        "description": "Internet of Things network with constrained devices",
        "features": [
            'pktcount', 'bytecount', 'dur', 'flows', 'pktperflow', 'byteperflow',
            'pktrate', 'device_id', 'signal_strength', 'battery_level', 'error_rate',
            'tx_bytes', 'rx_bytes', 'tx_kbps', 'rx_kbps', 'tot_kbps'
        ],
        "model_prefix": "iot",
        "feature_count": 16
    },
    "hybrid": {
        "name": "Hybrid Network",
        "description": "Mixed SDN and Traditional network architecture",
        "features": [
            'pktcount', 'bytecount', 'dur', 'dur_nsec', 'tot_dur', 'flows',
            'packetins', 'pktperflow', 'byteperflow', 'pktrate', 'port_no',
            'tx_bytes', 'rx_bytes', 'tx_kbps', 'rx_kbps', 'tot_kbps', 'routing_type'
        ],
        "model_prefix": "hybrid",
        "feature_count": 17
    }
}

def get_network_config(network_type):
    """Get configuration for a specific network type"""
    if network_type not in NETWORK_TYPES:
        raise ValueError(f"Unknown network type: {network_type}")
    return NETWORK_TYPES[network_type]

def get_features_for_network(network_type):
    """Get list of features for a specific network type"""
    config = get_network_config(network_type)
    return config["features"]

def get_feature_count(network_type):
    """Get feature count for a specific network type"""
    config = get_network_config(network_type)
    return config["feature_count"]

def validate_input_data(network_type, data_dict):
    """Validate that all required features are present"""
    required_features = get_features_for_network(network_type)
    missing_features = [f for f in required_features if f not in data_dict]
    if missing_features:
        raise ValueError(f"Missing features: {missing_features}")
    return True

def extract_features_in_order(network_type, data_dict):
    """Extract features in the correct order for the network type"""
    features = get_features_for_network(network_type)
    return [float(data_dict[f]) for f in features]
