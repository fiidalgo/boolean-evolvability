import json
import os
from typing import Dict, Any, List, Optional
import pickle
import numpy as np
import datetime


def save_results_json(results: Dict[str, Any], filepath: str):
    """
    Save experiment results to a JSON file.
    Note: This only saves the scalar statistics, not the actual hypotheses.
    
    Args:
        results: Dictionary of experiment results
        filepath: Path to save the JSON file
    """
    # Create a copy of results to modify for serialization
    serializable_results = results.copy()
    
    # Convert any non-JSON-serializable types
    serializable_results['function_class'] = str(serializable_results['function_class'])
    
    # Convert NumPy arrays to lists if present
    for key, value in serializable_results.items():
        if isinstance(value, np.ndarray):
            serializable_results[key] = value.tolist()
    
    # Add timestamp
    serializable_results['timestamp'] = datetime.datetime.now().isoformat()
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Save to JSON
    with open(filepath, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"Results saved to {filepath}")


def load_results_json(filepath: str) -> Dict[str, Any]:
    """
    Load experiment results from a JSON file.
    
    Args:
        filepath: Path to the JSON file
        
    Returns:
        Dictionary of experiment results
    """
    with open(filepath, 'r') as f:
        results = json.load(f)
    return results


def save_full_experiment(results: Dict[str, Any], hypothesis_list: List, 
                         filepath: str):
    """
    Save full experiment data including hypotheses using pickle.
    
    Args:
        results: Dictionary of experiment results
        hypothesis_list: List of final hypotheses
        filepath: Path to save the pickle file
    """
    data = {
        'results': results,
        'hypotheses': hypothesis_list
    }
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Save with pickle
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"Full experiment data saved to {filepath}")


def load_full_experiment(filepath: str) -> Dict[str, Any]:
    """
    Load full experiment data from a pickle file.
    
    Args:
        filepath: Path to the pickle file
        
    Returns:
        Dictionary with 'results' and 'hypotheses' keys
    """
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data


def create_results_dir(experiment_name: str) -> str:
    """
    Create a directory for storing experiment results.
    
    Args:
        experiment_name: Name of the experiment
        
    Returns:
        Path to the created directory
    """
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    dir_name = f"results/{experiment_name}_{timestamp}"
    os.makedirs(dir_name, exist_ok=True)
    return dir_name 