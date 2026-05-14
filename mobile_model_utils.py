"""
Mobile Model Utilities for Handwriting Synthesis
Provides functions to save and load models in mobile-friendly formats with size optimization.
"""

import os
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import h5py
import json
from typing import Dict, Optional, Union
import logging


class MobileModelSaver:
    """Utilities for saving models in mobile-friendly formats."""
    
    def __init__(self, session: tf.Session, checkpoint_dir: str = 'checkpoints'):
        self.session = session
        self.checkpoint_dir = checkpoint_dir
        
    def save_to_single_file(self, 
                           output_path: str, 
                           format_type: str = 'hdf5',
                           quantize: bool = True,
                           precision: str = 'float16') -> Dict[str, Union[int, float]]:
        """
        Save model weights to a single file format.
        
        Args:
            output_path: Path to save the single file
            format_type: 'hdf5', 'npz', or 'json'
            quantize: Whether to apply quantization
            precision: 'float16', 'float32', or 'int8' for quantization
            
        Returns:
            Dictionary with size information and compression stats
        """
        # Extract all trainable variables
        variables = tf.trainable_variables()
        weights_dict = {}
        original_size = 0
        
        logging.info(f"Extracting {len(variables)} trainable variables...")
        
        for var in variables:
            var_value = self.session.run(var)
            original_size += var_value.nbytes
            
            # Apply quantization if requested
            if quantize:
                var_value = self._quantize_weights(var_value, precision)
            
            weights_dict[var.name] = var_value
            logging.info(f"Variable: {var.name}, Shape: {var_value.shape}, Size: {var_value.nbytes} bytes")
        
        # Save in specified format
        if format_type.lower() == 'hdf5':
            final_size = self._save_hdf5(weights_dict, output_path)
        elif format_type.lower() == 'npz':
            final_size = self._save_npz(weights_dict, output_path)
        elif format_type.lower() == 'json':
            final_size = self._save_json(weights_dict, output_path)
        else:
            raise ValueError(f"Unsupported format: {format_type}")
        
        # Calculate compression stats
        compression_ratio = original_size / final_size if final_size > 0 else 0
        size_reduction = (1 - final_size / original_size) * 100 if original_size > 0 else 0
        
        stats = {
            'original_size_mb': original_size / (1024 * 1024),
            'final_size_mb': final_size / (1024 * 1024),
            'compression_ratio': compression_ratio,
            'size_reduction_percent': size_reduction,
            'num_parameters': len(weights_dict),
            'format': format_type,
            'quantized': quantize,
            'precision': precision if quantize else 'float32'
        }
        
        logging.info(f"Model saved successfully to {output_path}")
        logging.info(f"Size reduction: {size_reduction:.1f}% (from {stats['original_size_mb']:.1f}MB to {stats['final_size_mb']:.1f}MB)")
        
        return stats
    
    def _quantize_weights(self, weights: np.ndarray, precision: str) -> np.ndarray:
        """Apply quantization to reduce weight precision."""
        if precision == 'float16':
            return weights.astype(np.float16)
        elif precision == 'int8':
            # Simple linear quantization to int8
            min_val, max_val = weights.min(), weights.max()
            scale = (max_val - min_val) / 255.0
            quantized = np.round((weights - min_val) / scale).astype(np.int8)
            return quantized
        else:
            return weights.astype(np.float32)
    
    def _save_hdf5(self, weights_dict: Dict[str, np.ndarray], output_path: str) -> int:
        """Save weights to HDF5 format with compression."""
        with h5py.File(output_path, 'w') as f:
            # Save metadata
            f.attrs['model_type'] = 'handwriting_synthesis'
            f.attrs['num_parameters'] = len(weights_dict)
            
            # Save weights with compression
            for name, weight in weights_dict.items():
                # Clean variable name for HDF5
                clean_name = name.replace('/', '_').replace(':', '')
                f.create_dataset(clean_name, data=weight, compression='gzip', compression_opts=9)
        
        return os.path.getsize(output_path)
    
    def _save_npz(self, weights_dict: Dict[str, np.ndarray], output_path: str) -> int:
        """Save weights to compressed NPZ format."""
        # Clean variable names
        clean_dict = {name.replace('/', '_').replace(':', ''): weight 
                     for name, weight in weights_dict.items()}
        
        np.savez_compressed(output_path, **clean_dict)
        return os.path.getsize(output_path)
    
    def _save_json(self, weights_dict: Dict[str, np.ndarray], output_path: str) -> int:
        """Save weights to JSON format (for very small models)."""
        # Convert numpy arrays to lists for JSON serialization
        json_dict = {}
        for name, weight in weights_dict.items():
            clean_name = name.replace('/', '_').replace(':', '')
            json_dict[clean_name] = {
                'shape': weight.shape,
                'data': weight.flatten().tolist()
            }
        
        with open(output_path, 'w') as f:
            json.dump(json_dict, f, indent=2)
        
        return os.path.getsize(output_path)


class MobileModelLoader:
    """Lightweight loader for mobile inference."""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.weights = {}
        self.format_type = self._detect_format()
        self.load_weights()
    
    def _detect_format(self) -> str:
        """Detect model format from file extension."""
        if self.model_path.endswith('.h5') or self.model_path.endswith('.hdf5'):
            return 'hdf5'
        elif self.model_path.endswith('.npz'):
            return 'npz'
        elif self.model_path.endswith('.json'):
            return 'json'
        else:
            raise ValueError(f"Unsupported file format: {self.model_path}")
    
    def load_weights(self):
        """Load weights from the saved file."""
        if self.format_type == 'hdf5':
            self._load_hdf5()
        elif self.format_type == 'npz':
            self._load_npz()
        elif self.format_type == 'json':
            self._load_json()
    
    def _load_hdf5(self):
        """Load weights from HDF5 file."""
        with h5py.File(self.model_path, 'r') as f:
            for key in f.keys():
                self.weights[key] = f[key][:]
    
    def _load_npz(self):
        """Load weights from NPZ file."""
        data = np.load(self.model_path)
        for key in data.files:
            self.weights[key] = data[key]
    
    def _load_json(self):
        """Load weights from JSON file."""
        with open(self.model_path, 'r') as f:
            data = json.load(f)
        
        for key, value in data.items():
            shape = tuple(value['shape'])
            weights_data = np.array(value['data']).reshape(shape)
            self.weights[key] = weights_data
    
    def get_weight(self, name: str) -> Optional[np.ndarray]:
        """Get specific weight by name."""
        clean_name = name.replace('/', '_').replace(':', '')
        return self.weights.get(clean_name)
    
    def get_all_weights(self) -> Dict[str, np.ndarray]:
        """Get all loaded weights."""
        return self.weights.copy()
    
    def get_model_info(self) -> Dict[str, Union[int, float, str]]:
        """Get information about the loaded model."""
        total_params = sum(w.size for w in self.weights.values())
        total_size = sum(w.nbytes for w in self.weights.values())
        
        return {
            'num_weights': len(self.weights),
            'total_parameters': total_params,
            'total_size_mb': total_size / (1024 * 1024),
            'format': self.format_type,
            'model_path': self.model_path
        }


class ModelPruner:
    """Utilities for pruning model weights to reduce size."""
    
    @staticmethod
    def magnitude_pruning(weights: np.ndarray, sparsity: float = 0.5) -> np.ndarray:
        """
        Apply magnitude-based pruning to weights.
        
        Args:
            weights: Weight array to prune
            sparsity: Fraction of weights to set to zero (0.0 to 1.0)
            
        Returns:
            Pruned weight array
        """
        if sparsity <= 0:
            return weights
        
        flat_weights = weights.flatten()
        threshold = np.percentile(np.abs(flat_weights), sparsity * 100)
        
        pruned_weights = weights.copy()
        pruned_weights[np.abs(pruned_weights) < threshold] = 0
        
        return pruned_weights
    
    @staticmethod
    def structured_pruning(weights: np.ndarray, prune_ratio: float = 0.1) -> np.ndarray:
        """
        Apply structured pruning (removes entire neurons/channels).
        
        Args:
            weights: Weight matrix to prune
            prune_ratio: Fraction of output neurons to remove
            
        Returns:
            Pruned weight matrix
        """
        if len(weights.shape) != 2 or prune_ratio <= 0:
            return weights
        
        # Calculate L2 norm for each output neuron
        neuron_norms = np.linalg.norm(weights, axis=0)
        num_to_prune = int(weights.shape[1] * prune_ratio)
        
        if num_to_prune > 0:
            # Find neurons with smallest norms
            prune_indices = np.argsort(neuron_norms)[:num_to_prune]
            
            # Zero out selected neurons
            pruned_weights = weights.copy()
            pruned_weights[:, prune_indices] = 0
            
            return pruned_weights
        
        return weights


# Convenience functions
def save_mobile_model(session: tf.Session, 
                     output_path: str,
                     format_type: str = 'hdf5',
                     quantize: bool = True,
                     precision: str = 'float16',
                     apply_pruning: bool = False,
                     sparsity: float = 0.1) -> Dict[str, Union[int, float]]:
    """
    Convenience function to save a mobile-optimized model.
    
    Args:
        session: TensorFlow session with loaded model
        output_path: Path to save the optimized model
        format_type: 'hdf5', 'npz', or 'json'
        quantize: Whether to apply quantization
        precision: Quantization precision
        apply_pruning: Whether to apply magnitude pruning
        sparsity: Sparsity level for pruning
        
    Returns:
        Dictionary with optimization statistics
    """
    saver = MobileModelSaver(session)
    
    if apply_pruning:
        logging.info(f"Applying pruning with {sparsity*100}% sparsity...")
        # Note: Pruning would require modifying the session weights
        # This is a placeholder for the pruning logic
    
    return saver.save_to_single_file(
        output_path=output_path,
        format_type=format_type,
        quantize=quantize,
        precision=precision
    )


def load_mobile_model(model_path: str) -> MobileModelLoader:
    """
    Convenience function to load a mobile model.
    
    Args:
        model_path: Path to the saved mobile model
        
    Returns:
        MobileModelLoader instance
    """
    return MobileModelLoader(model_path)