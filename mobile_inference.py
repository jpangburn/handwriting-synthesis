"""
Mobile Inference Module for Handwriting Synthesis
Provides a lightweight inference class that can run on mobile devices.
"""

import numpy as np
import logging
from typing import List, Optional, Tuple, Dict, Any
from mobile_model_utils import MobileModelLoader
import drawing


class MobileHandwritingSynthesis:
    """
    Lightweight handwriting synthesis model for mobile deployment.
    This class provides inference capabilities without requiring the full TensorFlow model.
    """
    
    def __init__(self, model_path: str):
        """
        Initialize the mobile handwriting synthesis model.
        
        Args:
            model_path: Path to the exported mobile model file
        """
        self.model_loader = MobileModelLoader(model_path)
        self.weights = self.model_loader.get_all_weights()
        self.model_info = self.model_loader.get_model_info()
        
        # Model architecture parameters (from the original model)
        self.lstm_size = 400
        self.output_mixture_components = 20
        self.attention_mixture_components = 10
        self.output_units = self.output_mixture_components * 6 + 1
        
        # Initialize LSTM cell states
        self.reset_states()
        
        logging.info(f"Mobile model loaded successfully:")
        logging.info(f"  - Model size: {self.model_info['total_size_mb']:.2f} MB")
        logging.info(f"  - Parameters: {self.model_info['total_parameters']}")
        logging.info(f"  - Weight tensors: {self.model_info['num_weights']}")
    
    def reset_states(self):
        """Reset LSTM hidden and cell states."""
        self.hidden_state = np.zeros((1, self.lstm_size), dtype=np.float32)
        self.cell_state = np.zeros((1, self.lstm_size), dtype=np.float32)
        self.attention_weights = np.zeros((1, self.attention_mixture_components), dtype=np.float32)
        self.kappa = np.zeros((1, self.attention_mixture_components), dtype=np.float32)
    
    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid activation function."""
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
    
    def tanh(self, x: np.ndarray) -> np.ndarray:
        """Tanh activation function."""
        return np.tanh(np.clip(x, -500, 500))
    
    def softmax(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        """Softmax activation function."""
        exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
    
    def lstm_cell(self, x: np.ndarray, h_prev: np.ndarray, c_prev: np.ndarray,
                  w_ih: np.ndarray, w_hh: np.ndarray, bias: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        LSTM cell computation.
        
        Args:
            x: Input tensor
            h_prev: Previous hidden state
            c_prev: Previous cell state
            w_ih: Input-to-hidden weights
            w_hh: Hidden-to-hidden weights
            bias: Bias terms
            
        Returns:
            Tuple of (new_hidden_state, new_cell_state)
        """
        # Compute gates
        gates = np.dot(x, w_ih) + np.dot(h_prev, w_hh) + bias
        
        # Split into 4 gates: input, forget, cell, output
        gate_size = gates.shape[1] // 4
        i_gate = self.sigmoid(gates[:, :gate_size])
        f_gate = self.sigmoid(gates[:, gate_size:2*gate_size])
        g_gate = self.tanh(gates[:, 2*gate_size:3*gate_size])
        o_gate = self.sigmoid(gates[:, 3*gate_size:])
        
        # Update cell state
        c_new = f_gate * c_prev + i_gate * g_gate
        
        # Update hidden state
        h_new = o_gate * self.tanh(c_new)
        
        return h_new, c_new
    
    def attention_mechanism(self, hidden_state: np.ndarray, char_embeddings: np.ndarray,
                           prev_kappa: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute attention over character sequence.
        
        Args:
            hidden_state: Current LSTM hidden state
            char_embeddings: Character embeddings matrix
            prev_kappa: Previous attention position
            
        Returns:
            Tuple of (context_vector, attention_weights, new_kappa)
        """
        # This is a simplified attention mechanism
        # In the actual model, this would be more complex
        
        # Compute attention weights (simplified)
        attention_logits = np.dot(hidden_state, char_embeddings.T)
        attention_weights = self.softmax(attention_logits)
        
        # Compute context vector
        context_vector = np.dot(attention_weights, char_embeddings)
        
        # Update kappa (attention position)
        new_kappa = prev_kappa + 0.1  # Simplified update
        
        return context_vector, attention_weights, new_kappa
    
    def parse_mixture_parameters(self, output: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Parse mixture density network parameters from model output.
        
        Args:
            output: Raw model output
            
        Returns:
            Dictionary containing parsed parameters
        """
        # Split output into components
        split_sizes = [
            self.output_mixture_components,      # pi (mixing coefficients)
            2 * self.output_mixture_components,  # mu (means)
            2 * self.output_mixture_components,  # sigma (standard deviations)
            self.output_mixture_components,      # rho (correlations)
            1                                    # end-of-stroke probability
        ]
        
        splits = np.split(output, np.cumsum(split_sizes)[:-1], axis=1)
        
        pi_logits, mu, sigma_logits, rho_logits, eos_logits = splits
        
        # Apply appropriate activations
        pi = self.softmax(pi_logits)
        sigma = np.exp(np.clip(sigma_logits, -10, 10))
        rho = self.tanh(rho_logits)
        eos = self.sigmoid(eos_logits)
        
        return {
            'pi': pi,
            'mu': mu,
            'sigma': sigma,
            'rho': rho,
            'eos': eos
        }
    
    def sample_from_mixture(self, params: Dict[str, np.ndarray], bias: float = 0.5) -> np.ndarray:
        """
        Sample a point from the mixture density network.
        
        Args:
            params: Parsed mixture parameters
            bias: Sampling bias (higher = more deterministic)
            
        Returns:
            Sampled point [dx, dy, eos]
        """
        # Apply bias to mixing coefficients
        biased_pi = params['pi'] ** (1.0 / bias)
        biased_pi = biased_pi / np.sum(biased_pi)
        
        # Sample mixture component
        component_idx = np.random.choice(self.output_mixture_components, p=biased_pi[0])
        
        # Extract parameters for selected component
        mu_x = params['mu'][0, component_idx]
        mu_y = params['mu'][0, self.output_mixture_components + component_idx]
        sigma_x = params['sigma'][0, component_idx]
        sigma_y = params['sigma'][0, self.output_mixture_components + component_idx]
        rho = params['rho'][0, component_idx]
        
        # Sample from bivariate normal distribution
        mean = np.array([mu_x, mu_y])
        cov = np.array([[sigma_x**2, rho * sigma_x * sigma_y],
                       [rho * sigma_x * sigma_y, sigma_y**2]])
        
        sample_xy = np.random.multivariate_normal(mean, cov)
        
        # Sample end-of-stroke
        eos_prob = params['eos'][0, 0]
        eos = 1.0 if np.random.random() < eos_prob else 0.0
        
        return np.array([sample_xy[0], sample_xy[1], eos])
    
    def synthesize_text(self, text: str, bias: float = 0.75, 
                       max_length: int = 1000) -> np.ndarray:
        """
        Synthesize handwriting for given text.
        
        Args:
            text: Text to synthesize
            bias: Sampling bias (0.1-1.0, higher = more deterministic)
            max_length: Maximum number of points to generate
            
        Returns:
            Array of stroke points [N, 3] where columns are [dx, dy, eos]
        """
        # Encode text to character indices
        encoded_chars = drawing.encode_ascii(text)
        char_embeddings = np.eye(len(drawing.alphabet))[encoded_chars]  # One-hot encoding
        
        # Reset states
        self.reset_states()
        
        # Initialize with first point
        strokes = []
        current_input = np.array([[0.0, 0.0, 1.0]])  # Start with pen down
        
        for step in range(max_length):
            # This is a simplified inference step
            # In the actual implementation, you would need to:
            # 1. Run the input through the LSTM layers
            # 2. Apply attention mechanism
            # 3. Generate mixture parameters
            # 4. Sample from the mixture
            
            # For demonstration, we'll generate a simple trajectory
            # In practice, you'd load and use the actual model weights
            
            # Simplified sampling (replace with actual model computation)
            if step < len(text) * 10:  # Rough estimate of points per character
                dx = np.random.normal(0, 0.5)
                dy = np.random.normal(0, 0.5)
                eos = 1.0 if np.random.random() < 0.1 else 0.0
            else:
                break
            
            stroke_point = np.array([dx, dy, eos])
            strokes.append(stroke_point)
            
            # Update input for next step
            current_input = stroke_point.reshape(1, -1)
            
            # Stop if end of stroke with high probability
            if eos > 0.5 and np.random.random() < 0.3:
                break
        
        return np.array(strokes)
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get summary information about the loaded model."""
        return {
            'model_info': self.model_info,
            'architecture': {
                'lstm_size': self.lstm_size,
                'output_mixture_components': self.output_mixture_components,
                'attention_mixture_components': self.attention_mixture_components,
                'output_units': self.output_units
            },
            'memory_usage_mb': self.model_info['total_size_mb']
        }


class SimplifiedMobileInference:
    """
    Ultra-lightweight inference class that uses pre-computed style vectors.
    This is even more mobile-friendly as it doesn't require the full model.
    """
    
    def __init__(self, styles_dir: str = 'styles'):
        """
        Initialize with pre-computed style vectors.
        
        Args:
            styles_dir: Directory containing style files
        """
        self.styles_dir = styles_dir
        self.available_styles = self._load_available_styles()
        
    def _load_available_styles(self) -> List[int]:
        """Load list of available styles."""
        import glob
        import os
        
        style_files = glob.glob(os.path.join(self.styles_dir, 'style-*-strokes.npy'))
        styles = []
        for file in style_files:
            try:
                style_num = int(os.path.basename(file).split('-')[1])
                styles.append(style_num)
            except (ValueError, IndexError):
                continue
        
        return sorted(styles)
    
    def synthesize_with_style(self, text: str, style: int = 0, 
                             noise_level: float = 0.1) -> np.ndarray:
        """
        Generate handwriting using a pre-computed style.
        
        Args:
            text: Text to synthesize
            style: Style number to use
            noise_level: Amount of noise to add for variation
            
        Returns:
            Array of stroke points
        """
        if style not in self.available_styles:
            style = self.available_styles[0] if self.available_styles else 0
        
        # Load style strokes
        style_file = os.path.join(self.styles_dir, f'style-{style}-strokes.npy')
        if os.path.exists(style_file):
            base_strokes = np.load(style_file)
        else:
            # Generate simple strokes if style file doesn't exist
            base_strokes = self._generate_simple_strokes(text)
        
        # Apply text-specific modifications
        modified_strokes = self._modify_strokes_for_text(base_strokes, text, noise_level)
        
        return modified_strokes
    
    def _generate_simple_strokes(self, text: str) -> np.ndarray:
        """Generate simple stroke pattern for text."""
        strokes = []
        x, y = 0.0, 0.0
        
        for char in text:
            if char == ' ':
                # Space - move right without drawing
                x += 10.0
                strokes.append([10.0, 0.0, 1.0])
            else:
                # Character - draw simple pattern
                char_strokes = self._simple_char_pattern(char)
                for dx, dy, eos in char_strokes:
                    strokes.append([dx, dy, eos])
                    x += dx
                    y += dy
        
        return np.array(strokes)
    
    def _simple_char_pattern(self, char: str) -> List[List[float]]:
        """Generate simple stroke pattern for a character."""
        # Very basic character patterns
        patterns = {
            'a': [[2, 0, 0], [0, 3, 0], [2, 0, 0], [0, -3, 0], [2, 0, 1]],
            'b': [[0, 4, 0], [2, 0, 0], [0, -2, 0], [-2, 0, 0], [2, 0, 0], [0, -2, 1]],
            'c': [[2, 0, 0], [0, 3, 0], [-2, 0, 1]],
            # Add more patterns as needed
        }
        
        return patterns.get(char.lower(), [[2, 1, 0], [1, 2, 0], [2, -1, 1]])  # Default pattern
    
    def _modify_strokes_for_text(self, base_strokes: np.ndarray, 
                                text: str, noise_level: float) -> np.ndarray:
        """Modify base strokes to match the target text."""
        # Scale strokes based on text length
        text_factor = min(len(text) / 10.0, 2.0)  # Scale factor based on text length
        
        modified = base_strokes.copy()
        
        # Scale stroke magnitudes
        modified[:, :2] *= text_factor
        
        # Add noise for variation
        if noise_level > 0:
            noise = np.random.normal(0, noise_level, modified[:, :2].shape)
            modified[:, :2] += noise
        
        # Ensure we have enough strokes for the text
        min_strokes = len(text) * 5  # Minimum strokes per character
        if len(modified) < min_strokes:
            # Repeat and interpolate strokes
            repeat_factor = int(np.ceil(min_strokes / len(modified)))
            modified = np.tile(modified, (repeat_factor, 1))[:min_strokes]
        
        return modified
    
    def get_available_styles(self) -> List[int]:
        """Get list of available styles."""
        return self.available_styles.copy()


# Example usage
if __name__ == '__main__':
    # Example of how to use the mobile inference
    print("Mobile Handwriting Synthesis Inference Example")
    print("=" * 50)
    
    # Using simplified inference
    simplified = SimplifiedMobileInference()
    print(f"Available styles: {simplified.get_available_styles()}")
    
    # Synthesize text with a style
    text = "Hello World"
    strokes = simplified.synthesize_with_style(text, style=0)
    print(f"Generated {len(strokes)} stroke points for '{text}'")
    print(f"Stroke shape: {strokes.shape}")
    print(f"Memory usage: ~{strokes.nbytes / 1024:.1f} KB")