#!/usr/bin/env python3
"""
Export script to convert the handwriting synthesis model to mobile-friendly formats.
This script loads the existing checkpoint and saves it in optimized formats for mobile deployment.
"""

import os
import sys
import logging
import argparse
from mobile_model_utils import save_mobile_model, MobileModelLoader
from hand import Hand


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('mobile_export.log')
        ]
    )


def export_model(output_dir: str = 'mobile_models',
                format_type: str = 'hdf5',
                quantize: bool = True,
                precision: str = 'float16',
                apply_pruning: bool = False,
                sparsity: float = 0.1):
    """
    Export the handwriting synthesis model to mobile-friendly format.
    
    Args:
        output_dir: Directory to save the exported models
        format_type: Export format ('hdf5', 'npz', 'json')
        quantize: Whether to apply quantization
        precision: Quantization precision ('float16', 'int8', 'float32')
        apply_pruning: Whether to apply weight pruning
        sparsity: Sparsity level for pruning (0.0 to 1.0)
    """
    setup_logging()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    logging.info("Starting mobile model export...")
    logging.info(f"Export configuration:")
    logging.info(f"  - Output directory: {output_dir}")
    logging.info(f"  - Format: {format_type}")
    logging.info(f"  - Quantization: {quantize} ({'precision: ' + precision if quantize else 'disabled'})")
    logging.info(f"  - Pruning: {apply_pruning} ({'sparsity: ' + str(sparsity) if apply_pruning else 'disabled'})")
    
    try:
        # Load the existing model
        logging.info("Loading existing handwriting synthesis model...")
        hand = Hand()
        
        # Generate output filename
        filename_parts = ['handwriting_model']
        if quantize:
            filename_parts.append(f'{precision}')
        if apply_pruning:
            filename_parts.append(f'pruned_{int(sparsity*100)}pct')
        
        filename = '_'.join(filename_parts) + f'.{format_type}'
        output_path = os.path.join(output_dir, filename)
        
        # Export the model
        logging.info(f"Exporting model to {output_path}...")
        stats = save_mobile_model(
            session=hand.nn.session,
            output_path=output_path,
            format_type=format_type,
            quantize=quantize,
            precision=precision,
            apply_pruning=apply_pruning,
            sparsity=sparsity
        )
        
        # Print export statistics
        logging.info("Export completed successfully!")
        logging.info("Export Statistics:")
        logging.info(f"  - Original size: {stats['original_size_mb']:.2f} MB")
        logging.info(f"  - Final size: {stats['final_size_mb']:.2f} MB")
        logging.info(f"  - Size reduction: {stats['size_reduction_percent']:.1f}%")
        logging.info(f"  - Compression ratio: {stats['compression_ratio']:.2f}x")
        logging.info(f"  - Number of parameters: {stats['num_parameters']}")
        logging.info(f"  - Format: {stats['format']}")
        logging.info(f"  - Quantized: {stats['quantized']}")
        logging.info(f"  - Precision: {stats['precision']}")
        
        # Test loading the exported model
        logging.info("Testing exported model loading...")
        loader = MobileModelLoader(output_path)
        model_info = loader.get_model_info()
        
        logging.info("Model loading test successful!")
        logging.info(f"Loaded model info:")
        logging.info(f"  - Number of weight tensors: {model_info['num_weights']}")
        logging.info(f"  - Total parameters: {model_info['total_parameters']}")
        logging.info(f"  - Model size: {model_info['total_size_mb']:.2f} MB")
        
        # Save export metadata
        metadata_path = os.path.join(output_dir, f"{filename}_metadata.txt")
        with open(metadata_path, 'w') as f:
            f.write("Handwriting Synthesis Mobile Model Export\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Export Configuration:\n")
            f.write(f"  Format: {format_type}\n")
            f.write(f"  Quantization: {quantize}\n")
            f.write(f"  Precision: {precision if quantize else 'float32'}\n")
            f.write(f"  Pruning: {apply_pruning}\n")
            f.write(f"  Sparsity: {sparsity if apply_pruning else 'N/A'}\n\n")
            f.write(f"Export Statistics:\n")
            f.write(f"  Original size: {stats['original_size_mb']:.2f} MB\n")
            f.write(f"  Final size: {stats['final_size_mb']:.2f} MB\n")
            f.write(f"  Size reduction: {stats['size_reduction_percent']:.1f}%\n")
            f.write(f"  Compression ratio: {stats['compression_ratio']:.2f}x\n")
            f.write(f"  Number of parameters: {stats['num_parameters']}\n")
        
        logging.info(f"Export metadata saved to {metadata_path}")
        
        return output_path, stats
        
    except Exception as e:
        logging.error(f"Export failed: {str(e)}")
        raise


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description='Export handwriting synthesis model for mobile deployment')
    
    parser.add_argument('--output-dir', default='mobile_models',
                       help='Output directory for exported models (default: mobile_models)')
    
    parser.add_argument('--format', choices=['hdf5', 'npz', 'json'], default='hdf5',
                       help='Export format (default: hdf5)')
    
    parser.add_argument('--no-quantize', action='store_true',
                       help='Disable quantization (default: enabled)')
    
    parser.add_argument('--precision', choices=['float16', 'int8', 'float32'], default='float16',
                       help='Quantization precision (default: float16)')
    
    parser.add_argument('--enable-pruning', action='store_true',
                       help='Enable weight pruning (default: disabled)')
    
    parser.add_argument('--sparsity', type=float, default=0.1,
                       help='Sparsity level for pruning (0.0-1.0, default: 0.1)')
    
    parser.add_argument('--export-all', action='store_true',
                       help='Export all format variants')
    
    args = parser.parse_args()
    
    if args.export_all:
        # Export multiple variants
        configurations = [
            # Standard exports
            ('hdf5', True, 'float16', False, 0.0),
            ('npz', True, 'float16', False, 0.0),
            
            # Ultra-compressed exports
            ('hdf5', True, 'int8', False, 0.0),
            ('npz', True, 'int8', False, 0.0),
            
            # Pruned exports
            ('hdf5', True, 'float16', True, 0.1),
            ('hdf5', True, 'float16', True, 0.2),
        ]
        
        results = []
        for format_type, quantize, precision, apply_pruning, sparsity in configurations:
            try:
                logging.info(f"\n{'='*60}")
                logging.info(f"Exporting variant: {format_type}, {precision}, pruning={apply_pruning}")
                logging.info(f"{'='*60}")
                
                output_path, stats = export_model(
                    output_dir=args.output_dir,
                    format_type=format_type,
                    quantize=quantize,
                    precision=precision,
                    apply_pruning=apply_pruning,
                    sparsity=sparsity
                )
                results.append((output_path, stats))
            except Exception as e:
                logging.error(f"Failed to export variant: {e}")
        
        # Print summary
        logging.info(f"\n{'='*60}")
        logging.info("EXPORT SUMMARY")
        logging.info(f"{'='*60}")
        for output_path, stats in results:
            logging.info(f"{os.path.basename(output_path)}: {stats['final_size_mb']:.2f} MB ({stats['size_reduction_percent']:.1f}% reduction)")
    
    else:
        # Single export
        export_model(
            output_dir=args.output_dir,
            format_type=args.format,
            quantize=not args.no_quantize,
            precision=args.precision,
            apply_pruning=args.enable_pruning,
            sparsity=args.sparsity
        )


if __name__ == '__main__':
    main()