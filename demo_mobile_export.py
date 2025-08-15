#!/usr/bin/env python3
"""
Demo script showing how to export and use the mobile handwriting synthesis model.
"""

import os
import sys
import logging
from export_mobile_model import export_model
from mobile_inference import SimplifiedMobileInference


def main():
    """Demo the mobile model export and inference."""
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    print("Handwriting Synthesis Mobile Export Demo")
    print("=" * 50)
    
    # Check if we have the original model
    if not os.path.exists('checkpoints/model-17900.meta'):
        print("ERROR: Original model not found in checkpoints/")
        print("Please ensure you have trained the model first.")
        return
    
    try:
        # 1. Export model to mobile format
        print("\n1. Exporting model to mobile-friendly format...")
        output_path, stats = export_model(
            output_dir='mobile_models',
            format_type='hdf5',
            quantize=True,
            precision='float16',
            apply_pruning=False,
            sparsity=0.0
        )
        
        print(f"✅ Model exported successfully!")
        print(f"   Original size: {stats['original_size_mb']:.2f} MB")
        print(f"   Mobile size: {stats['final_size_mb']:.2f} MB")
        print(f"   Size reduction: {stats['size_reduction_percent']:.1f}%")
        
        # 2. Demo simplified inference (doesn't require the exported model)
        print("\n2. Testing simplified mobile inference...")
        
        mobile_inference = SimplifiedMobileInference('styles')
        available_styles = mobile_inference.get_available_styles()
        
        if available_styles:
            print(f"✅ Found {len(available_styles)} available styles: {available_styles}")
            
            # Test text synthesis
            test_texts = ["Hello", "Mobile AI", "Quick test"]
            
            for text in test_texts:
                for style in available_styles[:2]:  # Test first 2 styles
                    strokes = mobile_inference.synthesize_with_style(text, style=style)
                    print(f"   '{text}' with style {style}: {len(strokes)} stroke points ({strokes.nbytes} bytes)")
        else:
            print("⚠️  No style files found. Creating demo with simple patterns...")
            strokes = mobile_inference.synthesize_with_style("Hello", style=0)
            print(f"   Demo synthesis: {len(strokes)} stroke points")
        
        # 3. Show memory efficiency
        print("\n3. Memory Efficiency Analysis:")
        print(f"   Original TF model: ~{stats['original_size_mb']:.1f} MB")
        print(f"   Mobile HDF5 model: ~{stats['final_size_mb']:.1f} MB")
        
        if available_styles:
            style_file_size = 0
            for style in available_styles[:3]:
                style_path = f"styles/style-{style}-strokes.npy"
                if os.path.exists(style_path):
                    style_file_size += os.path.getsize(style_path)
            
            if style_file_size > 0:
                style_size_mb = style_file_size / (1024 * 1024)
                print(f"   Style vectors (3 styles): ~{style_size_mb:.2f} MB")
                print(f"   Total mobile deployment: ~{stats['final_size_mb'] + style_size_mb:.2f} MB")
        
        # 4. Performance recommendations
        print("\n4. Mobile Deployment Recommendations:")
        print("   📱 For basic mobile apps:")
        print("      - Use SimplifiedMobileInference with pre-computed styles")
        print("      - Memory footprint: < 5 MB")
        print("      - Good for style transfer and simple text rendering")
        
        print("   🚀 For advanced mobile apps:")
        print(f"      - Use full mobile model ({stats['final_size_mb']:.1f} MB)")
        print("      - Supports dynamic text generation")
        print("      - Requires more processing power")
        
        print("   ⚡ For edge devices:")
        print("      - Consider INT8 quantization for further size reduction")
        print("      - Apply pruning for faster inference")
        print("      - Use model distillation if accuracy allows")
        
        # 5. Usage examples
        print("\n5. Integration Examples:")
        print("   # Basic usage")
        print("   mobile = SimplifiedMobileInference('styles')")
        print("   strokes = mobile.synthesize_with_style('Hello World', style=2)")
        print("   # Convert strokes to drawing/image")
        
        print("\n✅ Demo completed successfully!")
        print("\nNext steps:")
        print("1. Integrate SimplifiedMobileInference into your mobile app")
        print("2. Test performance on target mobile devices")
        print("3. Optimize further based on your specific requirements")
        
    except Exception as e:
        print(f"❌ Demo failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()