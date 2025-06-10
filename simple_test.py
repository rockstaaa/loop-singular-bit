#!/usr/bin/env python3
"""
Simple test of Loop Singular Bit compression
"""

import torch
import sys
import os

# Add the source directory to Python path
sys.path.insert(0, os.path.join(os.getcwd(), 'src', 'loop_singular_bit'))

def test_compression():
    print("🧪 Testing Loop Singular Bit Compression")
    print("=" * 50)
    
    try:
        # Import the quantization module directly
        from quantization import OutlierPreservingQuantizer
        print("✅ Successfully imported OutlierPreservingQuantizer")
        
        # Create a test tensor
        test_tensor = torch.randn(500, 500) * 0.1
        print(f"📊 Test tensor: {test_tensor.shape}")
        print(f"   Size: {test_tensor.numel() * 4 / (1024**2):.2f} MB")
        
        # Initialize quantizer
        quantizer = OutlierPreservingQuantizer(outlier_ratio=0.02)
        
        # Perform compression
        print("\n🔄 Performing compression...")
        result = quantizer.quantize(test_tensor, "test_weight")
        
        # Show results
        print(f"\n✅ Compression Results:")
        print(f"   Compression ratio: {result['compression_ratio']:.2f}×")
        print(f"   Quality error: {result['quality_error_percent']:.3f}%")
        print(f"   Original size: {result['size_analysis']['original_size_mb']:.2f} MB")
        print(f"   Compressed size: {result['size_analysis']['compressed_size_mb']:.2f} MB")
        
        # Test reconstruction
        print(f"\n🔄 Testing reconstruction...")
        reconstructed = quantizer.dequantize(result)
        
        # Calculate reconstruction error
        max_error = torch.max(torch.abs(test_tensor - reconstructed)).item()
        mean_error = torch.mean(torch.abs(test_tensor - reconstructed)).item()
        
        print(f"   Max reconstruction error: {max_error:.6f}")
        print(f"   Mean reconstruction error: {mean_error:.6f}")
        print(f"   Shape match: {test_tensor.shape == reconstructed.shape}")
        
        print(f"\n🎉 Test completed successfully!")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("📁 Checking file structure...")
        
        # Check if files exist
        quantization_path = os.path.join('src', 'loop_singular_bit', 'quantization.py')
        if os.path.exists(quantization_path):
            print(f"✅ Found: {quantization_path}")
        else:
            print(f"❌ Missing: {quantization_path}")
            
        # List directory contents
        src_path = os.path.join('src', 'loop_singular_bit')
        if os.path.exists(src_path):
            files = os.listdir(src_path)
            print(f"📁 Contents of {src_path}: {files}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_compression()
