#!/usr/bin/env python3
"""
Test Loop Singular Bit Compression System
=========================================

Test the core compression algorithm to understand how it works.
"""

import torch
import numpy as np
import psutil
import time
import sys
import os

# Add src to path
sys.path.append('src')

def test_compression_algorithm():
    """Test the core compression algorithm"""
    
    print("🧪 Testing Loop Singular Bit Compression Algorithm")
    print("=" * 60)
    
    try:
        from loop_singular_bit.quantization import OutlierPreservingQuantizer
        print("✅ Successfully imported OutlierPreservingQuantizer")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("🔄 Trying alternative import...")
        try:
            sys.path.append('src/loop_singular_bit')
            from quantization import OutlierPreservingQuantizer
            print("✅ Successfully imported with alternative path")
        except ImportError as e2:
            print(f"❌ Alternative import failed: {e2}")
            return
    
    # Test with different tensor sizes
    test_cases = [
        (100, 100, "Small tensor"),
        (1000, 1000, "Medium tensor"),
        (2048, 2048, "Large tensor (simulating model layer)")
    ]
    
    for rows, cols, description in test_cases:
        print(f"\n📊 Testing {description}: {rows}×{cols}")
        print("-" * 40)
        
        # Create test tensor (simulating model weights)
        test_tensor = torch.randn(rows, cols) * 0.1
        original_size_mb = test_tensor.numel() * 4 / (1024**2)
        
        print(f"   Original size: {original_size_mb:.2f} MB")
        print(f"   Parameters: {test_tensor.numel():,}")
        
        # Initialize quantizer with 2% outlier ratio
        quantizer = OutlierPreservingQuantizer(outlier_ratio=0.02)
        
        # Measure compression time
        start_time = time.time()
        
        # Perform quantization
        result = quantizer.quantize(test_tensor, f"{description}_weight")
        
        compression_time = time.time() - start_time
        
        # Extract results
        compression_ratio = result["compression_ratio"]
        quality_error = result["quality_error_percent"]
        compressed_size_mb = result["size_analysis"]["compressed_size_mb"]
        
        print(f"   ✅ Compression completed in {compression_time:.3f}s")
        print(f"   📈 Compression ratio: {compression_ratio:.2f}×")
        print(f"   📉 Compressed size: {compressed_size_mb:.2f} MB")
        print(f"   🎯 Quality error: {quality_error:.3f}%")
        
        # Test reconstruction
        reconstructed = quantizer.dequantize(result)
        max_diff = torch.max(torch.abs(test_tensor - reconstructed)).item()
        
        print(f"   🔄 Reconstruction max difference: {max_diff:.6f}")
        
        # Verify shapes match
        if test_tensor.shape == reconstructed.shape:
            print(f"   ✅ Shape preservation: {test_tensor.shape}")
        else:
            print(f"   ❌ Shape mismatch: {test_tensor.shape} vs {reconstructed.shape}")

def test_outlier_detection():
    """Test the outlier detection mechanism"""
    
    print(f"\n🔍 Testing Outlier Detection Mechanism")
    print("=" * 60)
    
    try:
        from loop_singular_bit.quantization import OutlierPreservingQuantizer
        
        # Create tensor with known outliers
        tensor = torch.randn(1000, 1000) * 0.01  # Small normal weights
        
        # Add some large outliers
        outlier_positions = [(100, 100), (200, 200), (300, 300), (400, 400), (500, 500)]
        for i, j in outlier_positions:
            tensor[i, j] = 1.0  # Large outlier value
        
        print(f"📊 Test tensor: {tensor.shape}")
        print(f"   Added {len(outlier_positions)} known outliers")
        print(f"   Normal weight range: ±{torch.std(tensor[tensor != 1.0]):.4f}")
        print(f"   Outlier values: {[tensor[i, j].item() for i, j in outlier_positions]}")
        
        # Test outlier detection
        quantizer = OutlierPreservingQuantizer(outlier_ratio=0.02)
        outlier_mask = quantizer.identify_outliers(tensor)
        
        detected_outliers = torch.sum(outlier_mask).item()
        total_weights = tensor.numel()
        detection_ratio = detected_outliers / total_weights
        
        print(f"   🎯 Target outlier ratio: 2.0%")
        print(f"   📈 Detected outliers: {detected_outliers}/{total_weights}")
        print(f"   📊 Actual detection ratio: {detection_ratio*100:.2f}%")
        
        # Check if known outliers were detected
        detected_known = sum(outlier_mask[i, j].item() for i, j in outlier_positions)
        print(f"   ✅ Known outliers detected: {detected_known}/{len(outlier_positions)}")
        
    except Exception as e:
        print(f"❌ Error in outlier detection test: {e}")

def test_memory_efficiency():
    """Test memory efficiency during compression"""
    
    print(f"\n💾 Testing Memory Efficiency")
    print("=" * 60)
    
    try:
        from loop_singular_bit.quantization import OutlierPreservingQuantizer
        
        # Monitor memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / (1024**2)
        
        print(f"📊 Initial memory usage: {initial_memory:.2f} MB")
        
        # Create large tensor
        large_tensor = torch.randn(4096, 4096)  # ~67M parameters
        tensor_memory = process.memory_info().rss / (1024**2)
        
        print(f"📈 Memory after tensor creation: {tensor_memory:.2f} MB")
        print(f"   Tensor size: {large_tensor.numel() * 4 / (1024**2):.2f} MB")
        
        # Perform compression
        quantizer = OutlierPreservingQuantizer(outlier_ratio=0.02)
        result = quantizer.quantize(large_tensor, "large_test_weight")
        
        compression_memory = process.memory_info().rss / (1024**2)
        
        print(f"📊 Memory after compression: {compression_memory:.2f} MB")
        print(f"   Memory increase: {compression_memory - tensor_memory:.2f} MB")
        print(f"   Compression ratio: {result['compression_ratio']:.2f}×")
        
        # Clean up
        del large_tensor, result
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        final_memory = process.memory_info().rss / (1024**2)
        print(f"📉 Memory after cleanup: {final_memory:.2f} MB")
        
    except Exception as e:
        print(f"❌ Error in memory efficiency test: {e}")

def main():
    """Run all compression tests"""
    
    print("🚀 Loop Singular Bit Compression System Test Suite")
    print("=" * 70)
    
    # Test basic compression algorithm
    test_compression_algorithm()
    
    # Test outlier detection
    test_outlier_detection()
    
    # Test memory efficiency
    test_memory_efficiency()
    
    print(f"\n🎉 All tests completed!")
    print("=" * 70)

if __name__ == "__main__":
    main()
