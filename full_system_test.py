#!/usr/bin/env python3
"""
Full Loop Singular Bit System Test
==================================

Test the complete compression system including the main compressor.
"""

import torch
import sys
import os
import psutil
import time

# Add the source directory to Python path
sys.path.insert(0, os.path.join(os.getcwd(), 'src', 'loop_singular_bit'))

def test_full_compressor():
    print("🚀 Testing Full Loop Singular Bit Compressor")
    print("=" * 60)
    
    try:
        # Import the full compressor
        from compressor import LoopCompressor
        print("✅ Successfully imported LoopCompressor")
        
        # Initialize compressor with realistic settings
        compressor = LoopCompressor(
            outlier_ratio=0.02,      # 2% outliers
            target_ram_mb=400,       # 400MB RAM target
            target_storage_gb=4.0,   # 4GB storage target
            quality_threshold=1.0    # 1% quality loss max
        )
        
        # Simulate model layer weights (like transformer layers)
        print(f"\n📊 Creating simulated model layers...")
        
        # Simulate different layer types
        layer_configs = [
            ("attention.q_proj", (4096, 4096)),
            ("attention.k_proj", (4096, 4096)), 
            ("attention.v_proj", (4096, 4096)),
            ("attention.o_proj", (4096, 4096)),
            ("mlp.gate_proj", (4096, 11008)),
            ("mlp.up_proj", (4096, 11008)),
            ("mlp.down_proj", (11008, 4096))
        ]
        
        total_params = 0
        total_original_size = 0
        total_compressed_size = 0
        compression_results = []
        
        print(f"   Testing {len(layer_configs)} layer types...")
        
        for layer_name, shape in layer_configs:
            print(f"\n🔄 Processing {layer_name}: {shape}")
            
            # Create layer weights
            layer_weights = {
                "weight": torch.randn(shape) * 0.02  # Realistic weight scale
            }
            
            layer_params = shape[0] * shape[1]
            layer_size_mb = layer_params * 4 / (1024**2)
            total_params += layer_params
            total_original_size += layer_size_mb
            
            print(f"   Parameters: {layer_params:,}")
            print(f"   Size: {layer_size_mb:.2f} MB")
            
            # Compress layer
            start_time = time.time()
            result = compressor.compress_layer(layer_weights, layer_name)
            compression_time = time.time() - start_time
            
            if result:
                compression_ratio = result.get('compression_ratio', 0)
                quality_error = result.get('quality_error_percent', 0)
                compressed_mb = result.get('compressed_size_mb', 0)
                
                total_compressed_size += compressed_mb
                compression_results.append(result)
                
                print(f"   ✅ Compressed in {compression_time:.3f}s")
                print(f"   📈 Ratio: {compression_ratio:.2f}×")
                print(f"   🎯 Quality: {quality_error:.3f}% error")
                print(f"   💾 Size: {layer_size_mb:.2f} → {compressed_mb:.2f} MB")
            else:
                print(f"   ❌ Compression failed")
        
        # Calculate overall statistics
        print(f"\n📊 Overall Compression Results")
        print("=" * 50)
        print(f"   Total parameters: {total_params:,}")
        print(f"   Original size: {total_original_size:.2f} MB")
        print(f"   Compressed size: {total_compressed_size:.2f} MB")
        
        if total_compressed_size > 0:
            overall_ratio = total_original_size / total_compressed_size
            print(f"   Overall compression: {overall_ratio:.2f}×")
            
            # Calculate average quality
            if compression_results:
                avg_quality = sum(r.get('quality_error_percent', 0) for r in compression_results) / len(compression_results)
                print(f"   Average quality error: {avg_quality:.3f}%")
        
        # Test memory usage
        process = psutil.Process()
        current_memory = process.memory_info().rss / (1024**2)
        print(f"   Current RAM usage: {current_memory:.2f} MB")
        
        print(f"\n🎉 Full system test completed!")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

def test_streaming_simulation():
    """Test streaming-like behavior"""
    
    print(f"\n🌊 Testing Streaming Simulation")
    print("=" * 60)
    
    try:
        from quantization import OutlierPreservingQuantizer
        
        # Simulate streaming by processing layers one at a time
        quantizer = OutlierPreservingQuantizer(outlier_ratio=0.02)
        
        # Simulate 32 transformer layers (like Mistral 7B)
        num_layers = 8  # Reduced for testing
        layer_size = (2048, 2048)  # Smaller for testing
        
        print(f"📊 Simulating {num_layers} layers of size {layer_size}")
        
        total_compression = 0
        total_quality_error = 0
        max_memory = 0
        
        process = psutil.Process()
        
        for layer_idx in range(num_layers):
            print(f"\n🔄 Processing layer {layer_idx + 1}/{num_layers}")
            
            # Create layer (simulating loading from disk)
            layer_tensor = torch.randn(layer_size) * 0.02
            
            # Measure memory before compression
            memory_before = process.memory_info().rss / (1024**2)
            
            # Compress layer
            result = quantizer.quantize(layer_tensor, f"layer_{layer_idx}")
            
            # Measure memory after compression
            memory_after = process.memory_info().rss / (1024**2)
            max_memory = max(max_memory, memory_after)
            
            # Accumulate statistics
            total_compression += result['compression_ratio']
            total_quality_error += result['quality_error_percent']
            
            print(f"   Compression: {result['compression_ratio']:.2f}×")
            print(f"   Quality: {result['quality_error_percent']:.3f}% error")
            print(f"   Memory: {memory_before:.1f} → {memory_after:.1f} MB")
            
            # Simulate unloading (cleanup)
            del layer_tensor, result
            
        # Calculate averages
        avg_compression = total_compression / num_layers
        avg_quality = total_quality_error / num_layers
        
        print(f"\n📊 Streaming Results:")
        print(f"   Average compression: {avg_compression:.2f}×")
        print(f"   Average quality error: {avg_quality:.3f}%")
        print(f"   Peak memory usage: {max_memory:.2f} MB")
        
    except Exception as e:
        print(f"❌ Streaming test error: {e}")

def main():
    """Run all tests"""
    
    print("🚀 Loop Singular Bit Full System Test Suite")
    print("=" * 70)
    
    # Test full compressor
    test_full_compressor()
    
    # Test streaming simulation
    test_streaming_simulation()
    
    print(f"\n🎉 All tests completed!")
    print("=" * 70)

if __name__ == "__main__":
    main()
