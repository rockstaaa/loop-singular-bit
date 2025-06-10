#!/usr/bin/env python3
"""
Basic Compression Example
========================

This example demonstrates basic usage of Loop Singular Bit compression
on a Mistral 7B model with default settings.
"""

import os
import sys
import time
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from loop_singular_bit import LoopCompressor


def main():
    """Basic compression example"""
    
    print("🚀 Loop Singular Bit - Basic Compression Example")
    print("=" * 60)
    
    # Configuration
    model_path = "downloaded_models/mistral-7b-v0.1"
    output_dir = "compressed_output"
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        print("Please download Mistral 7B model first:")
        print("  huggingface-cli download mistralai/Mistral-7B-v0.1")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize compressor with default settings
    compressor = LoopCompressor(
        outlier_ratio=0.02,      # Preserve top 2% weights
        target_ram_mb=400,       # Target 400MB RAM usage
        target_storage_gb=4.0,   # Target 4GB storage
        quality_threshold=1.0    # Accept up to 1% quality loss
    )
    
    print(f"\n📊 Configuration:")
    print(f"   Model: {model_path}")
    print(f"   Outlier ratio: 2%")
    print(f"   RAM target: 400MB")
    print(f"   Storage target: 4GB")
    print(f"   Quality threshold: 1%")
    
    # Start compression
    print(f"\n🔄 Starting compression...")
    start_time = time.time()
    
    try:
        # Compress model (start with 2 layers for demo)
        results = compressor.compress_model(model_path, max_layers=2)
        
        if results:
            # Validate results
            validation = compressor.validate_compression()
            
            # Save results
            results_file = compressor.save_results(output_dir)
            
            # Display results
            compression_time = time.time() - start_time
            
            print(f"\n✅ Compression completed in {compression_time:.1f}s")
            print(f"📄 Results saved: {results_file}")
            
            # Summary
            summary = results['compression_summary']
            projections = results['target_projections']
            
            print(f"\n📊 Compression Results:")
            print(f"   Layers compressed: {summary['layers_compressed']}")
            print(f"   Average compression: {summary['average_compression_ratio']:.2f}×")
            print(f"   Average quality loss: {summary['average_quality_loss_percent']:.2f}%")
            print(f"   Max RAM usage: {summary['max_ram_usage_mb']:.0f}MB")
            
            print(f"\n🎯 Target Projections:")
            print(f"   Projected RAM: {projections['projected_ram_mb']:.0f}MB")
            print(f"   RAM target: {'✅ ACHIEVED' if projections['ram_target_achieved'] else '❌ MISSED'}")
            print(f"   Projected storage: {projections['projected_storage_gb']:.1f}GB")
            print(f"   Storage target: {'✅ ACHIEVED' if projections['storage_target_achieved'] else '❌ MISSED'}")
            
            print(f"\n🔍 Validation Results:")
            if validation:
                comp_val = validation['compression_validation']
                target_val = validation['target_validation']
                
                print(f"   Quality acceptable: {'✅ YES' if comp_val['quality_acceptable'] else '❌ NO'}")
                print(f"   Targets met: {'✅ YES' if target_val['both_targets_met'] else '❌ NO'}")
                print(f"   Overall success: {'✅ YES' if validation['overall_success'] else '❌ NO'}")
            
            if projections['both_targets_achieved'] and validation.get('overall_success'):
                print(f"\n🎉 SUCCESS: Both targets achieved with acceptable quality!")
            else:
                print(f"\n⚠️ PARTIAL: Some targets not met - see results for details")
        
        else:
            print(f"\n❌ Compression failed - check model path and permissions")
    
    except Exception as e:
        print(f"\n❌ Error during compression: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
