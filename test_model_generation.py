#!/usr/bin/env python3
"""
Test Loop Singular Bit Model Loading and Generation
===================================================

Test the complete model loading and text generation system.
"""

import sys
import os

def test_model_loading():
    """Test the model loading system"""
    
    print("🤖 Testing Loop Singular Bit Model Loading")
    print("=" * 60)
    
    try:
        # Import the main module
        sys.path.append('.')
        from loop_singular_bit import load_compressed_model, get_system_info
        
        print("✅ Successfully imported main functions")
        
        # Get system info
        print("\n📊 System Information:")
        system_info = get_system_info()
        for key, value in system_info.items():
            print(f"   {key}: {value}")
        
        # Test model loading
        print(f"\n🔄 Loading compressed model...")
        model = load_compressed_model("mistral-7b-v0.1")
        
        if model:
            print(f"✅ Model loaded successfully!")
            print(f"   Model name: {model.model_name}")
            print(f"   Compression ratio: {model.compression_ratio}×")
            print(f"   RAM usage: {model.ram_usage_mb}MB")
            print(f"   Quality: {model.quality_preservation}%")
            
            # Test text generation
            print(f"\n🔮 Testing text generation...")
            
            test_prompts = [
                "The future of artificial intelligence is",
                "Machine learning enables",
                "Quantum computing will",
                "The benefits of compression include"
            ]
            
            for prompt in test_prompts:
                print(f"\n📝 Prompt: '{prompt}'")
                try:
                    generated = model.generate(prompt, max_length=50)
                    print(f"   Generated: {generated[:100]}...")
                except Exception as e:
                    print(f"   ❌ Generation error: {e}")
            
            # Test model info
            print(f"\n📊 Model Information:")
            try:
                info = model.get_model_info()
                for key, value in info.items():
                    print(f"   {key}: {value}")
            except Exception as e:
                print(f"   ❌ Info error: {e}")
                
        else:
            print(f"❌ Model loading failed")
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("🔄 Trying direct file import...")
        
        try:
            # Try importing from the main file
            import loop_singular_bit as lsb
            print("✅ Direct import successful")
            
            # Test basic functionality
            system_info = lsb.get_system_info()
            print(f"📊 System info: {system_info}")
            
        except Exception as e2:
            print(f"❌ Direct import failed: {e2}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

def test_compression_engine():
    """Test the compression engine directly"""
    
    print(f"\n⚙️ Testing Compression Engine")
    print("=" * 60)
    
    try:
        # Import compression engine
        sys.path.append('compression')
        from loop_1bit_compressor import Loop1BitCompressor
        
        print("✅ Successfully imported Loop1BitCompressor")
        
        # Test with a small model path (simulated)
        print(f"\n🔄 Testing compression engine...")
        
        # Create a test directory structure
        test_model_path = "test_model"
        if not os.path.exists(test_model_path):
            os.makedirs(test_model_path)
            
        # Initialize compressor
        compressor = Loop1BitCompressor(test_model_path)
        
        # Test basic functionality
        print(f"   Model path: {compressor.model_path}")
        print(f"   Target RAM: {compressor.target_ram_mb}MB")
        
        # Test statistics
        stats = compressor.get_stats()
        print(f"📊 Compressor stats:")
        for key, value in stats.items():
            print(f"   {key}: {value}")
            
        print(f"✅ Compression engine test completed")
        
    except ImportError as e:
        print(f"❌ Compression engine import error: {e}")
        
    except Exception as e:
        print(f"❌ Compression engine error: {e}")

def test_research_validation():
    """Test research validation and results"""
    
    print(f"\n🔬 Testing Research Validation")
    print("=" * 60)
    
    try:
        # Check research files
        research_files = [
            "research/LOOP_Singular_Bit_Complete_Research_Paper.md",
            "research/outlier_preserving_quantization.md",
            "research/quality_preservation_study.md",
            "research/streaming_efficiency_analysis.md"
        ]
        
        for file_path in research_files:
            if os.path.exists(file_path):
                print(f"✅ Found: {file_path}")
                
                # Read first few lines to verify content
                with open(file_path, 'r', encoding='utf-8') as f:
                    first_line = f.readline().strip()
                    print(f"   Title: {first_line}")
            else:
                print(f"❌ Missing: {file_path}")
        
        # Check experiment results
        experiment_files = [
            "experiments/session_1_results.json",
            "experiments/session_2_results.json",
            "experiments/work_progress_log.json"
        ]
        
        print(f"\n📊 Experiment Results:")
        for file_path in experiment_files:
            if os.path.exists(file_path):
                print(f"✅ Found: {file_path}")
                
                # Try to read JSON content
                try:
                    import json
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        if isinstance(data, dict):
                            print(f"   Keys: {list(data.keys())}")
                        elif isinstance(data, list):
                            print(f"   Entries: {len(data)}")
                except Exception as e:
                    print(f"   ⚠️ Could not parse JSON: {e}")
            else:
                print(f"❌ Missing: {file_path}")
                
    except Exception as e:
        print(f"❌ Research validation error: {e}")

def main():
    """Run all model tests"""
    
    print("🚀 Loop Singular Bit Model Test Suite")
    print("=" * 70)
    
    # Test model loading and generation
    test_model_loading()
    
    # Test compression engine
    test_compression_engine()
    
    # Test research validation
    test_research_validation()
    
    print(f"\n🎉 All model tests completed!")
    print("=" * 70)

if __name__ == "__main__":
    main()
