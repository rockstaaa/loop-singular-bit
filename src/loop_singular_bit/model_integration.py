"""
Model Integration System for Loop Singular Bit
=============================================

Complete model integration with HuggingFace Transformers,
real model downloading, compression pipeline, and serving.
"""

import os
import torch
import json
import shutil
from typing import Dict, Any, Optional, List
from transformers import AutoTokenizer, AutoConfig, AutoModel
from huggingface_hub import snapshot_download
import psutil

from .compressor import LoopCompressor
from .inference import CompressedInferenceEngine
from .streaming import StreamingManager
from .validation import QualityValidator


class ModelIntegration:
    """
    Complete model integration system
    
    Features:
    - HuggingFace model downloading
    - Automatic compression pipeline
    - Model serving infrastructure
    - Batch inference optimization
    - GPU/CPU acceleration support
    """
    
    def __init__(self, cache_dir: str = "downloaded_models"):
        """
        Initialize model integration system
        
        Args:
            cache_dir: Directory to cache downloaded models
        """
        self.cache_dir = cache_dir
        self.compressed_models = {}
        self.inference_engines = {}
        
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)
        
        print(f"🔗 ModelIntegration initialized")
        print(f"   Cache directory: {cache_dir}")
    
    def download_model(self, model_name: str, force_download: bool = False) -> str:
        """
        Download model from HuggingFace Hub
        
        Args:
            model_name: HuggingFace model name
            force_download: Force re-download if model exists
            
        Returns:
            Path to downloaded model
        """
        model_path = os.path.join(self.cache_dir, model_name.replace("/", "_"))
        
        if os.path.exists(model_path) and not force_download:
            print(f"📁 Model already cached: {model_path}")
            return model_path
        
        print(f"📥 Downloading model: {model_name}")
        
        try:
            # Download model files
            snapshot_download(
                repo_id=model_name,
                local_dir=model_path,
                local_dir_use_symlinks=False
            )
            
            print(f"✅ Model downloaded: {model_path}")
            return model_path
            
        except Exception as e:
            print(f"❌ Download failed: {e}")
            return ""
    
    def compress_model(
        self, 
        model_name: str, 
        outlier_ratio: float = 0.02,
        target_ram_mb: int = 400,
        quality_threshold: float = 1.0
    ) -> Dict[str, Any]:
        """
        Download and compress a model
        
        Args:
            model_name: HuggingFace model name
            outlier_ratio: Outlier preservation ratio
            target_ram_mb: Target RAM usage
            quality_threshold: Quality threshold
            
        Returns:
            Compression results
        """
        print(f"🚀 Starting model compression pipeline: {model_name}")
        
        # Download model
        model_path = self.download_model(model_name)
        if not model_path:
            return {'success': False, 'error': 'Download failed'}
        
        # Initialize compressor
        compressor = LoopCompressor(
            outlier_ratio=outlier_ratio,
            target_ram_mb=target_ram_mb,
            quality_threshold=quality_threshold
        )
        
        # Compress model
        compression_results = compressor.compress_model(model_path)
        
        if compression_results:
            # Store compressed model
            self.compressed_models[model_name] = {
                'compression_results': compression_results,
                'model_path': model_path,
                'compressor': compressor
            }
            
            print(f"✅ Model compression completed: {model_name}")
            return {
                'success': True,
                'model_name': model_name,
                'compression_results': compression_results
            }
        else:
            return {'success': False, 'error': 'Compression failed'}
    
    def create_inference_engine(self, model_name: str) -> Optional[CompressedInferenceEngine]:
        """
        Create inference engine for compressed model
        
        Args:
            model_name: Name of compressed model
            
        Returns:
            Inference engine or None if failed
        """
        if model_name not in self.compressed_models:
            print(f"❌ Model {model_name} not found in compressed models")
            return None
        
        try:
            model_data = self.compressed_models[model_name]
            model_path = model_data['model_path']
            
            # Load tokenizer and config
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            config = AutoConfig.from_pretrained(model_path)
            
            # Extract compressed weights
            compression_results = model_data['compression_results']
            compressed_weights = {}
            
            for result in compression_results:
                if 'compressed_data' in result:
                    weight_name = result.get('weight_name', 'unknown')
                    compressed_weights[weight_name] = result
            
            # Create inference engine
            inference_engine = CompressedInferenceEngine(
                compressed_weights=compressed_weights,
                model_config=config,
                tokenizer=tokenizer
            )
            
            self.inference_engines[model_name] = inference_engine
            
            print(f"✅ Inference engine created: {model_name}")
            return inference_engine
            
        except Exception as e:
            print(f"❌ Failed to create inference engine: {e}")
            return None
    
    def generate_text(
        self, 
        model_name: str, 
        prompt: str, 
        max_tokens: int = 50,
        temperature: float = 0.7
    ) -> str:
        """
        Generate text using compressed model
        
        Args:
            model_name: Name of model to use
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated text
        """
        # Get or create inference engine
        if model_name not in self.inference_engines:
            engine = self.create_inference_engine(model_name)
            if not engine:
                return f"Error: Could not create inference engine for {model_name}"
        else:
            engine = self.inference_engines[model_name]
        
        # Generate text
        return engine.generate(prompt, max_tokens, temperature)
    
    def batch_inference(
        self, 
        model_name: str, 
        prompts: List[str],
        max_tokens: int = 50
    ) -> List[str]:
        """
        Perform batch inference
        
        Args:
            model_name: Name of model to use
            prompts: List of input prompts
            max_tokens: Maximum tokens per generation
            
        Returns:
            List of generated texts
        """
        print(f"🔄 Batch inference: {len(prompts)} prompts")
        
        results = []
        for i, prompt in enumerate(prompts):
            print(f"   Processing {i+1}/{len(prompts)}")
            result = self.generate_text(model_name, prompt, max_tokens)
            results.append(result)
        
        return results
    
    def benchmark_model(self, model_name: str, test_prompts: List[str]) -> Dict[str, Any]:
        """
        Benchmark compressed model performance
        
        Args:
            model_name: Name of model to benchmark
            test_prompts: List of test prompts
            
        Returns:
            Benchmark results
        """
        if model_name not in self.inference_engines:
            engine = self.create_inference_engine(model_name)
            if not engine:
                return {'error': 'Could not create inference engine'}
        else:
            engine = self.inference_engines[model_name]
        
        # Create validator
        validator = QualityValidator()
        
        # Benchmark function
        def model_func(prompt):
            return engine.generate(prompt, max_tokens=20)
        
        # Run benchmark
        benchmark_results = validator.benchmark_performance(
            model_func=model_func,
            test_inputs=test_prompts,
            iterations=2
        )
        
        return benchmark_results
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """
        Get information about compressed model
        
        Args:
            model_name: Name of model
            
        Returns:
            Model information
        """
        if model_name not in self.compressed_models:
            return {'error': 'Model not found'}
        
        model_data = self.compressed_models[model_name]
        compression_results = model_data['compression_results']
        
        # Calculate statistics
        if compression_results:
            total_layers = len(compression_results)
            avg_compression = sum(r.get('compression_ratio', 0) for r in compression_results) / total_layers
            avg_quality = sum(r.get('average_quality_loss_percent', 0) for r in compression_results) / total_layers
        else:
            total_layers = avg_compression = avg_quality = 0
        
        return {
            'model_name': model_name,
            'model_path': model_data['model_path'],
            'compressed_layers': total_layers,
            'average_compression_ratio': avg_compression,
            'average_quality_loss': avg_quality,
            'has_inference_engine': model_name in self.inference_engines,
            'memory_usage_mb': psutil.Process().memory_info().rss / (1024**2)
        }
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List all compressed models"""
        return [self.get_model_info(name) for name in self.compressed_models.keys()]
    
    def cleanup_model(self, model_name: str) -> None:
        """
        Clean up model from memory
        
        Args:
            model_name: Name of model to cleanup
        """
        if model_name in self.inference_engines:
            self.inference_engines[model_name].clear_cache()
            del self.inference_engines[model_name]
        
        if model_name in self.compressed_models:
            del self.compressed_models[model_name]
        
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        print(f"🧹 Model {model_name} cleaned up")
    
    def save_compressed_model(self, model_name: str, output_path: str) -> bool:
        """
        Save compressed model to disk
        
        Args:
            model_name: Name of model to save
            output_path: Output file path
            
        Returns:
            Success status
        """
        if model_name not in self.compressed_models:
            print(f"❌ Model {model_name} not found")
            return False
        
        try:
            model_data = self.compressed_models[model_name]
            
            save_data = {
                'model_name': model_name,
                'compression_results': model_data['compression_results'],
                'model_path': model_data['model_path'],
                'metadata': {
                    'compressed_layers': len(model_data['compression_results']),
                    'compression_timestamp': time.time()
                }
            }
            
            with open(output_path, 'w') as f:
                json.dump(save_data, f, indent=2, default=str)
            
            print(f"💾 Compressed model saved: {output_path}")
            return True
            
        except Exception as e:
            print(f"❌ Save failed: {e}")
            return False
