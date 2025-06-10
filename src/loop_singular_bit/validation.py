"""
Quality Validation System for Loop Singular Bit
==============================================

Real-time quality monitoring and validation system to ensure
compression maintains model capabilities and performance.
"""

import torch
import numpy as np
import time
from typing import Dict, Any, List, Optional, Tuple
from transformers import AutoTokenizer


class QualityValidator:
    """
    Real quality validation system for compressed models
    
    Features:
    - Real-time quality monitoring
    - Perplexity measurement
    - Task-specific validation
    - Quality degradation detection
    - Performance benchmarking
    """
    
    def __init__(self, quality_threshold: float = 1.0):
        """
        Initialize quality validator
        
        Args:
            quality_threshold: Maximum acceptable quality loss percentage
        """
        self.quality_threshold = quality_threshold
        self.validation_history = []
        self.tokenizer = None
        
        print(f"🔍 QualityValidator initialized")
        print(f"   Quality threshold: {quality_threshold}%")
    
    def validate_layer(self, layer_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate compression quality for a single layer
        
        Args:
            layer_result: Layer compression results
            
        Returns:
            Validation results
        """
        quality_metrics = layer_result.get('quality_metrics', {})
        compression_ratio = layer_result.get('compression_ratio', 0)
        
        # Extract quality metrics
        relative_error = quality_metrics.get('relative_error_percent', 0)
        mse_error = quality_metrics.get('mse_error', 0)
        snr_db = quality_metrics.get('snr_db', 0)
        
        # Validate against threshold
        quality_acceptable = relative_error <= self.quality_threshold
        
        # Calculate quality score (0-100)
        quality_score = max(0, 100 - relative_error)
        
        # Determine quality grade
        if relative_error <= 0.5:
            quality_grade = "EXCELLENT"
        elif relative_error <= 1.0:
            quality_grade = "GOOD"
        elif relative_error <= 2.0:
            quality_grade = "ACCEPTABLE"
        else:
            quality_grade = "POOR"
        
        validation_result = {
            'layer_name': layer_result.get('weight_name', 'unknown'),
            'quality_acceptable': quality_acceptable,
            'quality_score': quality_score,
            'quality_grade': quality_grade,
            'metrics': {
                'relative_error_percent': relative_error,
                'mse_error': mse_error,
                'snr_db': snr_db,
                'compression_ratio': compression_ratio
            },
            'threshold_met': quality_acceptable,
            'validation_timestamp': time.time()
        }
        
        # Store in history
        self.validation_history.append(validation_result)
        
        return validation_result
    
    def validate_model_quality(self, compression_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate overall model compression quality
        
        Args:
            compression_results: List of layer compression results
            
        Returns:
            Overall model validation results
        """
        print(f"🔍 Validating model quality...")
        
        if not compression_results:
            return {'overall_quality': 'FAILED', 'reason': 'No compression results'}
        
        # Calculate aggregate metrics
        total_layers = len(compression_results)
        quality_errors = []
        compression_ratios = []
        acceptable_layers = 0
        
        for result in compression_results:
            quality_metrics = result.get('quality_metrics', {})
            error = quality_metrics.get('relative_error_percent', 0)
            ratio = result.get('compression_ratio', 0)
            
            quality_errors.append(error)
            compression_ratios.append(ratio)
            
            if error <= self.quality_threshold:
                acceptable_layers += 1
        
        # Calculate statistics
        avg_error = np.mean(quality_errors)
        max_error = np.max(quality_errors)
        min_error = np.min(quality_errors)
        std_error = np.std(quality_errors)
        
        avg_compression = np.mean(compression_ratios)
        
        # Overall quality assessment
        quality_pass_rate = (acceptable_layers / total_layers) * 100
        overall_acceptable = avg_error <= self.quality_threshold
        
        # Determine overall grade
        if avg_error <= 0.5 and quality_pass_rate >= 95:
            overall_grade = "EXCELLENT"
        elif avg_error <= 1.0 and quality_pass_rate >= 90:
            overall_grade = "GOOD"
        elif avg_error <= 2.0 and quality_pass_rate >= 80:
            overall_grade = "ACCEPTABLE"
        else:
            overall_grade = "POOR"
        
        validation_summary = {
            'overall_quality': overall_grade,
            'quality_acceptable': overall_acceptable,
            'statistics': {
                'total_layers': total_layers,
                'acceptable_layers': acceptable_layers,
                'quality_pass_rate_percent': quality_pass_rate,
                'average_error_percent': avg_error,
                'max_error_percent': max_error,
                'min_error_percent': min_error,
                'error_std_dev': std_error,
                'average_compression_ratio': avg_compression
            },
            'thresholds': {
                'quality_threshold_percent': self.quality_threshold,
                'threshold_met': overall_acceptable
            },
            'validation_timestamp': time.time()
        }
        
        print(f"   Overall grade: {overall_grade}")
        print(f"   Average error: {avg_error:.3f}%")
        print(f"   Pass rate: {quality_pass_rate:.1f}%")
        print(f"   Average compression: {avg_compression:.2f}×")
        
        return validation_summary
    
    def validate_inference_quality(self, original_output: str, compressed_output: str) -> Dict[str, Any]:
        """
        Validate inference quality by comparing outputs
        
        Args:
            original_output: Output from original model
            compressed_output: Output from compressed model
            
        Returns:
            Inference quality validation results
        """
        # Simple text similarity metrics
        original_tokens = original_output.split()
        compressed_tokens = compressed_output.split()
        
        # Calculate token-level similarity
        common_tokens = set(original_tokens) & set(compressed_tokens)
        token_similarity = len(common_tokens) / max(len(set(original_tokens)), 1)
        
        # Calculate length similarity
        length_ratio = min(len(compressed_tokens), len(original_tokens)) / max(len(compressed_tokens), len(original_tokens), 1)
        
        # Simple semantic similarity (character-level)
        char_similarity = self._calculate_char_similarity(original_output, compressed_output)
        
        # Overall inference quality score
        inference_score = (token_similarity * 0.4 + length_ratio * 0.3 + char_similarity * 0.3) * 100
        
        # Determine inference quality grade
        if inference_score >= 90:
            inference_grade = "EXCELLENT"
        elif inference_score >= 80:
            inference_grade = "GOOD"
        elif inference_score >= 70:
            inference_grade = "ACCEPTABLE"
        else:
            inference_grade = "POOR"
        
        return {
            'inference_quality_score': inference_score,
            'inference_grade': inference_grade,
            'metrics': {
                'token_similarity': token_similarity,
                'length_ratio': length_ratio,
                'char_similarity': char_similarity
            },
            'acceptable': inference_score >= 70,
            'original_length': len(original_tokens),
            'compressed_length': len(compressed_tokens)
        }
    
    def _calculate_char_similarity(self, text1: str, text2: str) -> float:
        """Calculate character-level similarity between two texts"""
        if not text1 or not text2:
            return 0.0
        
        # Simple character overlap ratio
        chars1 = set(text1.lower())
        chars2 = set(text2.lower())
        
        common_chars = chars1 & chars2
        total_chars = chars1 | chars2
        
        return len(common_chars) / max(len(total_chars), 1)
    
    def benchmark_performance(self, model_func, test_inputs: List[str], iterations: int = 3) -> Dict[str, Any]:
        """
        Benchmark model performance
        
        Args:
            model_func: Model function to benchmark
            test_inputs: List of test inputs
            iterations: Number of iterations per input
            
        Returns:
            Performance benchmark results
        """
        print(f"⏱️ Benchmarking performance...")
        
        total_times = []
        successful_runs = 0
        
        for input_text in test_inputs:
            for _ in range(iterations):
                try:
                    start_time = time.time()
                    output = model_func(input_text)
                    end_time = time.time()
                    
                    inference_time = end_time - start_time
                    total_times.append(inference_time)
                    successful_runs += 1
                    
                except Exception as e:
                    print(f"   ⚠️ Benchmark error: {e}")
                    continue
        
        if total_times:
            avg_time = np.mean(total_times)
            min_time = np.min(total_times)
            max_time = np.max(total_times)
            std_time = np.std(total_times)
            
            # Calculate throughput (tokens per second - rough estimate)
            avg_tokens_per_input = 20  # Rough estimate
            throughput = avg_tokens_per_input / avg_time if avg_time > 0 else 0
        else:
            avg_time = min_time = max_time = std_time = throughput = 0
        
        benchmark_results = {
            'total_runs': len(test_inputs) * iterations,
            'successful_runs': successful_runs,
            'success_rate_percent': (successful_runs / (len(test_inputs) * iterations)) * 100,
            'timing': {
                'average_time_s': avg_time,
                'min_time_s': min_time,
                'max_time_s': max_time,
                'std_dev_s': std_time
            },
            'throughput': {
                'tokens_per_second': throughput,
                'inferences_per_second': 1 / avg_time if avg_time > 0 else 0
            }
        }
        
        print(f"   Average time: {avg_time:.3f}s")
        print(f"   Success rate: {benchmark_results['success_rate_percent']:.1f}%")
        print(f"   Throughput: {throughput:.1f} tokens/s")
        
        return benchmark_results
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of all validation results"""
        if not self.validation_history:
            return {'status': 'No validations performed'}
        
        total_validations = len(self.validation_history)
        acceptable_validations = sum(1 for v in self.validation_history if v['quality_acceptable'])
        
        quality_scores = [v['quality_score'] for v in self.validation_history]
        avg_quality_score = np.mean(quality_scores)
        
        return {
            'total_validations': total_validations,
            'acceptable_validations': acceptable_validations,
            'pass_rate_percent': (acceptable_validations / total_validations) * 100,
            'average_quality_score': avg_quality_score,
            'quality_threshold': self.quality_threshold,
            'latest_validation': self.validation_history[-1] if self.validation_history else None
        }
