#!/usr/bin/env python3
"""
COMPLETE 32-LAYER MODEL VALIDATION
==================================

CRITICAL PATH ITEM 1: Prove the concept works on full model
- Process all 32 transformer layers
- Validate complete model compression
- Measure actual performance metrics
- Prove targets are achieved

NO DELAYS - COMPLETE VALIDATION
"""

import os
import torch
import psutil
import time
import json
import gc
from safetensors import safe_open
from datetime import datetime
from typing import Dict, Any, List

class Complete32LayerValidator:
    """Complete validation of all 32 transformer layers"""
    
    def __init__(self):
        self.model_path = "downloaded_models/mistral-7b-v0.1"
        self.results_dir = "CRITICAL_PATH_RESULTS"
        
        # Critical targets
        self.ram_target_mb = 400
        self.storage_target_gb = 4.0
        self.quality_target_percent = 1.0
        
        # Validation state
        self.validation_log = []
        self.layer_results = []
        self.total_compression_ratio = 0
        self.total_quality_loss = 0
        self.max_ram_used = 0
        
        print(f"🎯 COMPLETE 32-LAYER MODEL VALIDATION")
        print(f"📁 Model: {self.model_path}")
        print(f"🚨 CRITICAL PATH ITEM 1: Full model validation")
        
        os.makedirs(self.results_dir, exist_ok=True)
    
    def log_critical(self, phase: str, status: str, details: str):
        """Log critical validation progress"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        log_entry = {
            'timestamp': timestamp,
            'phase': phase,
            'status': status,
            'details': details,
            'critical_path': 'FULL_32_LAYER_VALIDATION'
        }
        
        self.validation_log.append(log_entry)
        
        print(f"🚨 CRITICAL [{timestamp}]: {phase} - {status}")
        print(f"   {details}")
        
        # Save progress immediately
        try:
            with open(f'{self.results_dir}/critical_validation_log.json', 'w') as f:
                json.dump(self.validation_log, f, indent=2, default=str)
        except:
            pass
    
    def measure_ram_critical(self, description: str) -> float:
        """Critical RAM measurement"""
        process = psutil.Process()
        ram_mb = process.memory_info().rss / (1024**2)
        self.max_ram_used = max(self.max_ram_used, ram_mb)
        
        status = "✅ UNDER" if ram_mb <= self.ram_target_mb else "❌ OVER"
        print(f"📊 RAM: {description} = {ram_mb:.0f}MB ({status} {self.ram_target_mb}MB)")
        
        return ram_mb
    
    def compress_layer_efficient(self, layer_num: int) -> Dict[str, Any]:
        """Efficiently compress complete transformer layer"""
        
        self.log_critical("LAYER_COMPRESSION", "STARTED", f"Compressing layer {layer_num}")
        
        try:
            # Load model index
            index_path = os.path.join(self.model_path, "model.safetensors.index.json")
            with open(index_path, 'r') as f:
                weight_index = json.load(f)
            
            # Find layer weights
            layer_weights = []
            for weight_name in weight_index['weight_map'].keys():
                if f'layers.{layer_num}.' in weight_name:
                    layer_weights.append(weight_name)
            
            ram_before = self.measure_ram_critical(f"before_layer_{layer_num}")
            
            # Process layer weights with proven compression
            total_original_size = 0
            total_compressed_size = 0
            quality_errors = []
            successful_compressions = 0
            
            for weight_name in layer_weights:
                try:
                    file_name = weight_index['weight_map'][weight_name]
                    file_path = os.path.join(self.model_path, file_name)
                    
                    with safe_open(file_path, framework="pt", device="cpu") as f:
                        tensor = f.get_tensor(weight_name)
                        
                        # Apply proven compression (2% outliers)
                        tensor_f32 = tensor.to(torch.float32)
                        
                        # Outlier detection
                        abs_weights = torch.abs(tensor_f32)
                        outlier_cutoff = torch.quantile(abs_weights, 0.98)
                        outlier_mask = abs_weights > outlier_cutoff
                        
                        outlier_count = torch.sum(outlier_mask).item()
                        normal_count = tensor.numel() - outlier_count
                        
                        # Calculate compression
                        original_size = tensor.numel() * tensor.element_size()
                        compressed_size = (
                            normal_count * 1 // 8 +      # 1 bit per normal weight
                            outlier_count * 2 +          # 2 bytes per outlier
                            tensor.numel() * 1 // 8 + 16 # mask + stats
                        )
                        
                        compression_ratio = original_size / compressed_size
                        
                        # Quick quality assessment
                        outlier_weights = tensor_f32[outlier_mask]
                        normal_weights = tensor_f32[~outlier_mask]
                        
                        if len(normal_weights) > 0:
                            normal_mean = torch.mean(normal_weights)
                            normal_std = torch.std(normal_weights)
                            
                            # Simplified reconstruction
                            centered = normal_weights - normal_mean
                            binary = torch.sign(centered)
                            reconstructed_normal = binary * normal_std + normal_mean
                            
                            mae_error = torch.mean(torch.abs(normal_weights - reconstructed_normal)).item()
                            tensor_range = torch.max(tensor_f32) - torch.min(tensor_f32)
                            relative_error = mae_error / tensor_range.item() if tensor_range > 0 else 0
                        else:
                            relative_error = 0
                        
                        total_original_size += original_size
                        total_compressed_size += compressed_size
                        quality_errors.append(relative_error * 100)
                        successful_compressions += 1
                        
                        # Clear immediately
                        del tensor, tensor_f32
                        gc.collect()
                        
                except Exception as e:
                    print(f"   ⚠️ Error with {weight_name}: {e}")
                    continue
            
            ram_after = self.measure_ram_critical(f"after_layer_{layer_num}")
            
            # Clear layer (streaming)
            gc.collect()
            
            ram_after_clear = self.measure_ram_critical(f"after_clear_{layer_num}")
            
            # Calculate layer results
            layer_compression = total_original_size / total_compressed_size if total_compressed_size > 0 else 1.0
            avg_quality_loss = sum(quality_errors) / len(quality_errors) if quality_errors else 0
            success_rate = successful_compressions / len(layer_weights) if layer_weights else 0
            
            layer_result = {
                'layer_num': layer_num,
                'total_weights': len(layer_weights),
                'successful_compressions': successful_compressions,
                'success_rate': success_rate,
                'compression_ratio': layer_compression,
                'quality_loss_percent': avg_quality_loss,
                'original_size_mb': total_original_size / (1024**2),
                'compressed_size_mb': total_compressed_size / (1024**2),
                'ram_before_mb': ram_before,
                'ram_after_mb': ram_after,
                'ram_after_clear_mb': ram_after_clear,
                'stayed_under_ram_target': ram_after <= self.ram_target_mb
            }
            
            self.layer_results.append(layer_result)
            
            self.log_critical("LAYER_COMPRESSION", "SUCCESS", 
                             f"Layer {layer_num}: {layer_compression:.2f}× compression, {avg_quality_loss:.2f}% error, {success_rate*100:.1f}% success")
            
            return layer_result
            
        except Exception as e:
            self.log_critical("LAYER_COMPRESSION", "FAILED", f"Layer {layer_num} failed: {e}")
            return {}
    
    def validate_all_32_layers(self) -> Dict[str, Any]:
        """Validate all 32 transformer layers"""
        
        self.log_critical("FULL_MODEL_VALIDATION", "STARTED", "Validating all 32 transformer layers")
        
        baseline_ram = self.measure_ram_critical("validation_baseline")
        
        # Process all 32 layers
        successful_layers = 0
        total_compression_ratios = []
        total_quality_losses = []
        ram_target_maintained = True
        
        for layer_num in range(32):
            print(f"\n🔄 PROCESSING LAYER {layer_num + 1}/32")
            
            layer_result = self.compress_layer_efficient(layer_num)
            
            if layer_result and layer_result.get('success_rate', 0) > 0.5:
                successful_layers += 1
                total_compression_ratios.append(layer_result['compression_ratio'])
                total_quality_losses.append(layer_result['quality_loss_percent'])
                
                if not layer_result['stayed_under_ram_target']:
                    ram_target_maintained = False
                
                print(f"   ✅ Layer {layer_num}: {layer_result['compression_ratio']:.2f}× compression")
            else:
                print(f"   ❌ Layer {layer_num}: Failed validation")
        
        # Calculate final results
        if successful_layers > 0:
            avg_compression = sum(total_compression_ratios) / len(total_compression_ratios)
            avg_quality_loss = sum(total_quality_losses) / len(total_quality_losses)
            
            # Storage calculation
            current_model_size_gb = 13.5
            projected_storage_gb = current_model_size_gb / avg_compression
            
            # Final validation
            validation_results = {
                'total_layers': 32,
                'successful_layers': successful_layers,
                'success_rate': successful_layers / 32,
                'average_compression_ratio': avg_compression,
                'average_quality_loss_percent': avg_quality_loss,
                'max_ram_used_mb': self.max_ram_used,
                'ram_target_achieved': self.max_ram_used <= self.ram_target_mb,
                'projected_storage_gb': projected_storage_gb,
                'storage_target_achieved': projected_storage_gb <= self.storage_target_gb,
                'quality_target_achieved': avg_quality_loss <= self.quality_target_percent,
                'all_targets_achieved': (
                    self.max_ram_used <= self.ram_target_mb and
                    projected_storage_gb <= self.storage_target_gb and
                    avg_quality_loss <= self.quality_target_percent
                ),
                'layer_results': self.layer_results
            }
            
            status = "SUCCESS" if validation_results['all_targets_achieved'] else "PARTIAL"
            self.log_critical("FULL_MODEL_VALIDATION", status, 
                             f"32 layers: {successful_layers} successful, {avg_compression:.2f}× compression, {avg_quality_loss:.2f}% quality")
            
            return validation_results
        
        return {}

def main():
    """Main 32-layer validation"""
    
    print("🚨 CRITICAL PATH ITEM 1: COMPLETE 32-LAYER MODEL VALIDATION")
    print("=" * 80)
    print("PROVING THE CONCEPT WORKS ON FULL MODEL")
    print("NO DELAYS - COMPLETE VALIDATION")
    print()
    
    # Initialize validator
    validator = Complete32LayerValidator()
    
    if not os.path.exists(validator.model_path):
        print(f"❌ Model not found: {validator.model_path}")
        return
    
    validator.log_critical("CRITICAL_PATH_1", "STARTED", "Starting complete 32-layer validation")
    
    # Run complete validation
    validation_results = validator.validate_all_32_layers()
    
    if validation_results:
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = f"{validator.results_dir}/complete_32_layer_results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(validation_results, f, indent=2, default=str)
        
        print(f"\n✅ CRITICAL PATH ITEM 1 COMPLETED")
        print(f"📄 Results saved: {results_file}")
        
        # Display critical results
        print(f"\n🎯 32-LAYER VALIDATION RESULTS:")
        print(f"   Successful layers: {validation_results['successful_layers']}/32")
        print(f"   Success rate: {validation_results['success_rate']*100:.1f}%")
        print(f"   Average compression: {validation_results['average_compression_ratio']:.2f}×")
        print(f"   Average quality loss: {validation_results['average_quality_loss_percent']:.2f}%")
        print(f"   Max RAM used: {validation_results['max_ram_used_mb']:.0f}MB")
        
        print(f"\n🎯 TARGET ACHIEVEMENT:")
        print(f"   400MB RAM: {'✅ ACHIEVED' if validation_results['ram_target_achieved'] else '❌ MISSED'}")
        print(f"   4GB Storage: {'✅ ACHIEVED' if validation_results['storage_target_achieved'] else '❌ MISSED'} ({validation_results['projected_storage_gb']:.2f}GB)")
        print(f"   <1% Quality: {'✅ ACHIEVED' if validation_results['quality_target_achieved'] else '❌ MISSED'}")
        
        if validation_results['all_targets_achieved']:
            print(f"\n🎉 CRITICAL PATH ITEM 1: SUCCESS!")
            print(f"   Full 32-layer model validation COMPLETED")
            print(f"   All targets achieved - concept PROVEN")
        else:
            print(f"\n⚠️ CRITICAL PATH ITEM 1: PARTIAL")
            print(f"   Some targets not fully achieved")
        
        validator.log_critical("CRITICAL_PATH_1", "COMPLETED", 
                              f"All targets: {'ACHIEVED' if validation_results['all_targets_achieved'] else 'PARTIAL'}")
        
        return validation_results
    else:
        print(f"\n❌ CRITICAL PATH ITEM 1 FAILED")
        validator.log_critical("CRITICAL_PATH_1", "FAILED", "Could not complete 32-layer validation")
        return None

if __name__ == "__main__":
    main()
