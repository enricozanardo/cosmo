import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from loguru import logger
import os
import psutil

class HardwareOptimizer:
    """Hardware-specific optimizations for MicroO1"""
    
    def __init__(self, device_type: str = 'cuda'):
        logger.info(f"Initializing hardware optimizations for {device_type}")
        self.device_type = device_type
        
        # Set up CPU optimizations
        if device_type == 'cpu':
            # Get CPU info
            self.num_cores = psutil.cpu_count(logical=False)
            self.num_threads = psutil.cpu_count(logical=True)
            
            # Set thread settings only if not already set
            try:
                if torch.get_num_threads() == 1:
                    torch.set_num_threads(self.num_threads)
            except Exception as e:
                logger.warning(f"Could not set number of threads: {e}")
            
            logger.debug(f"CPU optimization: Using {torch.get_num_threads()} threads")
        
        # Set up GPU optimizations
        elif device_type == 'cuda' and torch.cuda.is_available():
            # Get GPU info
            self.gpu_name = torch.cuda.get_device_name()
            self.gpu_memory = torch.cuda.get_device_properties(0).total_memory
            
            # Enable cudnn benchmarking
            torch.backends.cudnn.benchmark = True
            
            # Create CUDA streams
            self.main_stream = torch.cuda.Stream()
            self.memory_stream = torch.cuda.Stream()
            
            logger.debug(f"GPU optimization: Using {self.gpu_name}")
        
        # Set up MPS optimizations for Apple Silicon
        elif torch.backends.mps.is_available():
            logger.info("Using MPS (Metal Performance Shaders) for Apple Silicon")
            self.device = torch.device("mps")
            self.device_type = "mps"
        
        logger.debug("Hardware optimizations initialized")
    
    def optimize_model(self, model: nn.Module) -> nn.Module:
        """Apply hardware-specific optimizations to model"""
        if self.device_type == 'cpu':
            return self._optimize_for_cpu(model)
        elif self.device_type == 'cuda':
            return self._optimize_for_gpu(model)
        return model
    
    def _optimize_for_cpu(self, model: nn.Module) -> nn.Module:
        """Apply CPU-specific optimizations"""
        try:
            # Fuse operations where possible
            model = torch.jit.script(model)
            
            # Enable memory sharing if supported
            if hasattr(model, 'share_memory'):
                model.share_memory()
            
            return model
        except Exception as e:
            logger.warning(f"CPU optimization failed: {e}")
            return model
    
    def _optimize_for_gpu(self, model: nn.Module) -> nn.Module:
        """Apply GPU-specific optimizations"""
        if self.device_type == "cuda":
            # Move model to GPU
            model = model.cuda()
            
            # Use mixed precision where appropriate
            if hasattr(model, 'mixed_precision'):
                logger.debug("Using existing mixed precision settings")
            else:
                from torch.cuda.amp import autocast
                model = torch.cuda.amp.autocast()(model)
            
            # Optimize memory allocation
            torch.cuda.empty_cache()
        
        elif self.device_type == "mps":
            # Move model to MPS device
            model = model.to(self.device)
            logger.debug("Model moved to MPS device")
        
        return model
    
    def optimize_dataloader(self, dataloader: torch.utils.data.DataLoader) -> torch.utils.data.DataLoader:
        """Apply hardware-specific optimizations to dataloader"""
        # Create a new dataloader with optimized settings
        optimized_settings = {
            'batch_size': dataloader.batch_size,
            'shuffle': False,  # Default to False, can be overridden
            'num_workers': 0,  # Start with safe default
            'pin_memory': False  # Start with safe default
        }
        
        try:
            # Get original settings if available
            if hasattr(dataloader, '_iterator') and dataloader._iterator is not None:
                optimized_settings['shuffle'] = dataloader._iterator.shuffle
            
            if self.device_type == 'cpu':
                # Set number of workers based on CPU cores
                optimized_settings['num_workers'] = max(1, self.num_cores - 1)
            
            elif self.device_type in ['cuda', 'mps']:
                # Enable asynchronous data loading
                optimized_settings['num_workers'] = 2
                optimized_settings['pin_memory'] = True
            
            # Create new dataloader with optimized settings
            optimized_loader = torch.utils.data.DataLoader(
                dataloader.dataset,
                **optimized_settings
            )
            
            return optimized_loader
        
        except Exception as e:
            logger.warning(f"Dataloader optimization failed: {e}")
            return dataloader
    
    def estimate_memory_per_sample(self, model: nn.Module, input_shape: tuple) -> int:
        """Estimate memory usage per sample"""
        try:
            # Create dummy input
            dummy_input = torch.randn(1, *input_shape)
            
            if self.device_type == 'cuda':
                # Track CUDA memory usage
                torch.cuda.reset_peak_memory_stats()
                with torch.cuda.amp.autocast():
                    _ = model(dummy_input)
                return torch.cuda.max_memory_allocated()
            
            elif self.device_type == 'mps':
                # For MPS, use a rough estimation
                total_params = sum(p.numel() * p.element_size() for p in model.parameters())
                activation_estimate = total_params * 2  # Rough estimate for activations
                return total_params + activation_estimate
            
            else:
                # For CPU, estimate based on model parameters
                total_params = sum(p.numel() * p.element_size() for p in model.parameters())
                return total_params * 3  # Rough estimate including gradients
        
        except Exception as e:
            logger.warning(f"Memory estimation failed: {e}")
            return 1024 * 1024  # Return 1MB as a safe default
    
    def get_optimal_batch_size(self, model: nn.Module, input_shape: tuple) -> int:
        """Calculate optimal batch size based on hardware"""
        try:
            if self.device_type == 'cuda':
                # Get GPU memory info
                gpu_mem = torch.cuda.get_device_properties(0).total_memory
                mem_per_sample = self.estimate_memory_per_sample(model, input_shape)
                
                # Leave some memory for gradients and buffers
                optimal_batch = int(gpu_mem * 0.7 / mem_per_sample)
                return max(1, min(optimal_batch, 128))
            
            elif self.device_type == 'mps':
                # For MPS, use a conservative batch size
                return 32
            
            else:
                # For CPU, use available memory
                available_mem = psutil.virtual_memory().available
                mem_per_sample = self.estimate_memory_per_sample(model, input_shape)
                
                optimal_batch = int(available_mem * 0.5 / mem_per_sample)
                return max(1, min(optimal_batch, 64))
        
        except Exception as e:
            logger.warning(f"Batch size calculation failed: {e}")
            return 16  # Return a safe default batch size
    
    def __enter__(self):
        """Context manager entry"""
        if self.device_type == 'cuda':
            torch.cuda.stream(self.main_stream)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        if self.device_type == 'cuda':
            # Synchronize streams
            torch.cuda.current_stream().synchronize()

def estimate_memory_per_sample(model: nn.Module, input_shape: tuple) -> int:
    """Estimate memory usage per sample"""
    # Create dummy input
    dummy_input = torch.randn(1, *input_shape)
    
    # Track memory usage
    torch.cuda.reset_peak_memory_stats()
    with torch.cuda.amp.autocast():
        _ = model(dummy_input)
    
    return torch.cuda.max_memory_allocated() 