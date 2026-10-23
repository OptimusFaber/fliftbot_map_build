"""
Fourier-based Circular Marker Decoding System for Autonomous Forklift Navigation

This module implements a complete GPU-accelerated circular marker decoding system for
autonomous forklift robots. The system processes camera crops from YOLO bounding box
detections to identify Hamming-encoded circular markers mounted on warehouse ceilings.
These markers contain unique identifiers that enable precise indoor positioning and navigation.

The system uses Fourier-based correlation techniques to decode binary patterns embedded
in circular markers, providing robust identification under varying lighting conditions
and camera angles typical in warehouse environments.

Architecture Overview:
    1. ImagePreprocessor: Normalizes and prepares YOLO-cropped marker images from top camera
    2. Image2SignalDecoder: Extracts circular signal patterns using optimized convolution
    3. SignalDecoder: Matches signals against database using FFT correlation
    4. FourierDecoder: Main system with spatial filtering for navigation context

Key Features:
    - GPU-accelerated processing optimized for real-time forklift operation
    - Batch processing of multiple marker crops from single camera frame
    - Spatial filtering using known warehouse layout for improved accuracy
    - Automatic utility file generation with intelligent caching
    - Robust to lighting variations and perspective distortion

Usage for Forklift Navigation System:
    ```python
    # Initialize once at startup
    decoder = FourierDecoder.from_labels2coords("warehouse_layout.pickle")
    
    # Process YOLO detections in real-time navigation loop
    marker_crops = yolo_detector.get_marker_crops(camera_frame)  # List of crop images
    marker_ids, confidence_scores = decoder.predict_with_labels(marker_crops)
    
    # marker_ids is now List[List[int]] - each sublist contains valid marker IDs for one image
    # confidence_scores is List[List[float]] - corresponding confidence values
    for i, (ids, scores) in enumerate(zip(marker_ids, confidence_scores)):
        if ids:  # Check if any markers were detected
            best_id = ids[0]  # Highest confidence marker
            best_score = scores[0]
            print(f"Image {i}: Best marker ID {best_id} with confidence {best_score:.3f}")
    ```

Performance Characteristics:
    - ~0.6ms per marker crop on modern GPU
    - Scales efficiently with batch size for multiple simultaneous detections
    - Memory usage: ~6-20MB GPU memory for typical configurations
    - Spatial filtering adds <0.3ms overhead for position-aware navigation

Author: Matalygin Andrew
"""

from __future__ import annotations
import os
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import cv2
import pickle
import glob
import json
import shutil
import time
from tqdm import tqdm
import hashlib
import sys
import subprocess
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union, Any
# import line_profiler
# from line_profiler_pycharm import profile


# Check for CUDA availability and configure device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DEVICE = "cpu"  # Uncomment to force CPU usage for debugging
print(f"Using device: {DEVICE}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name()}")
    print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


def load_masks(mask_dir: str, mask_prefix: str, expected_count: int = 32) -> torch.Tensor:
    """
    Load mask images from directory, sorting by numerical index extracted from filename.
    
    This function loads binary mask images used for Fourier-based marker detection.
    The masks are sorted numerically based on their filename indices and stacked
    into a single tensor for efficient batch processing.
    
    Args:
        mask_dir: Directory containing mask files
        mask_prefix: Prefix of mask files (e.g., "in_mask", "out_mask")
        expected_count: Expected number of mask files to load
    
    Returns:
        torch.Tensor: Stacked mask tensors with shape [expected_count, 1, 128, 128]
        
    Raises:
        AssertionError: If fewer than expected_count mask files are found
    """
    # Find all mask files with the given prefix
    mask_files = [
        f for f in os.listdir(mask_dir)
        if f.endswith('.png') and f.startswith(mask_prefix + '_')
    ]
    
    # Extract index from filename and create (index, filepath) pairs
    indexed_files = []
    for filename in mask_files:
        try:
            # Extract index from filename: "prefix_index.png" -> index
            index_str = filename[len(mask_prefix) + 1:-(len('.png'))]  # Remove prefix_, and .png
            index = int(index_str)
            filepath = os.path.join(mask_dir, filename)
            indexed_files.append((index, filepath))
        except ValueError:
            print(f"Warning: Could not extract index from filename '{filename}', skipping")
            continue
    
    # Sort by index to ensure consistent ordering
    indexed_files.sort(key=lambda x: x[0])
    
    # Validate we have enough files
    assert len(indexed_files) >= expected_count, \
        f"Found {len(indexed_files)} {mask_prefix} masks, expected {expected_count}"
    
    # Load masks in index order and normalize
    masks = []
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ])
    
    for i, (index, path) in enumerate(indexed_files[:expected_count]):
        img = Image.open(path)
        tensor = transform(img)  # Shape: [1, 128, 128]
        tensor /= tensor.sum()  # Normalize to unit sum for correlation
        masks.append(tensor)
    
    return torch.stack(masks)  # Shape: [expected_count, 1, 128, 128]


def load_labels2coords_mapping(labels2coords_path: Union[str, Path]) -> Dict[int, Tuple[float, float]]:
    """Load marker code -> (x, y) from JSON or pickle labels2coords files."""
    path = Path(labels2coords_path)
    if path.suffix.lower() == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        with open(path, "rb") as f:
            data = pickle.load(f)

    cleaned: Dict[int, Tuple[float, float]] = {}
    for key, value in data.items():
        try:
            code = int(key)
            if isinstance(value, dict):
                cleaned[code] = (float(value["x"]), float(value["y"]))
            else:
                cleaned[code] = (float(value[0]), float(value[1]))
        except (TypeError, ValueError, KeyError, IndexError):
            continue
    if not cleaned:
        raise ValueError(f"No valid labels2coords entries in {path}")
    return cleaned


def get_file_hash(file_path: str) -> Optional[str]:
    """
    Generate MD5 hash of a file to detect changes for caching purposes.
    
    Args:
        file_path: Path to the file to hash
        
    Returns:
        Optional[str]: MD5 hash as hexadecimal string, or None if file not found
    """
    hash_md5 = hashlib.md5()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except FileNotFoundError:
        return None


def setup_tmp_directory_and_files(labels2coords_path: str, decoder_dir: str, 
                                  custom_tmp_dir: Optional[str] = None) -> Tuple[str, str, str, str, str]:
    """
    Set up temporary directory and generate required files if needed.
    
    This function creates a temporary directory structure and generates the necessary
    files for the Fourier decoder system, including warehouse weights, index mappings,
    and bit masks. Uses intelligent caching to avoid regenerating files when inputs
    haven't changed.
    
    Args:
        labels2coords_path: Path to labels2coords pickle file containing marker coordinates
        decoder_dir: Directory where decoder.py is located
        custom_tmp_dir: Optional custom directory for temporary files. If provided, 
                       'FourierDecoderTmp' folder will be created there. If None, 
                       defaults to 'tmp' folder in decoder_dir.
        
    Returns:
        Tuple containing:
            - tmp_dir: Path to temporary directory
            - warehouse_weights_path: Path to warehouse weights file
            - index2label_path: Path to index-to-label mapping file
            - label2index_path: Path to label-to-index mapping file
            - bit_masks_dir: Path to bit masks directory
            
    Raises:
        FileNotFoundError: If labels2coords file is not found
        ImportError: If required generator modules are not found
    """
    # Create tmp directory structure
    if custom_tmp_dir is not None:
        tmp_dir = os.path.join(custom_tmp_dir, "FourierDecoderTmp")
    else:
        tmp_dir = os.path.join(decoder_dir, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    
    # Create bit_masks subdirectory
    bit_masks_dir = os.path.join(tmp_dir, "bit_masks")
    os.makedirs(bit_masks_dir, exist_ok=True)
    
    # Define output file paths
    warehouse_weights_path = os.path.join(tmp_dir, "warehouse_weights.pickle")
    index2label_path = os.path.join(tmp_dir, "index2label.pickle")
    label2index_path = os.path.join(tmp_dir, "label2index.pickle")
    cache_info_path = os.path.join(tmp_dir, "cache_info.pickle")
    
    # Calculate hash of input file for cache validation
    labels2coords_hash = get_file_hash(labels2coords_path)
    if labels2coords_hash is None:
        raise FileNotFoundError(f"Labels2coords file not found: {labels2coords_path}")
    
    # Check if cache is valid
    cache_valid = False
    try:
        with open(cache_info_path, 'rb') as f:
            cache_info = pickle.load(f)
        if (cache_info.get('labels2coords_hash') == labels2coords_hash and
            os.path.exists(warehouse_weights_path) and
            os.path.exists(index2label_path) and
            os.path.exists(label2index_path) and
            len(os.listdir(bit_masks_dir)) >= 64):  # Should have at least 64 mask files
            cache_valid = True
            print("Cache is valid, skipping file generation.")
    except (FileNotFoundError, pickle.PickleError):
        cache_valid = False
    
    if not cache_valid:
        print("Cache invalid or missing, generating files...")
        
        # Generate warehouse weights using external generator
        print("Generating warehouse weights...")
        try:
            # Import and run warehouse weights generator
            sys.path.append(decoder_dir)
            from warehouse_weights_generator import generate_warehouse_weights
            
            generate_warehouse_weights(
                labels2coords_file=labels2coords_path,
                output_file=warehouse_weights_path,
                index2label_file=index2label_path,
                label2index_file=label2index_path,
                bits=16,
                samples=32
            )
            print(f"Warehouse weights generated at: {warehouse_weights_path}")
            
        except ImportError as e:
            print(f"Error importing warehouse_weights_generator: {e}")
            print("Make sure warehouse_weights_generator.py is in the same directory as decoder.py")
            raise
        
        # Generate bit masks using external generator
        print("Generating bit masks...")
        try:
            from bit_mask_generator import generate_bit_mask
            
            # Clear existing masks
            for file in os.listdir(bit_masks_dir):
                if file.endswith('.png'):
                    os.remove(os.path.join(bit_masks_dir, file))
            
            total_bits = 16
            samples = 32
            
            # Temporarily change to the decoder directory to run bit mask generator
            original_cwd = os.getcwd()
            os.chdir(decoder_dir)
            
            try:
                # Create a temporary bit_masks directory in decoder dir
                temp_masks_dir = "bit_masks"
                os.makedirs(temp_masks_dir, exist_ok=True)
                
                # Generate inner and outer masks for each sample position
                for i in range(samples):
                    generate_bit_mask(position_index=i, total_bits=total_bits, 
                                    samples=samples, radius=64, outer=True)
                    generate_bit_mask(position_index=i, total_bits=total_bits, 
                                    samples=samples, radius=64, outer=False)
                
                # Move generated masks to tmp directory
                for file in os.listdir(temp_masks_dir):
                    if file.endswith('.png'):
                        src = os.path.join(temp_masks_dir, file)
                        dst = os.path.join(bit_masks_dir, file)
                        os.rename(src, dst)
                
                # Clean up temporary directory
                shutil.rmtree(temp_masks_dir, ignore_errors=True)
                
            finally:
                os.chdir(original_cwd)
            
            print(f"Bit masks generated at: {bit_masks_dir}")
            
        except ImportError as e:
            print(f"Error importing bit_mask_generator: {e}")
            print("Make sure bit_mask_generator.py is in the same directory as decoder.py")
            raise
        
        # Save cache info for future validation
        cache_info = {
            'labels2coords_hash': labels2coords_hash,
            'labels2coords_path': labels2coords_path,
            'generation_time': __import__('time').time()
        }
        with open(cache_info_path, 'wb') as f:
            pickle.dump(cache_info, f)
        
        print("File generation complete and cache updated.")
    
    return tmp_dir, warehouse_weights_path, index2label_path, label2index_path, bit_masks_dir


class Image2SignalDecoder(nn.Module):
    """
    Neural network module for converting preprocessed marker images to signal representations.
    
    This class implements a convolutional approach to extract circular signal patterns from
    marker images using pre-computed bit masks. The masks are designed to sample the inner
    and outer circular regions of Fourier-encoded markers, producing normalized signal
    vectors suitable for correlation-based matching.
    
    Attributes:
        conv: Convolutional layer with fixed weights from loaded masks
    """
    
    def __init__(self, mask_dir: str, mask_count_per_type: int = 32) -> None:
        """
        Initialize the Image2SignalDecoder with bit masks.
        
        Args:
            mask_dir: Directory containing bit mask files
            mask_count_per_type: Number of masks per type (inner/outer), defaults to 32
        """
        super().__init__()
        
        # Load inner and outer circular masks
        inner_masks = load_masks(mask_dir, "in_mask", mask_count_per_type)
        outer_masks = load_masks(mask_dir, "out_mask", mask_count_per_type)
        
        # Combine masks: first 32 channels for inner, next 32 for outer
        conv_weight = torch.cat([inner_masks, outer_masks], dim=0)
        
        # Create convolution layer with fixed weights (no training required)
        self.conv = nn.Conv2d(1, 64, kernel_size=(128, 128), bias=False)
        self.conv.weight = nn.Parameter(conv_weight, requires_grad=False)
        
        # Move to configured device (GPU if available)
        self.to(DEVICE)

    # @profile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert input images to normalized signal vectors.
        
        Args:
            x: Input tensor of shape [batch_size, 1, 128, 128]
            
        Returns:
            torch.Tensor: Signal vectors of shape [batch_size, 2, 32] where:
                - [:, 0, :] represents inner circle samples
                - [:, 1, :] represents outer circle samples
                Each signal is normalized to [0, 1] range
        """
        # Ensure input is on the correct device
        x = x.to(DEVICE)
        
        # Apply convolution to extract signals from both rings
        x = self.conv(x)  # Shape: [batch, 64, 1, 1]
        x = x.squeeze(-1).squeeze(-1)  # Shape: [batch, 64]
        x = x.view(-1, 2, 32)  # Shape: [batch, 2, 32] - separate inner/outer rings
        
        # Normalize each ring independently to [0, 1] range
        # This ensures consistent signal amplitude regardless of image brightness
        x_min = x.min(dim=-1, keepdim=True)[0]  # Shape: [batch, 2, 1]
        x_max = x.max(dim=-1, keepdim=True)[0]  # Shape: [batch, 2, 1]
        x_range = x_max - x_min
        
        # Avoid division by zero for flat signals
        x_range = torch.where(x_range == 0, torch.ones_like(x_range), x_range)
        x = (x - x_min) / x_range  # Scale to [0, 1]
        
        return x


class ImagePreprocessor(nn.Module):
    """
    Neural network module for preprocessing raw marker images.
    
    This class handles the conversion of raw marker crop images to standardized format
    suitable for signal extraction. It performs grayscale conversion, resizing, brightness
    normalization, and conditional inversion based on outer mask regions to optimize
    signal quality for the Fourier-based detection system.
    
    Attributes:
        combined_outer_mask: Binary mask indicating outer ring regions for conditional inversion
    """
    
    def __init__(self, mask_dir: str, mask_count_per_type: int = 32) -> None:
        """
        Initialize the ImagePreprocessor with outer masks for conditional inversion.
        
        Args:
            mask_dir: Directory containing mask files
            mask_count_per_type: Number of masks per type, defaults to 32
        """
        super().__init__()
        
        # Load outer masks and create combined mask for inversion
        outer_masks = load_masks(mask_dir, "out_mask", mask_count_per_type)
        # Create binary mask: any pixel that's part of any outer mask
        self.combined_outer_mask = (torch.sum(outer_masks, dim=0) > 0).float().to(DEVICE)
        
        # Move to configured device
        self.to(DEVICE)


    # @profile
    def _preprocess_single(self, img_np: np.ndarray) -> torch.Tensor:
        """
        Preprocess a single image crop.
        
        This method performs the complete preprocessing pipeline:
        1. Convert to grayscale if needed
        2. Resize to 128x128 pixels
        3. Normalize brightness using percentile-based scaling
        4. Apply conditional inversion in outer mask regions
        
        Args:
            img_np: Input image as numpy array (can be color or grayscale)
            
        Returns:
            torch.Tensor: Preprocessed image tensor of shape [1, 1, 128, 128]
        """
        # Convert to grayscale if the input is a color image
        if img_np.ndim == 3 and img_np.shape[2] == 3:
            img_gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = img_np
        
        # Resize to standard 128x128 size using area interpolation for quality
        img_resized = cv2.resize(img_gray, (128, 128), interpolation=cv2.INTER_AREA)
        
        # Robust brightness normalization using percentiles to handle varying lighting
        p10, p90 = np.percentile(img_resized, [10, 90])
        img_scaled = np.clip(img_resized, p10, p90)
        img_scaled = (img_scaled - p10) / (p90 - p10 + 1e-8)  # Add epsilon to avoid division by zero
        
        # Convert to tensor and move to device
        img_tensor = torch.from_numpy(img_scaled).float().unsqueeze(0)
        img_tensor = img_tensor.to(self.combined_outer_mask.device)
        
        # Apply conditional inversion: invert pixels in outer mask regions
        # This optimizes contrast for better signal extraction
        inverted = img_tensor * (1 - self.combined_outer_mask) + (1 - img_tensor) * self.combined_outer_mask
        
        return inverted.unsqueeze(0)  # Add batch dimension: [1, 1, 128, 128]

    # @profile
    def forward(self, batch_np: List[np.ndarray]) -> torch.Tensor:
        """
        Preprocess a batch of images.
        
        Args:
            batch_np: List of numpy arrays representing image crops
            
        Returns:
            torch.Tensor: Batch of preprocessed images with shape [batch_size, 1, 128, 128]
        """
        processed_batch = []
        for img_np in batch_np:
            processed = self._preprocess_single(img_np)
            processed_batch.append(processed)
        return torch.cat(processed_batch, dim=0)


class SignalDecoder(nn.Module):
    """
    Neural network module for matching extracted signals against a database of known markers.
    
    This class implements the core correlation-based matching algorithm using normalized
    circular cross-correlation in the frequency domain. It compares both the outer ring
    marker patterns and inner ring start position indicators to identify the most likely
    marker matches from a precomputed database.
    
    Attributes:
        num_markers: Total number of markers in the database
        top_n: Number of top matches to return
        marker_matrix: Precomputed outer ring signal patterns for all markers
        start_pos_matrix: Precomputed inner ring start position patterns for all markers
    """
    
    def __init__(self, warehouse_weights_path: str, top_n: int = 3) -> None:
        """
        Initialize the SignalDecoder with precomputed warehouse weights.
        
        Args:
            warehouse_weights_path: Path to .npy/.pickle file containing marker signal patterns
            top_n: Number of top matches to return for each input signal
        """
        super().__init__()
        
        # Load precomputed marker signal patterns
        weights_np = np.load(warehouse_weights_path, allow_pickle=True)
        self.num_markers = weights_np.shape[0]
        self.top_n = top_n
        
        # Register signal patterns as buffers (automatically moved to device)
        # weights_np[:, 0, :] = outer ring patterns, weights_np[:, 1, :] = inner ring start positions
        self.register_buffer('marker_matrix', torch.from_numpy(weights_np[:, 0, :]).float())
        self.register_buffer('start_pos_matrix', torch.from_numpy(weights_np[:, 1, :]).float())
        
        # Move to configured device
        self.to(DEVICE)
        
    def compute_normalized_circular_correlation_matrix(self, observed: torch.Tensor, 
                                                     db_signals: torch.Tensor) -> torch.Tensor:
        """
        Compute normalized circular cross-correlation between observed and database signals.
        
        This method efficiently computes correlations using FFT-based convolution in the
        frequency domain. The correlation is normalized to account for signal amplitude
        variations, making the matching robust to lighting changes.
        
        Args:
            observed: Observed signal tensor of shape [batch_size, signal_length]
            db_signals: Database signals tensor of shape [num_markers, signal_length]
            
        Returns:
            torch.Tensor: Correlation matrix of shape [batch_size, num_markers, signal_length]
                         containing normalized correlation values for all circular shifts
        """
        L = observed.shape[-1]  # Signal length (typically 32)
        batch_size = observed.shape[0]
        N = db_signals.shape[0]  # Number of database signals
        
        # Zero-mean normalization for robust correlation
        observed_z = observed - observed.mean(dim=-1, keepdim=True)
        db_signals_z = db_signals - db_signals.mean(dim=-1, keepdim=True)
        
        # Compute standard deviations for normalization
        std_obs = torch.std(observed_z, dim=-1, unbiased=False)
        std_db = torch.std(db_signals_z, dim=-1, unbiased=False)
        
        # Compute FFTs for frequency domain processing
        fft_obs = torch.fft.fft(observed_z, dim=-1)
        fft_db = torch.fft.fft(db_signals_z, dim=-1)
        
        # Compute circular cross-correlation via element-wise multiplication in frequency domain
        # Using complex conjugate of database signals for correlation
        product = fft_obs.unsqueeze(1) * torch.conj(fft_db).unsqueeze(0)
        corr = torch.fft.ifft(product, dim=-1).real
        
        # Normalize correlation values to [-1, 1] range
        denominator = L * std_obs.unsqueeze(1) * std_db.unsqueeze(0)
        # Avoid division by zero for constant signals
        denominator = torch.where(denominator == 0, torch.ones_like(denominator) * 1e-10, denominator)
        normalized_corr = corr / denominator.unsqueeze(-1)
        
        return normalized_corr
    
    def find_top_picks(self, corr_matrix: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Identify the top_n markers with the highest maximum correlation values.
        
        Args:
            corr_matrix: Correlation matrix of shape [batch_size, num_markers, signal_length]
            
        Returns:
            Tuple of tensors both with shape [batch_size, top_n]:
                - top_indices: Indices of the top matching markers
                - top_scores: Maximum correlation scores for the top matches
        """
        # Find maximum correlation value for each marker across all circular shifts
        max_vals, _ = torch.max(corr_matrix, dim=-1)  # Shape: [batch_size, num_markers]
        
        # Select top_n markers with highest correlation scores
        top_scores, top_indices = torch.topk(max_vals, k=self.top_n, dim=-1, sorted=True, largest=True)
        
        return top_indices, top_scores
    
    def forward(self, signals: torch.Tensor, allowed_indices_per_image: Optional[List[Optional[List[int]]]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process decoded signals through the matching pipeline.
        
        This method combines correlation results from both outer ring patterns and
        inner ring start positions to provide robust marker identification.
        
        Args:
            signals: Signal tensor of shape [batch_size, 2, 32] from Image2SignalDecoder
                    where signals[:, 0, :] = inner ring, signals[:, 1, :] = outer ring
            
        Returns:
            Tuple of tensors with shape [batch_size, top_n]:
                - top_indices: Indices of the most likely matching markers
                - top_scores: Combined correlation scores for the matches
        """
        # Ensure signals are on the correct device
        signals = signals.to(DEVICE)
        
        # Split signals into inner and outer ring components
        outer_signals = signals[:, 1, :]  # Outer circle activations (main pattern)
        inner_signals = signals[:, 0, :]  # Inner circle activations (start position)
        
        # Compute correlations for both signal types
        correlated_matrix = self.compute_normalized_circular_correlation_matrix(
            outer_signals, self.marker_matrix)
        start_bit_matrix = self.compute_normalized_circular_correlation_matrix(
            inner_signals, self.start_pos_matrix)
        final_matrix = correlated_matrix * 0.5 + start_bit_matrix * 0.5

        if allowed_indices_per_image is not None:
            # Build mask of disallowed indices set to -inf so they are not picked
            batch_size = final_matrix.shape[0]
            # Prepare a base mask of zeros
            mask = torch.zeros_like(final_matrix[:, :, 0])  # [batch, num_markers]
            # Fill mask with -inf for disallowed markers per image
            neg_inf = torch.tensor(float('-inf'), device=final_matrix.device)
            all_indices_tensor = torch.arange(self.num_markers, device=final_matrix.device)
            for b in range(batch_size):
                allowed = allowed_indices_per_image[b] if b < len(allowed_indices_per_image) else None
                if allowed is None:
                    continue  # no restriction for this image
                allowed_set = set(allowed)
                disallowed_mask = ~torch.isin(all_indices_tensor, torch.tensor(list(allowed_set), device=final_matrix.device))
                mask[b][disallowed_mask] = neg_inf
            # Apply mask across shifts
            final_matrix = final_matrix + mask.unsqueeze(-1)

        top_indices, top_scores = self.find_top_picks(final_matrix)
        return top_indices, top_scores


class FourierDecoder(nn.Module):
    """
    Complete Fourier-based marker detection and matching system for robotic applications.
    
    This class integrates image preprocessing, signal extraction, and correlation-based
    matching to identify warehouse markers from camera images. The system is optimized
    for real-time performance with GPU acceleration and includes optional spatial filtering
    for improved accuracy in known environments.
    
    The decoder works by:
    1. Preprocessing raw marker crop images (grayscale, normalize, resize)
    2. Extracting circular signal patterns using convolution with bit masks
    3. Matching signals against a precomputed database using FFT-based correlation
    4. Optionally filtering results by spatial proximity to expected locations
    
    Attributes:
        preprocessor: Module for image preprocessing
        decoder: Module for signal extraction from images
        matcher: Module for correlation-based signal matching
        index2label: Mapping from internal indices to marker label codes
        label2index: Mapping from marker label codes to internal indices
        top_n: Number of top matches to return
        labels2coords: Dictionary mapping marker codes to (x, y) coordinates
        marker_labels: List of marker labels in index order
        marker_coords: Array of marker coordinates in index order
        marker_coords_tensor: GPU tensor of marker coordinates for spatial filtering
        _neg_one_tensor: Pre-allocated tensor for invalid index masking
        _zero_tensor: Pre-allocated tensor for invalid score masking
        _query_point_tensor: Pre-allocated tensor for spatial query points
    """
    
    def __init__(self, mask_dir: str, warehouse_weights_path: str, index2label: Dict[int, int], 
                 label2index: Dict[int, int], labels2coords_path: str, 
                 mask_count_per_type: int = 32, top_n: int = 3, tmp_dir: Optional[str] = None) -> None:
        """
        Initialize the FourierDecoder with all required components.
        
        Args:
            mask_dir: Directory containing bit mask files for signal extraction
            warehouse_weights_path: Path to precomputed marker signal patterns
            index2label: Dictionary mapping internal indices to marker label codes
            label2index: Dictionary mapping marker label codes to internal indices
            labels2coords_path: Path to pickled dictionary mapping marker codes to coordinates
            mask_count_per_type: Number of masks per type (inner/outer), defaults to 32
            top_n: Number of top matches to return per query, defaults to 3
            tmp_dir: Optional custom directory for temporary files. Used for reference only,
                    actual tmp directory location is already determined during construction.
        """
        super().__init__()
        
        # Initialize processing pipeline components
        self.preprocessor = ImagePreprocessor(mask_dir, mask_count_per_type)
        self.decoder = Image2SignalDecoder(mask_dir, mask_count_per_type)
        self.matcher = SignalDecoder(warehouse_weights_path, top_n)
        
        # Store label mapping dictionaries
        self.index2label = index2label
        self.index2label_keys = torch.tensor(list(index2label.keys()), dtype=torch.long).to(DEVICE)
        self.index2label_values = torch.tensor([self.index2label[index] for index in range(len(self.index2label))], dtype=torch.int32).to(DEVICE)
        self.label2index = label2index
        self.top_n = top_n

        if not ((isinstance(labels2coords_path, str) and labels2coords_path.endswith('.txt')) or (hasattr(labels2coords_path, 'suffix') and labels2coords_path.suffix == '.txt')):
            # Load spatial coordinate data for spatial filtering
            self.labels2coords = load_labels2coords_mapping(labels2coords_path)
            self.marker_labels = []
            self.marker_coords = []
            for i in range(len(index2label)):
                label = index2label[i]
                if label in self.labels2coords:
                    self.marker_labels.append(label)
                    self.marker_coords.append(self.labels2coords[label])
                else:
                    print(f"Warning: Label {label} not found in labels2coords")
                    # Use dummy coordinates for missing labels to maintain indexing
                    self.marker_labels.append(label)
                    self.marker_coords.append((0.0, 0.0))
            self.marker_coords = np.array(self.marker_coords)
            
            # Register coordinate data as GPU buffers for efficient spatial filtering
            self.register_buffer('marker_coords_tensor', torch.from_numpy(self.marker_coords).float())
            
            # Pre-allocate commonly used tensors to avoid repeated GPU memory allocation
            self.register_buffer('_neg_one_tensor', torch.tensor(-1, dtype=torch.long))
            self.register_buffer('_zero_tensor', torch.tensor(0.0, dtype=torch.float32))
            self.register_buffer('_query_point_tensor', torch.zeros(2, dtype=torch.float32))
        else:
            self.labels2coords = None

        
        
        # Move complete system to configured device
        self.to(DEVICE)
        
        # Report GPU memory usage for monitoring
        if torch.cuda.is_available():
            print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e6:.1f} MB")
            print(f"GPU memory cached: {torch.cuda.memory_reserved() / 1e6:.1f} MB")

    @classmethod
    def from_labels2coords(cls, labels2coords_path: str, mask_count_per_type: int = 32, 
                          top_n: int = -1, tmp_dir: Optional[str] = None) -> FourierDecoder:
        """
        Alternative constructor that automatically generates required files from labels2coords.
        
        This factory method simplifies system initialization by automatically generating
        all required auxiliary files (warehouse weights, bit masks, index mappings) from
        a single labels2coords pickle file. It uses intelligent caching to avoid
        regenerating files when inputs haven't changed.
        
        Args:
            labels2coords_path: Path to pickled dictionary mapping marker codes to coordinates
            mask_count_per_type: Number of masks per type (inner/outer), defaults to 32
            top_n: Number of top matches to return per query, defaults to -1 meaning all markers
            tmp_dir: Optional custom directory for temporary files. If provided, 
                    'FourierDecoderTmp' folder will be created there. If None, 
                    defaults to 'tmp' folder in decoder.py directory.
            
        Returns:
            FourierDecoder: Fully initialized decoder system ready for inference
            
        Raises:
            FileNotFoundError: If labels2coords file is not found
            ImportError: If required generator modules are not available
        """
        # Get directory where decoder.py is located for relative file operations
        decoder_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Setup utilities and generate/load required files
        tmp_directory, warehouse_weights_path, index2label_path, label2index_path, bit_masks_dir = \
            setup_tmp_directory_and_files(labels2coords_path, decoder_dir, tmp_dir)
        
        # Load generated mapping files
        with open(index2label_path, 'rb') as f:
            index2label = pickle.load(f)

        # get the number of markers from the index2label mapping
        num_markers = len(index2label)

        if top_n == -1:
            top_n = num_markers  # Use all markers if top_n is -1
        
        with open(label2index_path, 'rb') as f:
            label2index = pickle.load(f)
        
        # Debug information for system verification
        print(f"Loaded {len(index2label)} labels from index2label mapping")
        print(f"Sample index2label entries: {dict(list(index2label.items())[:5])}")
        print(f"Sample label2index entries: {dict(list(label2index.items())[:5])}")
        
        # Create and return the fully configured system
        return cls(
            mask_dir=bit_masks_dir,
            warehouse_weights_path=warehouse_weights_path,
            index2label=index2label,
            label2index=label2index,
            labels2coords_path=labels2coords_path,
            mask_count_per_type=mask_count_per_type,
            top_n=top_n,
            tmp_dir=tmp_dir
        )
        
    def _filter_top_picks_by_distance(self, top_indices: torch.Tensor, top_scores: torch.Tensor, 
                                    point: Optional[Tuple[float, float]], 
                                    radius: Optional[float]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Filter correlation results by spatial distance to a query point.
        
        This method provides spatial filtering to eliminate marker candidates that are
        too far from an expected location. It's optimized for minimal GPU operations
        and maximum performance in real-time applications.
        
        Args:
            top_indices: Tensor of shape [batch_size, top_n] with marker indices from correlation
            top_scores: Tensor of shape [batch_size, top_n] with correlation scores
            point: Optional tuple (x, y) representing the spatial query center
            radius: Optional search radius in coordinate units (typically centimeters)
            
        Returns:
            Tuple of tensors with same shape as inputs:
                - filtered_indices: Indices with distant markers set to -1 (invalid)
                - filtered_scores: Scores with distant markers set to 0.0 (invalid)
        """
        if point is None or radius is None:
            return top_indices, top_scores
        
        # Very fast early exit
        if (top_indices < 0).all():
            return top_indices, top_scores
        
        # Simple direct approach - no intermediate tensors
        # Clamp indices to valid range in-place
        clamped_indices = torch.clamp(top_indices, 0, len(self.marker_coords) - 1)
        
        # Get coordinates directly 
        coords = self.marker_coords_tensor[clamped_indices]  # Shape: [batch, top_n, 2]
        
        # Compute squared distances efficiently
        dx = coords[:, :, 0] - point[0]
        dy = coords[:, :, 1] - point[1]
        distances_squared = dx * dx + dy * dy
        
        # Create mask: valid indices AND within radius
        within_radius = distances_squared <= (radius * radius)
        valid_indices = top_indices >= 0
        keep_mask = valid_indices & within_radius
        
        # Apply mask directly - this is the fastest approach
        filtered_indices = torch.where(keep_mask, top_indices, self._neg_one_tensor)
        filtered_scores = torch.where(keep_mask, top_scores, self._zero_tensor)
        
        return filtered_indices, filtered_scores
    
    # @profile
    def forward(self, batch_np: List[np.ndarray], point: Optional[Tuple[float, float]] = None, 
               radius: Optional[float] = 1000,
               allowed_codes_per_image: Optional[List[Optional[List[int]]]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Main inference method for processing marker image crops.
        
        This method runs the complete detection pipeline: preprocessing, signal extraction,
        correlation matching, and optional spatial filtering. It's designed for batch
        processing to maximize GPU utilization efficiency.
        
        Args:
            batch_np: List of numpy arrays representing marker crop images
            point: Optional center point (x, y) for spatial filtering
            radius: Optional radius for spatial filtering in coordinate units
            
        Returns:
            Tuple of tensors with shape [batch_size, top_n]:
                - top_indices: Indices of most likely matching markers (-1 for invalid)
                - top_scores: Correlation scores for the matches (0.0 for invalid)
        """
        if len(batch_np) == 0:
            # Handle empty batch case gracefully
            return torch.empty(0, self.top_n, dtype=torch.long), torch.empty(0, self.top_n, dtype=torch.float32)
        # Use no_grad for inference to save GPU memory and improve performance
        with torch.no_grad():
            # Step 1: Preprocess raw images to standardized format
            images = self.preprocessor(batch_np)
            
            # Step 2: Extract circular signal patterns from preprocessed images
            signals = self.decoder(images)

            # Prepare optional allowed indices per image
            allowed_indices_per_image: Optional[List[Optional[List[int]]]] = None
            if allowed_codes_per_image is not None:
                allowed_indices_per_image = []
                for allowed_codes in allowed_codes_per_image:
                    if not allowed_codes:
                        allowed_indices_per_image.append(None)
                        continue
                    idxs: List[int] = []
                    for code in allowed_codes:
                        if code in self.label2index:
                            idxs.append(int(self.label2index[code]))
                    allowed_indices_per_image.append(idxs if idxs else None)

            # Step 3: Match signals against database using correlation (with optional constraints)
            top_indices, top_scores = self.matcher(signals, allowed_indices_per_image=allowed_indices_per_image)
            
            # Step 4: Apply optional spatial filtering for improved accuracy
            if point is not None and radius is not None:
                top_indices, top_scores = self._filter_top_picks_by_distance(
                    top_indices, top_scores, point, radius)
            
            return top_indices, top_scores
    
    def indices2labels(self, top_indices: torch.Tensor) -> torch.Tensor:
        """
        Convert tensor of internal indices to tensor of marker label codes.
        
        This method maps internal system indices to actual marker label codes using
        the precomputed index-to-label mapping. The output maintains the same tensor
        shape as the input for efficient batch processing.
        
        Args:
            top_indices: Tensor of internal indices with shape [batch_size, top_n]
            
        Returns:
            torch.Tensor: Tensor of marker label codes with same shape as input.
                         Invalid indices (-1) are preserved to maintain tensor structure.
        """
        index = top_indices.flatten()
        return self.index2label_values[index].reshape(top_indices.shape)

    # @profile
    def predict_with_labels(self, batch_np: List[np.ndarray], 
                          point: Optional[Tuple[float, float]] = None,
                          radius: Optional[float] = None) -> Tuple[List[List[int]], List[List[float]]]:
        """
        Predict marker labels with human-readable output format.
        
        This method provides a convenient interface that returns actual marker label codes
        instead of internal indices, making it easier to integrate with higher-level systems.
        The output format filters out invalid predictions and returns only valid results.
        
        Args:
            batch_np: List of numpy arrays representing marker crop images
            point: Optional center point (x, y) for spatial filtering
            radius: Optional radius for spatial filtering
            
        Returns:
            Tuple containing:
                - predicted_labels: List of lists of valid marker label codes (one list per input image).
                                  Invalid labels (-1) from spatial filtering are excluded.
                - scores: List of lists of correlation scores corresponding to valid labels.
                         Invalid scores (0.0) from spatial filtering are excluded.
        """
        with torch.no_grad():
            # Get raw prediction results
            top_indices, top_scores = self.forward(batch_np, point, radius)
            
            if self.index2label is not None:
                # Convert indices to codes and return them as ints
                top_labels = self.indices2labels(top_indices)

            top_labels[top_indices == -1] = -1

            top_labels = top_labels.cpu().numpy().tolist()
            top_scores = top_scores.cpu().numpy().tolist()
            
            # Filter out invalid predictions to provide clean output
            # Remove invalid labels (-1) that were rejected by spatial filtering
            for i in range(len(top_labels)):
                top_labels[i] = [label for label in top_labels[i] if label != -1]
            # Remove invalid scores (0.0) corresponding to rejected spatial filtering results  
            for i in range(len(top_scores)):
                top_scores[i] = [score for score in top_scores[i] if score != 0.0]

            return top_labels, top_scores

    def evaluate_accuracy(self, batch_np: List[np.ndarray], true_labels: List[int], 
                        point: Optional[Tuple[float, float]] = None,
                        radius: Optional[float] = None) -> Tuple[int, int, int]:
        """
        Evaluate prediction accuracy for a batch of images with known ground truth.
        
        This method is useful for system validation and performance monitoring in
        controlled environments where true marker labels are known.
        
        Args:
            batch_np: List of numpy arrays representing marker crop images
            true_labels: List of true marker label codes for accuracy calculation
            point: Optional center point for spatial filtering
            radius: Optional radius for spatial filtering
            
        Returns:
            Tuple of integers:
                - top1_correct: Number of correct top-1 predictions
                - top3_correct: Number of correct top-3 predictions (label in top 3)
                - total_valid: Total number of valid predictions made
        """
        with torch.no_grad():
            top_indices, top_scores = self.forward(batch_np, point, radius)
            
            top1_correct = 0
            top3_correct = 0
            total_valid = 0
            
            for i, true_label in enumerate(true_labels):
                if true_label in self.label2index:
                    true_index = self.label2index[true_label]
                    pred_indices = top_indices[i].cpu().numpy()
                    
                    # Filter out invalid indices (-1) from spatial filtering
                    valid_pred_indices = [idx for idx in pred_indices if idx >= 0]
                    
                    # Check top-1 accuracy
                    if len(valid_pred_indices) > 0 and valid_pred_indices[0] == true_index:
                        top1_correct += 1
                    
                    # Check top-3 accuracy
                    if true_index in valid_pred_indices:
                        top3_correct += 1
                    
                    total_valid += 1
            
            return top1_correct, top3_correct, total_valid

    def get_average_position(self, marker_codes: Union[List[int], torch.Tensor]) -> Optional[Tuple[float, float]]:
        """
        Calculate the average spatial position of a list of marker codes.
        
        This utility method is useful for adaptive spatial filtering, where the expected
        position for the next frame is estimated from current detections. It accepts
        either a list of marker codes or a tensor and automatically handles conversion.
        
        Args:
            marker_codes: List of marker label codes or tensor of codes to average.
                         Tensor inputs are automatically flattened and converted to list.
            
        Returns:
            Optional tuple (avg_x, avg_y) or None if no valid coordinates found
        """
        if isinstance(marker_codes, torch.Tensor):
            marker_codes = marker_codes.flatten().cpu().numpy().tolist()
        assert self.labels2coords is not None, "labels2coords must be initialized with a pickled coordinate dictionary, not just a text file."
        valid_coords = []
        for code in marker_codes:
            if code in self.labels2coords:
                valid_coords.append(self.labels2coords[code])
        
        if not valid_coords:
            return None
        
        coords_array = np.array(valid_coords)
        avg_x = np.mean(coords_array[:, 0])
        avg_y = np.mean(coords_array[:, 1])
        
        return (avg_x, avg_y)


# ================================================================================================
# TESTING AND EVALUATION FUNCTIONS
# ================================================================================================

def process_directory_auto(image_dir: str, labels2coords_path: str, output_file: Optional[str] = None,
                          point: Optional[Tuple[float, float]] = None, radius: Optional[float] = None,
                          use_spatial_filtering: bool = True, mask_count_per_type: int = 32,
                          top_n: int = 3, calculate_accuracy: bool = True, 
                          tmp_dir: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """
    Process all marker images in a directory using automatic system initialization.
    
    This function provides a comprehensive testing and evaluation framework for the
    FourierDecoder system. It processes images grouped by frame number to simulate
    real-time video processing, with support for adaptive spatial filtering and
    accuracy evaluation.
    
    The function automatically:
    1. Initializes the FourierDecoder system from labels2coords data
    2. Groups images by frame number for realistic batch processing
    3. Applies adaptive spatial filtering using detected marker positions
    4. Calculates accuracy metrics when ground truth is available
    5. Reports timing performance for real-time application assessment
    
    Features adaptive spatial filtering: if use_spatial_filtering is True, the system
    will use spatial filtering. If point is provided, it starts with that point, 
    otherwise it skips spatial filtering for the first frame and then uses the 
    average position of detected markers for subsequent frames.
    
    Args:
        image_dir: Directory containing marker crop images to process
        labels2coords_path: Path to labels2coords pickle file for system initialization
        output_file: Optional JSON file path to save detailed results
        point: Optional initial center point (x, y) for spatial filtering startup
        radius: Optional radius for spatial filtering in coordinate units (default: 1500 cm)
        use_spatial_filtering: Enable adaptive spatial filtering for improved accuracy
        mask_count_per_type: Number of bit masks per type (inner/outer), defaults to 32
        top_n: Number of top candidate matches to return per image, defaults to 3
        calculate_accuracy: Extract ground truth from filenames for accuracy calculation
        tmp_dir: Optional custom directory for temporary files. If provided, 
                'FourierDecoderTmp' folder will be created there. If None, 
                defaults to 'tmp' folder in decoder.py directory.
        
    Returns:
        Dict mapping filenames to detection results containing:
            - labels: List of detected marker label codes
            - scores: List of correlation scores for the detections
            - indices: List of internal indices for the detections
            - frame_time_ms: Processing time per marker in milliseconds
    
    Note:
        Expected filename format for accuracy calculation: "frame_X_code_Y.ext"
        where X is frame number and Y is the true marker code.
    """
    # Create system with auto-generated utilities
    system = FourierDecoder.from_labels2coords(
        labels2coords_path=labels2coords_path,
        mask_count_per_type=mask_count_per_type,
        top_n=top_n,
        tmp_dir=tmp_dir
    )
    
    results = {}
    
    # Get list of image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(image_dir, f"*{ext}")))
        image_files.extend(glob.glob(os.path.join(image_dir, f"*{ext.upper()}")))
    
    image_files.sort()
    print(f"Found {len(image_files)} images to process")
    
    # Group images by frame number
    frame_groups = {}
    for image_path in image_files:
        filename = os.path.basename(image_path)
        try:
            # Extract frame number from filename like "frame_X_code_Y.jpg"
            if '_code_' in filename and filename.startswith('frame_'):
                parts = filename.split('_')
                if len(parts) >= 3:
                    frame_num = int(parts[1])
                    code_part = parts[3].split('.')[0] if len(parts) > 3 else None
                    
                    if frame_num not in frame_groups:
                        frame_groups[frame_num] = []
                    
                    frame_groups[frame_num].append({
                        'path': image_path,
                        'filename': filename,
                        'code': int(code_part) if code_part and code_part.isdigit() else None
                    })
                else:
                    print(f"Warning: Unexpected filename format '{filename}', processing individually")
                    # Process as individual image
                    frame_groups[f"individual_{len(frame_groups)}"] = [{
                        'path': image_path,
                        'filename': filename,
                        'code': None
                    }]
            else:
                print(f"Warning: Filename '{filename}' doesn't match expected pattern, processing individually")
                # Process as individual image
                frame_groups[f"individual_{len(frame_groups)}"] = [{
                    'path': image_path,
                    'filename': filename,
                    'code': None
                }]
        except (ValueError, IndexError) as e:
            print(f"Warning: Could not parse filename '{filename}': {e}, processing individually")
            frame_groups[f"individual_{len(frame_groups)}"] = [{
                'path': image_path,
                'filename': filename,
                'code': None
            }]
    
    print(f"Grouped into {len(frame_groups)} frames")
    
    # Initialize accuracy and timing tracking
    total_predictions = 0
    top1_correct = 0
    top3_correct = 0
    total_codes_processed = 0
    total_forward_time = 0.0
    frame_times = []
    
    # Initialize adaptive spatial filtering
    current_search_point = point if use_spatial_filtering else None  # Start with provided point if filtering enabled
    adaptive_radius = radius if radius is not None else 1500  # Default 1000 cm radius
    first_frame_processed = False  # Track if we've processed the first frame
    
    print(f"Spatial filtering: {'Enabled' if use_spatial_filtering else 'Disabled'}")
    if use_spatial_filtering:
        if point is not None:
            print(f"Starting with initial search point: {point}, radius: {adaptive_radius}")
        else:
            print(f"No initial point provided - will skip filtering for first frame, then use detected markers")
    
    # Sort frames for consistent processing order
    sorted_frame_keys = sorted([k for k in frame_groups.keys() if isinstance(k, int)]) + \
                       sorted([k for k in frame_groups.keys() if not isinstance(k, int)])
    
    # Process each frame
    try:
        progress_bar = tqdm(sorted_frame_keys, desc="Processing frames", unit="frame")
    except NameError:
        # Fallback if tqdm is not available
        progress_bar = sorted_frame_keys
        print("Processing frames...")
    
    for frame_key in progress_bar:
        frame_info_list = frame_groups[frame_key]
        
        # Load all images for this frame
        batch_images = []
        true_labels = []
        valid_files = []
        
        for img_info in frame_info_list:
            try:
                image = cv2.imread(img_info['path'], cv2.IMREAD_GRAYSCALE)
                if image is not None:
                    batch_images.append(image)
                    true_labels.append(img_info['code'] if calculate_accuracy else None)
                    valid_files.append(img_info)
                else:
                    print(f"Warning: Could not load {img_info['path']}")
            except Exception as e:
                print(f"Error loading {img_info['path']}: {e}")
                continue
        
        if not batch_images:
            continue
            
        # Process frame batch with timing
        try:
            # Determine if we should use spatial filtering for this frame
            use_filtering_this_frame = (use_spatial_filtering and 
                                      (current_search_point is not None or first_frame_processed))
            
            if use_filtering_this_frame:
                filter_point = current_search_point
                filter_radius = adaptive_radius
            else:
                filter_point = None
                filter_radius = None
            
            start_time = time.time()
            indices, scores = system.forward(batch_images, point=filter_point, radius=filter_radius)
            forward_time = time.time() - start_time
            
            # Move results to CPU for processing
            indices_cpu = indices.cpu() if indices.is_cuda else indices
            scores_cpu = scores.cpu() if scores.is_cuda else scores
            
            total_forward_time += forward_time
            frame_times.append(forward_time)
            codes_in_frame = len(batch_images)
            total_codes_processed += codes_in_frame
            
            # Clear GPU cache periodically to avoid memory buildup
            if torch.cuda.is_available() and len(frame_times) % 10 == 0:
                torch.cuda.empty_cache()
            
            # Collect detected marker labels for adaptive filtering
            frame_detected_labels = []
            
            # Store results and calculate accuracy
            for j, file_info in enumerate(valid_files):
                filename = file_info['filename']
                if j < len(indices_cpu):
                    # Convert indices to labels - ensure indices are integers and filter out invalid ones
                    idx_list = indices_cpu[j].tolist() if hasattr(indices_cpu[j], 'tolist') else list(indices_cpu[j])
                    idx_ints = [int(idx) for idx in idx_list]  # Ensure integer conversion
                    
                    # Filter out invalid indices (-1) that were rejected by spatial filtering
                    valid_idx_ints = [idx for idx in idx_ints if idx >= 0]
                    valid_scores = []
                    score_list = scores_cpu[j].tolist() if hasattr(scores_cpu[j], 'tolist') else list(scores_cpu[j])
                    
                    # Keep only scores corresponding to valid indices
                    for k, idx in enumerate(idx_ints):
                        if idx >= 0 and k < len(score_list):
                            valid_scores.append(score_list[k])
                    
                    # Debug: Check what's happening with the mapping (only for first few)
                    if total_predictions < 5:  # Only for first few predictions
                        print(f"Debug for {filename}:")
                        print(f"  Raw indices: {indices_cpu[j]}")
                        print(f"  Converted indices: {idx_ints}")
                        print(f"  Valid indices after spatial filtering: {valid_idx_ints}")
                        print(f"  Index2label keys (first 10): {list(system.index2label.keys())[:10]}")
                        if valid_idx_ints:
                            print(f"  Index2label[{valid_idx_ints[0]}] = {system.index2label.get(valid_idx_ints[0], 'NOT FOUND')}")
                    
                    top_labels = [system.index2label.get(idx, f"unknown_{idx}") for idx in valid_idx_ints]
                    
                    # Collect top prediction label for adaptive filtering (ignore unknown labels and invalid indices)
                    if top_labels and str(top_labels[0]).startswith('unknown_') == False:
                        frame_detected_labels.append(top_labels[0])
                    
                    results[filename] = {
                        'labels': top_labels,
                        'scores': valid_scores,
                        'indices': valid_idx_ints,
                        'frame_time_ms': forward_time * 1000 / codes_in_frame  # Time per code in this frame
                    }
                    
                    # Calculate accuracy if we have true labels
                    if calculate_accuracy and j < len(true_labels) and true_labels[j] is not None:
                        true_label = true_labels[j]
                        if true_label in system.label2index:
                            true_index = system.label2index[true_label]
                            
                            # Check top-1 accuracy (only if we have valid predictions)
                            if len(valid_idx_ints) > 0 and valid_idx_ints[0] == true_index:
                                top1_correct += 1
                            
                            # Check top-3 accuracy  
                            if true_index in valid_idx_ints:
                                top3_correct += 1
                            
                            total_predictions += 1
                            
                            # Print detailed results for first few predictions
                            if total_predictions <= 10:
                                print(f"Image: {filename}")
                                print(f"  True label: {true_label} (index: {true_index})")
                                print(f"  Predicted labels: {top_labels}")
                                print(f"  Predicted indices: {valid_idx_ints}")
                                print(f"  Scores: {results[filename]['scores']}")
                                print(f"  Top-1 correct: {valid_idx_ints[0] == true_index if valid_idx_ints else False}")
                                print(f"  Top-3 correct: {true_index in valid_idx_ints}")
                                print()
                        
                else:
                    results[filename] = {'labels': [], 'scores': [], 'indices': [], 'frame_time_ms': 0}
            
            # Update search point based on detected markers in this frame
            if use_spatial_filtering and frame_detected_labels:
                new_search_point = system.get_average_position(frame_detected_labels)
                if new_search_point is not None:
                    current_search_point = new_search_point
                    first_frame_processed = True
                    if len(frame_times) <= 10:  # Print info for first few frames
                        print(f"Frame {frame_key}: Updated search point to {current_search_point} based on {len(frame_detected_labels)} detections")
            elif use_spatial_filtering and not first_frame_processed:
                # First frame processed, but no detections - mark as processed anyway
                first_frame_processed = True
                if len(frame_times) <= 10:
                    print(f"Frame {frame_key}: No detections in first frame, will continue without initial search point")
            elif len(frame_times) <= 10:
                print(f"Frame {frame_key}: No valid detections found, keeping search point at {current_search_point}")
                    
        except Exception as e:
            print(f"Error processing frame {frame_key}: {e}")
            # Add empty results for failed frame
            for file_info in valid_files:
                results[file_info['filename']] = {'labels': [], 'scores': [], 'indices': [], 'frame_time_ms': 0}
            continue
    
    # Calculate timing statistics
    avg_time_per_frame_ms = (total_forward_time * 1000 / len(frame_times)) if frame_times else 0
    avg_time_per_code_ms = (total_forward_time * 1000 / total_codes_processed) if total_codes_processed > 0 else 0
    fps_frame = 1000 / avg_time_per_frame_ms if avg_time_per_frame_ms > 0 else 0
    fps_code = 1000 / avg_time_per_code_ms if avg_time_per_code_ms > 0 else 0
    
    # Print final results
    print("\n" + "="*60)
    print("PROCESSING RESULTS")
    print("="*60)
    
    # Device and memory info
    print(f"Device used: {DEVICE}")
    if torch.cuda.is_available():
        print(f"Peak GPU memory allocated: {torch.cuda.max_memory_allocated() / 1e6:.1f} MB")
        print(f"Peak GPU memory cached: {torch.cuda.max_memory_reserved() / 1e6:.1f} MB")
        torch.cuda.reset_peak_memory_stats()  # Reset for next run
    
    # Timing results
    print(f"Total frames processed: {len(frame_times)}")
    print(f"Total codes processed: {total_codes_processed}")
    print(f"Total forward time: {total_forward_time:.3f}s")
    print(f"Average time per frame: {avg_time_per_frame_ms:.2f}ms ({fps_frame:.1f} FPS)")
    print(f"Average time per code: {avg_time_per_code_ms:.2f}ms ({fps_code:.1f} codes/sec)")
    
    # Accuracy results
    if calculate_accuracy and total_predictions > 0:
        top1_accuracy = top1_correct / total_predictions
        top3_accuracy = top3_correct / total_predictions
        
        print(f"\nACCURACY RESULTS:")
        print(f"Total predictions: {total_predictions}")
        print(f"Top-1 accuracy: {top1_correct}/{total_predictions} = {top1_accuracy:.4f} ({top1_accuracy*100:.2f}%)")
        print(f"Top-3 accuracy: {top3_correct}/{total_predictions} = {top3_accuracy:.4f} ({top3_accuracy*100:.2f}%)")
    
    print("="*60)
    
    # Save results if output file specified
    if output_file:
        # Add timing summary to results
        timing_summary = {
            'device': str(DEVICE),
            'cuda_available': torch.cuda.is_available(),
            'peak_gpu_memory_mb': torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0,
            'total_frames': len(frame_times),
            'total_codes': total_codes_processed,
            'total_forward_time_s': total_forward_time,
            'avg_time_per_frame_ms': avg_time_per_frame_ms,
            'avg_time_per_code_ms': avg_time_per_code_ms,
            'fps_frame': fps_frame,
            'fps_code': fps_code
        }
        
        results_with_timing = {
            'timing_summary': timing_summary,
            'results': results
        }
        
        with open(output_file, 'w') as f:
            json.dump(results_with_timing, f, indent=2)
        print(f"Results with timing data saved to {output_file}")
    
    return results



# ================================================================================================
# EXAMPLE USAGE AND TESTING
# ================================================================================================

if __name__ == "__main__":
    """
    Example usage and testing script for the FourierDecoder system.
    
    This section demonstrates three different usage modes:
    1. auto_single: Process a single image with automatic system setup
    2. auto_directory: Process a directory of images with comprehensive evaluation
    3. manual: Process images using pre-generated utility files
    
    For production use in a robot server, initialize FourierDecoder once and then
    call predict_with_labels() or forward() methods on image crops.
    """
    
    # ============================================================================================
    # CONFIGURATION - Update these paths for your environment
    # ============================================================================================
    
    # Path to labels2coords pickle file containing marker spatial coordinates
    # labels2coords_path = "/home/alricofrahls/bigdisk/AlricOfRahls/VkusvillODM/VUC_0deg/VUC_labels2coords.pickle"
    labels2coords_path = "/home/user/fliftbot/warehouse/x5/labels2coords.pickle"
    
    # Test image for single processing mode
    test_image_path = "/home/user/Downloads/fourier_decoder/frame1_code38_id3.png"
    
    # Test directory for batch processing mode
    # test_directory = "/home/alricofrahls/bigdisk/AlricOfRahls/VkusvillODM/VUC_0deg/crops_single"
    test_directory = "/home/alricofrahls/bigdisk/AlricOfRahls/VkusvillODM/Veshki_0deg/Veshki_0deg_single"
    
    # ============================================================================================
    # PROCESSING MODE SELECTION
    # ============================================================================================
    
    # Processing mode: 'auto_single', 'auto_directory', or 'manual'
    # - auto_single: Quick test with one image, automatic setup
    # - auto_directory: Full evaluation with timing and accuracy metrics
    # - manual: Use pre-generated utility files (advanced users)
    # mode = 'auto_directory'
    mode = 'auto_single'
    
    # ============================================================================================
    # MODE EXECUTION
    # ============================================================================================
    
    if mode == 'auto_single':
        # Single image processing with automated utilities - good for quick testing
        print("Processing single image with automated utilities...")
        print("=" * 60)
        
        # Initialize decoder system (this handles all file generation automatically)
        system = FourierDecoder.from_labels2coords(labels2coords_path, top_n=3)
        
        # Load and process test image
        img_np = cv2.imread(test_image_path)
        if img_np is None:
            raise FileNotFoundError(f"Test image not found: {test_image_path}")
        
        # Get predictions with human-readable labels
        predicted_labels, scores = system.predict_with_labels([img_np])
        
        # Display results
        print(f"Image: {test_image_path}")
        print(f"Top predictions: {predicted_labels[0]}")
        print(f"Confidence scores: {scores[0]}")  # scores is now a list, not a tensor
        
    elif mode == 'auto_directory':
        # Directory processing with comprehensive evaluation - production testing
        print("Processing directory with automated utilities...")
        print("=" * 60)
        
        # Process entire directory with timing and accuracy analysis
        results = process_directory_auto(
            image_dir=test_directory,
            labels2coords_path=labels2coords_path,
            output_file="detection_results.json",
            point=None,  # No initial spatial filtering point
            radius=None,  # Will use default radius if spatial filtering enabled
            use_spatial_filtering=False,  # Enable adaptive spatial filtering
            top_n=3
        )
        
        # Print summary statistics
        total_files = len(results)
        successful_detections = sum(1 for r in results.values() if r['labels'])
        print(f"\nProcessing Summary:")
        print(f"Files processed: {total_files}")
        print(f"Successful detections: {successful_detections}")
        print(f"Success rate: {successful_detections/total_files*100:.1f}%" if total_files > 0 else "No files processed")
        
    elif mode == 'manual':
        # Manual mode using existing utilities - for advanced users with pre-generated files
        print("Manual mode - using existing utilities...")
        print("=" * 60)
        print("Note: This mode requires pre-generated utility files")
        
        # Define paths to manually created utility files
        mask_dir = "./bit_masks"
        warehouse_weights_path = "./warehouse_weights.pickle"
        index2label_path = "./index2label.pickle"
        label2index_path = "./label2index.pickle"
        
        # Verify all required files exist
        required_files = [mask_dir, warehouse_weights_path, index2label_path, label2index_path]
        missing_files = [f for f in required_files if not os.path.exists(f)]
        
        if missing_files:
            print(f"Required files not found: {missing_files}")
            print("Please generate utilities first or use 'auto_single' or 'auto_directory' mode")
            exit(1)
        
        # Load pre-generated mapping files
        with open(index2label_path, 'rb') as f:
            index2label = pickle.load(f)
        with open(label2index_path, 'rb') as f:
            label2index = pickle.load(f)
        
        # Create decoder system manually with explicit file paths
        system = FourierDecoder(mask_dir, warehouse_weights_path, index2label, label2index, labels2coords_path, top_n=3)
        
        # Process test image
        img_np = cv2.imread(test_image_path)
        if img_np is None:
            raise FileNotFoundError(f"Test image not found: {test_image_path}")
        
        # Get predictions and display results
        predicted_labels, scores = system.predict_with_labels([img_np])
        print(f"Image: {test_image_path}")
        print(f"Top predictions: {predicted_labels[0]}")
        print(f"Confidence scores: {scores[0].numpy()}")
    
    else:
        print(f"Unknown mode: {mode}")
        print("Available modes:")
        print("  - 'auto_single': Process one image with automatic setup (quick testing)")
        print("  - 'auto_directory': Process directory with full evaluation (production testing)")
        print("  - 'manual': Use pre-generated files (advanced users)")