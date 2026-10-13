import json
import os
import pickle
from pathlib import Path
# add a path to the navigation to path variable
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from hamming import encode
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal

def circular_convolve(x, h):
    """
    Compute the circular convolution of a 1D array `x` with an FIR filter `h`.

    Parameters:
    x (array_like): Input signal (circular array).
    h (array_like): FIR filter coefficients.

    Returns:
    numpy.ndarray: Circular convolution result, same length as `x`.
    """
    x = np.asarray(x)
    h = np.asarray(h)
    N = x.size
    M = h.size
    
    if N == 0 or M == 0:
        return np.zeros_like(x)
    
    # Compute full linear convolution
    y_linear = scipy.signal.convolve(x, h, mode='full')
    
    # Take the first N elements
    y_circular = y_linear[:N].copy()
    
    # Add the tail for wrap-around if M > 1
    if M > 1:
        tail = y_linear[N:N + M - 1]
        y_circular[:len(tail)] += tail
    
    return y_circular


class MarkerSignalSimulator:
    def __init__(self, bits: str | int, samples: int, subsample_shift: float, roll: int = 0, marker_length: int | None = None, noise_sigma: float = 0.0):
        if isinstance(bits, str):
            if marker_length is not None:
                raise ValueError("Marker length cannot be specified when bits are provided as a string.")
            marker_length = len(bits)
        self.bits = bits
        self.samples = samples
        self.subsample_shift = subsample_shift
        self.roll = roll
        self.marker_length = marker_length if marker_length is not None else len(self.bits)

        if isinstance(bits, int):
            bin_str = bin(bits)[2:]
            if len(bin_str) > marker_length:
                bin_str = bin_str[-marker_length:]
            else:
                bin_str = bin_str.zfill(marker_length)
            bits = list(bin_str)
        elif isinstance(bits, str):
            if not all(c in '01' for c in bits):
                raise ValueError("String bits must contain only '0' and '1'")
            if len(bits) > marker_length:
                bin_str = bits[-marker_length:]
            else:
                bin_str = bits.zfill(marker_length)
            bits = list(bin_str)
        else:
            raise TypeError("bits must be int or str")
        
        assert self.samples % len(bin_str) == 0, "Samples must be divisible by the length of bits"

        
        if marker_length == 0:
            arr = np.array([], dtype=np.int8)
        else:
            arr = np.array([int(bit) for bit in bin_str], dtype=np.int8)
        
        arr_rep = np.repeat(arr, samples // len(arr))
        arr_roll = np.roll(arr_rep, roll)
        arr_subsample_shifted = np.zeros_like(arr_roll, dtype=np.float64)
        arr_subsample_shifted[:-1] = arr_roll[:-1] * subsample_shift + arr_roll[1:] * (1 - subsample_shift)
        arr_subsample_shifted[-1] = arr_roll[-1] * subsample_shift + arr_roll[0] * (1 - subsample_shift)

        self.signal = arr_subsample_shifted
        kernel = np.zeros(self.samples)
        kernel[:self.samples // self.marker_length] = 1

        self.signal_convolved = circular_convolve(self.signal, kernel) / np.sum(kernel) + np.random.normal(0, noise_sigma, len(arr_subsample_shifted))

    def plot_signal(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.signal, label='Signal', marker='.')
        plt.plot(self.signal_convolved, label='Convolved Signal', linestyle='--', marker='x')
        plt.title('Marker Signal')
        plt.xlabel('Sample Index')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid()
        plt.show()


def generate_warehouse_weights(labels_file: str = None, output_file: str = 'warehouse_weights.pickle', 
                              index2label_file: str = 'index2label.pickle', label2index_file: str = 'label2index.pickle', 
                              bits: int = 16, samples: int = 32, labels2coords_file: str = None):
    """
    Generate warehouse weights for the decoding markers.
    
    Args:
        labels_file: Path to either:
                    - A pickle file containing labels2coords dictionary (legacy format)
                    - A text file with one integer label per line (new format)
        output_file: Path to save the generated warehouse weights
        index2label_file: Path to save index-to-label mapping
        label2index_file: Path to save label-to-index mapping
        bits: Number of bits for marker encoding
        samples: Number of samples for signal generation
        labels2coords_file: (Deprecated) Old parameter name for backward compatibility,
                           use labels_file instead
    """
    # Handle backward compatibility
    if labels2coords_file is not None and labels_file is None:
        labels_file = labels2coords_file
        print("Warning: 'labels2coords_file' parameter is deprecated, use 'labels_file' instead")
    elif labels_file is None:
        raise ValueError("Either 'labels_file' or 'labels2coords_file' must be provided")
    
    # Determine file type and load labels accordingly, labels file can be str or PosixPath
    labels_path = Path(labels_file)
    if labels_path.suffix.lower() == '.txt':
        # Load labels from text file (one label per line)
        print(f"Loading labels from text file: {labels_file}")
        with open(labels_file, 'r') as f:
            labels_list = []
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:  # Skip empty lines
                    try:
                        label = int(line)
                        labels_list.append(label)
                    except ValueError:
                        print(f"Warning: Skipping invalid label on line {line_num}: '{line}' (not an integer)")
        
        if not labels_list:
            raise ValueError(f"No valid integer labels found in {labels_file}")
        
        keys_list = labels_list
        print(f"Loaded {len(keys_list)} labels from text file")
    elif labels_path.suffix.lower() == '.json':
        print(f"Loading labels from JSON file: {labels_file}")
        with open(labels_file, 'r', encoding='utf-8') as f:
            labels2coords = json.load(f)
        keys_list = [int(k) for k in labels2coords.keys()]
        print(f"Extracted {len(keys_list)} labels from labels2coords JSON")
    else:
        # Load labels from pickle file (legacy format with labels2coords dictionary)
        print(f"Loading labels from pickle file: {labels_file}")
        with open(labels_file, 'rb') as f:
            labels2coords = pickle.load(f)
        keys_list = [int(k) for k in labels2coords.keys()]
        print(f"Extracted {len(keys_list)} labels from labels2coords dictionary")
    
    total_markers = len(keys_list)
    print(f"Total markers to process: {total_markers}")

    result = np.zeros((total_markers, 2, samples), dtype=np.float64)
    # keys_list = list(labels2coords.keys())
    for i, label in enumerate(keys_list):
        # print(f"Code for {label} is {bin(encode(label))}")
        marker_signal = MarkerSignalSimulator(encode(label), samples, subsample_shift=0, roll=0, marker_length=bits, noise_sigma=0)
        start_bit_signal = MarkerSignalSimulator(bits='1100000000000000', samples=samples, subsample_shift=0, roll=0, noise_sigma=0)

        result[i, 0] = marker_signal.signal_convolved
        result[i, 1] = start_bit_signal.signal_convolved

        print(f'Label: {label}, Signal: {marker_signal.signal_convolved}, Start Bit Signal: {start_bit_signal.signal_convolved}')

    # pickle the result
    with open(output_file, 'wb') as f:
        pickle.dump(result, f)
    print(result)

    # Construct and save index2label mapping
    index2label = {i: label for i, label in enumerate(keys_list)}
    with open(index2label_file, 'wb') as f:
        pickle.dump(index2label, f)
    # Construct and save label2index mapping
    label2index = {label: i for i, label in enumerate(keys_list)}
    with open(label2index_file, 'wb') as f:
        pickle.dump(label2index, f)

    print(f"Warehouse weights saved to {output_file}")
    print(f"Index to label mapping saved to {index2label_file}")
    print(f"Label to index mapping saved to {label2index_file}")


def generate_warehouse_weights_from_labels_txt(labels_txt_file: str, output_file: str = 'warehouse_weights.pickle', 
                                              index2label_file: str = 'index2label.pickle', 
                                              label2index_file: str = 'label2index.pickle', 
                                              bits: int = 16, samples: int = 32):
    """
    Convenience function to generate warehouse weights from a text file containing labels.
    
    Args:
        labels_txt_file: Path to text file with one integer label per line
        output_file: Path to save the generated warehouse weights
        index2label_file: Path to save index-to-label mapping
        label2index_file: Path to save label-to-index mapping
        bits: Number of bits for marker encoding
        samples: Number of samples for signal generation
    """
    return generate_warehouse_weights(
        labels_file=labels_txt_file,
        output_file=output_file,
        index2label_file=index2label_file,
        label2index_file=label2index_file,
        bits=bits,
        samples=samples
    )


if __name__ == "__main__":
    # Example usage - both formats supported
    
    # Option 1: Using pickle file with labels2coords dictionary (legacy format)
    generate_warehouse_weights(labels_file='/home/alricofrahls/bigdisk/AlricOfRahls/VkusvillODM/warehouse/veshki/labels2coords.pickle')
    
    # Option 2: Using text file with labels (new format)
    # Create example labels.txt file:
    # 100
    # 101 
    # 102
    # 200
    # generate_warehouse_weights(labels_file='labels.txt')
    
    # Option 3: Using convenience function for text files
    # generate_warehouse_weights_from_labels_txt('labels.txt')