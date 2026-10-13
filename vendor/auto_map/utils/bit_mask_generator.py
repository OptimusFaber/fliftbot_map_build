"""
Circular 16-bit marker generator for fliftbot navigation system.

This module creates circular markers with binary encoding around the perimeter
for robot navigation and localization purposes.
"""

from typing import Optional
import numpy as np
import cv2
from random import randint


def generate_bit_mask(position_index: int, total_bits: int = 16, samples: float = 0, radius: float = 64, outer: bool = True) -> np.ndarray:
    
    assert samples % total_bits == 0, "Samples must be a multiple of total bits"
    # Calculate marker geometry
    R = radius  # Outer radius
    Rs = int(0.85 * radius)  # Data ring radius (85% of outer radius)
    r_big = int(0.45 * radius)   # Center circle radius (45% of outer radius)
    r = int(0.3 * radius)   # Center circle radius (30% of outer radius)

    # Convert string code to list of integers for processing
    code = [0 for _ in range(samples)]
    for i in range(position_index, position_index + samples // total_bits):
        code[i % samples] = 1

    # Create black background image (RGB format)
    img = np.full((R * 2, R * 2, 3), 0, dtype=np.uint8)

    # Draw the 16-bit binary encoding around the perimeter
    # Each bit occupies π/8 radians (22.5 degrees) of the circle
    angle_per_sample = 360 / samples  # 22.5 degrees per bit
    current_angle = 0
    
    if outer:
        for i, bit_value in enumerate(code):
            start_angle = current_angle
            end_angle = current_angle + angle_per_sample
            
            # Draw black segment for '1' bits (leave white for '0' bits)
            if bit_value == 1:
                cv2.ellipse(img, (R, R), (Rs + 2, Rs + 2), 0, start_angle, end_angle, (255, 255, 255), -1)
            
            current_angle = end_angle
        cv2.circle(img, (R, R), r_big, (0, 0, 0), -1)
    else:
        for i, bit_value in enumerate(code):
            start_angle = current_angle
            end_angle = current_angle + angle_per_sample
            
            # Draw black segment for '1' bits (leave white for '0' bits)
            if bit_value == 1:
                cv2.ellipse(img, (R, R), (r + 2, r + 2), 0, start_angle, end_angle, (255, 255, 255), -1)
            
            current_angle = end_angle
    
    
    # Generate unique filename with rotation and random identifier
    final_name = f"{'out' if outer else 'in'}_mask_{position_index}"
    output_path = f"./bit_masks/{final_name}.png"
    
    # Save the generated marker
    cv2.imwrite(output_path, img)
    
    return final_name


if __name__ == "__main__":
    total_bits = 16
    samples = 32
    for i in range(samples):
        for offset in [0, 0.5]:
            generate_bit_mask(position_index=i, total_bits=total_bits, samples=samples, radius=64, outer=True)
            generate_bit_mask(position_index=i, total_bits=total_bits, samples=samples, radius=64, outer=False)
    print("Bit masks generated and saved in 'bit_masks' directory.")

