import numpy as np

from pathlib import Path

cos_phi = np.fromfile(Path(Path(__file__).parent, 'cos_phi.npy'))
sin_phi = np.fromfile(Path(Path(__file__).parent, 'sin_phi.npy'))
