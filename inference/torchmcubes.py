"""Stand-in for ``torchmcubes`` so TripoSR runs on Windows without a C++ compiler.

TripoSR does ``from torchmcubes import marching_cubes``. The real package has to be compiled
(Visual Studio Build Tools + CMake), which is the most common install failure on Windows.
This module offers the same function using scikit-image, which ships ready-made wheels.

``uvicorn`` is started from the ``inference`` folder, so this file is found before any
installed ``torchmcubes``. Delete it if you prefer the compiled package.
"""

import numpy as np
import torch
from skimage import measure


def marching_cubes(vol: torch.Tensor, thresh: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(vertices[V,3] float32, faces[F,3] int64)`` for the ``thresh`` iso-surface of ``vol``."""
    volume = np.ascontiguousarray(vol.detach().cpu().numpy(), dtype=np.float32)
    try:
        verts, faces, _normals, _values = measure.marching_cubes(volume, level=float(thresh))
    except ValueError:
        # No surface crosses the level (empty or solid volume). The real package returns an empty mesh.
        return torch.zeros((0, 3), dtype=torch.float32), torch.zeros((0, 3), dtype=torch.int64)
    return torch.from_numpy(verts.astype(np.float32)), torch.from_numpy(faces.astype(np.int64))
