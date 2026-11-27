import numpy as np
import trimesh

from app.inference import _decimate_mesh, _remove_small_components, _smooth_mesh


def _colored_sphere(subdivisions=5):
    mesh = trimesh.creation.icosphere(subdivisions=subdivisions)
    # red on one half, blue on the other
    colors = np.where(mesh.vertices[:, [0]] > 0, [[255, 0, 0, 255]], [[0, 0, 255, 255]]).astype(np.uint8)
    mesh.visual = trimesh.visual.ColorVisuals(mesh, vertex_colors=colors)
    return mesh


def test_decimation_hits_target_and_keeps_colours():
    mesh = _colored_sphere()
    out = _decimate_mesh(mesh, 2000)
    assert len(out.faces) <= 2100 and len(out.faces) < len(mesh.faces)
    colours = out.visual.vertex_colors
    assert len(colours) == len(out.vertices)
    right = colours[out.vertices[:, 0] > 0.3]
    left = colours[out.vertices[:, 0] < -0.3]
    assert (right[:, 0] > 200).all() and (left[:, 2] > 200).all()  # red stays red, blue stays blue


def test_small_floating_pieces_are_removed():
    body = trimesh.creation.icosphere(subdivisions=4)
    speck = trimesh.creation.icosphere(subdivisions=1)
    speck.apply_translation([5, 0, 0])
    combined = trimesh.util.concatenate([body, speck])
    cleaned = _remove_small_components(combined, 0.03)
    assert len(cleaned.faces) == len(body.faces)


def test_smoothing_keeps_vertex_count_and_colours():
    mesh = _colored_sphere(subdivisions=3)
    mesh.vertices += np.random.default_rng(0).normal(0, 0.01, mesh.vertices.shape)
    before = len(mesh.vertices)
    out = _smooth_mesh(mesh, 5)
    assert len(out.vertices) == before and out.visual.kind == "vertex"
