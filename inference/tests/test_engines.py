from app.engines import _extract_mesh


class NewTSR:
    """TripoSR with the required has_vertex_color argument."""

    def extract_mesh(self, scene_codes, has_vertex_color, resolution=256, threshold=25.0):
        return [("mesh", has_vertex_color, resolution)]


class OldTSR:
    """Older TripoSR without that argument."""

    def extract_mesh(self, scene_codes, resolution=256, threshold=25.0):
        return [("mesh", None, resolution)]


def test_new_triposr_gets_vertex_colors():
    assert _extract_mesh(NewTSR(), object(), 256) == ("mesh", True, 256)


def test_old_triposr_still_works():
    assert _extract_mesh(OldTSR(), object(), 128) == ("mesh", None, 128)
