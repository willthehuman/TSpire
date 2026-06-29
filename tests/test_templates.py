"""TemplateDB tests: jar (zip) + directory loading, category filtering, colour matching.

Uses a synthetic jar of solid-colour PNGs so it doesn't depend on the installed game.
Skips if OpenCV/numpy aren't available.
"""

import zipfile

import pytest

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")

from tspire.host.game_assets import find_game_jar
from tspire.host.vision.templates import TemplateDB


def _solid_png(bgr: tuple[int, int, int], size: int = 32) -> bytes:
    img = np.zeros((size, size, 4), np.uint8)
    img[:, :, 0], img[:, :, 1], img[:, :, 2] = bgr
    img[:, :, 3] = 255  # opaque
    return cv2.imencode(".png", img)[1].tobytes()


def _make_jar(path) -> None:
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("images/relics/red.png", _solid_png((0, 0, 255)))
        z.writestr("images/relics/blue.png", _solid_png((255, 0, 0)))
        z.writestr("images/relics/green.png", _solid_png((0, 255, 0)))
        # these must be ignored by the loader:
        z.writestr("images/relics/outline/red.png", _solid_png((0, 0, 255)))
        z.writestr("images/relics/test1.png", _solid_png((0, 0, 255)))
        z.writestr("images/ui/intent/attack.png", _solid_png((10, 10, 200)))


def test_jar_loads_only_real_relics(tmp_path):
    jar = tmp_path / "desktop-1.0.jar"
    _make_jar(jar)
    db = TemplateDB(jar)
    assert db.is_jar
    ids = {name for name, _ in db._load_category("relics")}
    assert ids == {"red", "blue", "green"}  # outline/ and test1 excluded
    assert db.available("relics") and db.available("intents")


def test_jar_classify_matches_by_colour(tmp_path):
    jar = tmp_path / "desktop-1.0.jar"
    _make_jar(jar)
    db = TemplateDB(jar)
    red_crop = np.zeros((20, 20, 3), np.uint8)
    red_crop[:, :, 2] = 255  # BGR red
    name, score = db.classify(red_crop, "relics")
    assert name == "red" and score > 0.9


def test_directory_mode(tmp_path):
    rel = tmp_path / "relics"
    rel.mkdir()
    (rel / "red.png").write_bytes(_solid_png((0, 0, 255)))
    (rel / "blue.png").write_bytes(_solid_png((255, 0, 0)))
    db = TemplateDB(tmp_path)
    assert not db.is_jar
    blue_crop = np.zeros((20, 20, 3), np.uint8)
    blue_crop[:, :, 0] = 255  # BGR blue
    assert db.classify(blue_crop, "relics")[0] == "blue"


def test_find_game_jar_explicit(tmp_path):
    jar = tmp_path / "desktop-1.0.jar"
    jar.write_bytes(b"PK\x03\x04")  # just needs to exist
    assert find_game_jar(str(jar)) == jar
    assert find_game_jar(str(tmp_path / "missing.jar")) != jar
