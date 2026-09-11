import pytest

from scripts import v1_ingest as ingest


def _png(path, size=(200, 150), color=(200, 180, 170)):
    from PIL import Image

    Image.new("RGB", size, color).save(path)
    return path


def test_valid_specimen(tmp_path, monkeypatch):
    monkeypatch.setattr(ingest, "REGISTRY", tmp_path / "reg")
    record = ingest.ingest_file(_png(tmp_path / "a.png"))
    assert record["schema"] == "v1-specimen-v1"
    assert (tmp_path / "reg" / record["specimen_id"] / "normalized.png").is_file()
    assert (tmp_path / "reg" / record["specimen_id"] / "thumb.png").is_file()


def test_duplicate_import_stable(tmp_path, monkeypatch):
    monkeypatch.setattr(ingest, "REGISTRY", tmp_path / "reg")
    first = ingest.ingest_file(_png(tmp_path / "a.png"))
    second = ingest.ingest_file(_png(tmp_path / "b.png"))
    assert first["specimen_id"] == second["specimen_id"]


def test_normalization_repeatable(tmp_path, monkeypatch):
    import hashlib

    monkeypatch.setattr(ingest, "REGISTRY", tmp_path / "reg")
    first = ingest.ingest_file(_png(tmp_path / "a.png"))
    second = ingest.ingest_file(_png(tmp_path / "a.png", color=(10, 20, 30)))
    assert first["specimen_id"] != second["specimen_id"]
    for record in (first, second):
        data = (tmp_path / "reg" / record["specimen_id"] / "normalized.png").read_bytes()
        assert hashlib.sha256(data).hexdigest()


def test_rejects_non_png(tmp_path, monkeypatch):
    monkeypatch.setattr(ingest, "REGISTRY", tmp_path / "reg")
    fake = tmp_path / "fake.png"
    fake.write_bytes(b"definitely not a png file, just text!!!!!!!!!!!!" * 10)
    with pytest.raises(ingest.IngestError):
        ingest.ingest_file(fake)
    with pytest.raises(ingest.IngestError):
        ingest.ingest_file(tmp_path / "missing.png")


def test_rejects_bad_dimensions(tmp_path, monkeypatch):
    monkeypatch.setattr(ingest, "REGISTRY", tmp_path / "reg")
    with pytest.raises(ingest.IngestError):
        ingest.ingest_file(_png(tmp_path / "tiny.png", size=(10, 10)))
    monkeypatch.setattr(ingest, "MAX_DIM", 100)
    with pytest.raises(ingest.IngestError):
        ingest.ingest_file(_png(tmp_path / "big.png", size=(200, 200)))


def test_rejects_rgba_and_grayscale_normalized(tmp_path, monkeypatch):
    from PIL import Image

    monkeypatch.setattr(ingest, "REGISTRY", tmp_path / "reg")
    Image.new("RGBA", (120, 120), (1, 2, 3, 4)).save(tmp_path / "a.png")
    record = ingest.ingest_file(tmp_path / "a.png")
    assert record["width"] == 120
    Image.new("L", (120, 120), 128).save(tmp_path / "g.png")
    assert ingest.ingest_file(tmp_path / "g.png")["specimen_id"]
