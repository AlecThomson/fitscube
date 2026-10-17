"""Tests for optional cube compression"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from fitscube.cli import cli
from fitscube.combine_fits import combine_fits
from fitscube.compress import (
    CompressMethod,
    compress_cube,
    decompress_cube,
    tile_compress_cube,
)


def write_noise_cube(path: Path, shape: tuple[int, ...] = (2, 5, 40, 48)) -> np.ndarray:
    """A float cube of noise with a blanked strip, an all-blank plane and a beam table."""
    rng = np.random.default_rng(42)
    data = rng.normal(0, 1e-3, shape).astype(np.float32)
    data[..., :8, :] = np.nan
    data[(0,) * (len(shape) - 3) + (1,)] = np.nan
    header = fits.Header()
    header["BUNIT"] = "Jy/beam"
    header["CTYPE3"] = "FREQ"
    beams = fits.BinTableHDU.from_columns(
        [fits.Column(name="BMAJ", format="E", array=np.arange(shape[-3], dtype=float))],
        name="BEAMS",
    )
    fits.HDUList([fits.PrimaryHDU(data, header), beams]).writeto(path)
    return data


@pytest.mark.parametrize("method", ["gzip", "pgzip", "tile-gzip"])
def test_compress_cube_roundtrip(
    file_list_onebeam: list[Path], tmp_path: Path, method: CompressMethod
) -> None:
    """Lossless compression should keep the data and beams, and remove the original"""
    output_file = tmp_path / "test.fits"
    combine_fits(
        file_list=file_list_onebeam,
        out_cube=output_file,
        time_domain_mode=True,
        overwrite=True,
    )

    with fits.open(output_file) as hdul:
        original_data = hdul[0].data.copy()
        original_beams = hdul[1].data.copy()

    compressed_path = compress_cube(output_file, method=method)

    suffix = ".fz" if method == "tile-gzip" else ".gz"
    assert compressed_path.name == output_file.name + suffix
    assert not output_file.exists()
    assert compressed_path.exists()

    with fits.open(compressed_path) as hdul:
        image_index = 1 if method == "tile-gzip" else 0
        np.testing.assert_array_equal(hdul[image_index].data, original_data)
        np.testing.assert_array_equal(
            hdul["BEAMS"].data["BMAJ"], original_beams["BMAJ"]
        )


def test_tile_rice_error_is_a_fraction_of_the_noise(tmp_path: Path) -> None:
    """RICE should keep blanks exactly and add an error well below the noise"""
    data = write_noise_cube(tmp_path / "cube.fits")

    compressed = compress_cube(
        tmp_path / "cube.fits", method="tile-rice", quantize_level=4
    )

    with fits.open(compressed) as hdul:
        assert hdul[1].header["BUNIT"] == "Jy/beam"
        got = hdul[1].data
    with fits.open(compressed, disable_image_compression=True) as hdul:
        assert hdul[1].header["ZQUANTIZ"] == "SUBTRACTIVE_DITHER_2"
    np.testing.assert_array_equal(np.isnan(got), np.isnan(data))
    finite = np.isfinite(data)
    # One quantization step is sigma / 4, so the rms error is about step / sqrt(12)
    assert np.std(got[finite] - data[finite]) < 1e-3 / 4 / np.sqrt(12) * 1.2


@pytest.mark.parametrize("tile_shape", [None, (16, 16)])
@pytest.mark.parametrize(
    ("compression_type", "quantize_level"), [("GZIP_2", 0.0), ("RICE_1", 4.0)]
)
def test_tile_compress_matches_astropy(
    tmp_path: Path,
    compression_type: str,
    quantize_level: float,
    tile_shape: tuple[int, int] | None,
) -> None:
    """Compressing plane by plane should decode exactly as astropy compressing the whole cube"""
    data = write_noise_cube(tmp_path / "cube.fits")

    compressed = tile_compress_cube(
        tmp_path / "cube.fits",
        compression_type=compression_type,  # type: ignore[arg-type]
        quantize_level=quantize_level,
        tile_shape=tile_shape,
        dither_seed=9999,
        max_workers=3,
    )

    reference = fits.CompImageHDU(
        data,
        compression_type=compression_type,
        tile_shape=(1, 1, *(tile_shape or data.shape[-2:])),
        quantize_level=quantize_level,
        quantize_method=2 if quantize_level else -1,
        dither_seed=9999,
    )
    fits.HDUList([fits.PrimaryHDU(), reference]).writeto(tmp_path / "reference.fits.fz")
    np.testing.assert_array_equal(
        fits.getdata(compressed, 1), fits.getdata(tmp_path / "reference.fits.fz", 1)
    )


def test_tile_compress_is_the_same_for_any_worker_count(tmp_path: Path) -> None:
    """Threads should never reorder the planes"""
    outputs = []
    for workers in (1, 4):
        cube = tmp_path / f"{workers}" / "cube.fits"
        cube.parent.mkdir()
        write_noise_cube(cube)
        outputs.append(
            tile_compress_cube(cube, "RICE_1", 4, max_workers=workers).read_bytes()
        )
    assert outputs[0] == outputs[1]


def test_tile_compress_refuses_compressed_input(tmp_path: Path) -> None:
    write_noise_cube(tmp_path / "cube.fits")
    compressed = tile_compress_cube(tmp_path / "cube.fits")
    with pytest.raises(ValueError, match="already compressed"):
        tile_compress_cube(compressed)


def test_tile_rice_needs_a_quantize_level(tmp_path: Path) -> None:
    write_noise_cube(tmp_path / "cube.fits")
    with pytest.raises(ValueError, match="RICE_1"):
        tile_compress_cube(tmp_path / "cube.fits", "RICE_1", 0)


@pytest.mark.parametrize("method", ["gzip", "tile-gzip"])
def test_decompress_cube_roundtrip(tmp_path: Path, method: CompressMethod) -> None:
    """Decompressing a losslessly compressed cube should give back the same file contents"""
    cube = tmp_path / "cube.fits"
    data = write_noise_cube(cube)
    original_header = fits.getheader(cube)

    compressed = compress_cube(cube, method=method)
    restored = decompress_cube(compressed)

    assert restored == cube
    assert compressed.exists()
    with fits.open(restored) as hdul:
        np.testing.assert_array_equal(hdul[0].data, data)
        for key in ("BUNIT", "CTYPE3", "NAXIS4"):
            assert hdul[0].header[key] == original_header[key]
        assert "FCLOSSY" not in hdul[0].header
        np.testing.assert_array_equal(hdul["BEAMS"].data["BMAJ"], np.arange(5))


def test_decompress_cube_refuses_to_overwrite(tmp_path: Path) -> None:
    cube = tmp_path / "cube.fits"
    write_noise_cube(cube)
    compressed = compress_cube(cube, method="tile-gzip")
    cube.touch()
    with pytest.raises(FileExistsError):
        decompress_cube(compressed)
    decompress_cube(compressed, overwrite=True, remove=True)
    assert not compressed.exists()


def test_requantizing_a_lossy_cube_is_refused(tmp_path: Path) -> None:
    """A decompressed RICE cube is marked lossy and not quantized a second time by accident"""
    cube = tmp_path / "cube.fits"
    write_noise_cube(cube)
    restored = decompress_cube(compress_cube(cube, method="tile-rice"), remove=True)
    lossy_values = fits.getdata(restored)
    assert fits.getheader(restored)["FCLOSSY"]
    assert fits.getheader(restored)["FCQLEVEL"] == 4

    with pytest.raises(ValueError, match="already quantized"):
        compress_cube(restored, method="tile-rice")

    # Lossless compression of the quantized values is fine
    lossless = compress_cube(restored, method="tile-gzip")
    np.testing.assert_array_equal(fits.getdata(lossless, 1), lossy_values)

    restored = decompress_cube(lossless, remove=True)
    assert compress_cube(restored, method="tile-rice", allow_requantize=True).exists()


def test_compress_and_decompress_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`fitscube compress` keeps its input unless asked; `decompress` restores the cube"""
    cube = tmp_path / "cube.fits"
    data = write_noise_cube(cube)

    monkeypatch.setattr("sys.argv", ["fitscube", "compress", str(cube)])
    cli()
    compressed = tmp_path / "cube.fits.fz"
    assert cube.exists()
    assert compressed.exists()

    cube.unlink()
    monkeypatch.setattr(
        "sys.argv", ["fitscube", "decompress", str(compressed), "--remove"]
    )
    cli()
    assert not compressed.exists()
    np.testing.assert_array_equal(fits.getdata(cube), data)


def test_requantizing_dithers_afresh(tmp_path: Path) -> None:
    """Repeated quantization must not reuse the dither, or the errors line up and add"""
    cube = tmp_path / "cube.fits"
    write_noise_cube(cube)
    first = compress_cube(cube, method="tile-rice")
    seed = fits.getheader(first, 1, disable_image_compression=True)["ZDITHER0"]

    second = compress_cube(
        decompress_cube(first, remove=True), method="tile-rice", allow_requantize=True
    )

    assert fits.getheader(second, 1, disable_image_compression=True)["ZDITHER0"] != seed


@pytest.mark.parametrize("extra_cards", range(36))
def test_tile_rice_header_never_grows(tmp_path: Path, extra_cards: int) -> None:
    """Blanks only past the first tile must not grow the header over its placeholder"""
    rng = np.random.default_rng(1)
    data = rng.normal(0, 1e-3, (3, 40, 48)).astype(np.float32)
    data[-1, -8:] = np.nan
    header = fits.Header()
    for i in range(extra_cards):
        header.add_comment(f"padding {i}")
    fits.PrimaryHDU(data, header).writeto(tmp_path / "cube.fits")

    compressed = compress_cube(tmp_path / "cube.fits", method="tile-rice")

    got = fits.getdata(compressed, 1)
    np.testing.assert_array_equal(np.isnan(got), np.isnan(data))
    finite = np.isfinite(data)
    assert np.std(got[finite] - data[finite]) < 1e-3 / 4 / np.sqrt(12) * 1.2
