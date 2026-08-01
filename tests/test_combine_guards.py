# Regression tests for the cube-scrambling and blank-channel bugs
from __future__ import annotations

from pathlib import Path
from typing import Any

import astropy.units as u
import numpy as np
import pytest
from astropy.io import fits
from fitscube.bounding_box import get_common_bounding_box
from fitscube.combine_fits import check_matching_shapes, combine_fits
from fitscube.exceptions import AxisOrderException, ShapeMismatchException


def make_plane(
    path: Path,
    spec: u.Quantity,
    shape: tuple[int, ...] = (1, 1, 8, 8),
    value: float = 1.0,
    pol_outer: bool = False,
) -> Path:
    """Write a single-channel image with a FREQ axis at ``spec``.

    By default the axes are (FREQ, STOKES, DEC, RA), the usual ASKAP ordering.
    With ``pol_outer`` they are (STOKES, FREQ, DEC, RA), which puts a
    non-degenerate STOKES axis above FREQ in the output cube.
    """
    header = fits.Header()
    header["CTYPE1"] = "RA---SIN"
    header["CRPIX1"] = 1.0
    header["CRVAL1"] = 0.0
    header["CDELT1"] = -1e-3
    header["CUNIT1"] = "deg"
    header["CTYPE2"] = "DEC--SIN"
    header["CRPIX2"] = 1.0
    header["CRVAL2"] = 0.0
    header["CDELT2"] = 1e-3
    header["CUNIT2"] = "deg"
    spec_axis, pol_axis = (4, 3) if not pol_outer else (3, 4)
    header[f"CTYPE{spec_axis}"] = "FREQ"
    header[f"CRPIX{spec_axis}"] = 1.0
    header[f"CRVAL{spec_axis}"] = spec.to(u.Hz).value
    header[f"CDELT{spec_axis}"] = 1e6
    header[f"CUNIT{spec_axis}"] = "Hz"
    header[f"CTYPE{pol_axis}"] = "STOKES"
    header[f"CRPIX{pol_axis}"] = 1.0
    header[f"CRVAL{pol_axis}"] = 1.0
    header[f"CDELT{pol_axis}"] = 1.0
    fits.PrimaryHDU(np.full(shape, value, dtype=np.float32), header=header).writeto(
        path, overwrite=True
    )
    return path


@pytest.fixture
def specs() -> u.Quantity:
    return np.arange(4) * 1e6 * u.Hz + 1e9 * u.Hz


@pytest.fixture
def file_list(tmp_path: Path, specs: u.Quantity) -> list[Path]:
    return [
        make_plane(tmp_path / f"plane_{i}.fits", spec, value=float(i))
        for i, spec in enumerate(specs)
    ]


def test_mismatched_shapes_raise(
    tmp_path: Path, file_list: list[Path], specs: u.Quantity
) -> None:
    """Differing NAXIS1/NAXIS2 used to slide the planes against each other."""
    odd_one_out = make_plane(
        tmp_path / "plane_odd.fits", specs[-1] + 1e6 * u.Hz, shape=(1, 1, 6, 6)
    )

    with pytest.raises(ShapeMismatchException, match="plane_odd"):
        combine_fits(
            file_list=[*file_list, odd_one_out],
            out_cube=tmp_path / "cube.fits",
            overwrite=True,
        )


def test_check_matching_shapes(file_list: list[Path]) -> None:
    assert check_matching_shapes(file_list=file_list) == (8, 8)


def test_spectral_axis_must_be_slowest(tmp_path: Path, specs: u.Quantity) -> None:
    """A non-degenerate axis above FREQ breaks the per-plane seek."""
    file_list = [
        make_plane(tmp_path / f"pol_{i}.fits", spec, shape=(2, 1, 8, 8), pol_outer=True)
        for i, spec in enumerate(specs)
    ]

    with pytest.raises(AxisOrderException, match="NAXIS4"):
        combine_fits(
            file_list=file_list, out_cube=tmp_path / "cube.fits", overwrite=True
        )


def test_small_cube_keeps_input_precision(
    tmp_path: Path, file_list: list[Path]
) -> None:
    """The small-cube path used to promote the output to float64."""
    out_cube = tmp_path / "cube.fits"
    combine_fits(file_list=file_list, out_cube=out_cube, overwrite=True)

    header = fits.getheader(out_cube)
    assert header["BITPIX"] == -32
    cube = fits.getdata(out_cube)
    assert cube.dtype.itemsize == 4
    assert cube.shape == (len(file_list), 1, 8, 8)
    for chan in range(len(file_list)):
        assert np.allclose(cube[chan], fits.getdata(file_list[chan]))


def test_float_length_is_respected(tmp_path: Path, file_list: list[Path]) -> None:
    out_cube = tmp_path / "cube.fits"
    combine_fits(
        file_list=file_list, out_cube=out_cube, overwrite=True, float_length=64
    )

    assert fits.getheader(out_cube)["BITPIX"] == -64
    assert fits.getdata(out_cube).dtype.itemsize == 8


def test_blank_channels_are_created(
    tmp_path: Path, specs: u.Quantity, monkeypatch: pytest.MonkeyPatch
) -> None:
    """create_blanks used to hit `fits.getdata(..., memamp=False)`.

    Some astropy versions swallow the unknown keyword, so reject it explicitly
    here the way a strict astropy does.
    """
    real_getdata = fits.getdata

    def strict_getdata(*args: Any, **kwargs: Any) -> Any:
        unexpected = set(kwargs) - {
            "filename",
            "header",
            "memmap",
            "lazy_load_hdus",
            "ext",
        }
        if unexpected:
            msg = f"getdata() got unexpected keyword arguments {unexpected}"
            raise TypeError(msg)
        return real_getdata(*args, **kwargs)

    monkeypatch.setattr(fits, "getdata", strict_getdata)

    # Drop the second channel to leave a gap in the frequency grid
    gapped = [specs[0], specs[2], specs[3]]
    file_list = [
        make_plane(tmp_path / f"gap_{i}.fits", spec, value=float(i))
        for i, spec in enumerate(gapped)
    ]

    out_cube = tmp_path / "cube.fits"
    out_specs = combine_fits(
        file_list=file_list,
        out_cube=out_cube,
        create_blanks=True,
        overwrite=True,
    )

    assert len(out_specs) == 4
    cube = fits.getdata(out_cube)
    assert np.isnan(cube[1]).all()
    assert np.allclose(cube[0], 0.0)
    assert np.allclose(cube[2], 1.0)
    assert np.allclose(cube[3], 2.0)


def test_bounding_box_can_be_supplied(tmp_path: Path, specs: u.Quantity) -> None:
    """A caller-supplied box is used as is, so two cubes can share a grid."""
    # Images blanked to different extents, as per-channel linmos mosaics are
    images = []
    weights = []
    for i, spec in enumerate(specs):
        image = make_plane(tmp_path / f"image_{i}.fits", spec, value=float(i))
        with fits.open(image, mode="update") as hdu_list:
            hdu_list[0].data[..., : i + 1, :] = np.nan
        images.append(image)
        weights.append(make_plane(tmp_path / f"weight_{i}.fits", spec, value=1.0))

    common_box = get_common_bounding_box(file_list=images)
    assert common_box.x_span == 8 - 1  # first row blanked in every image

    image_cube = tmp_path / "image_cube.fits"
    weight_cube = tmp_path / "weight_cube.fits"
    for file_list, out_cube in ((images, image_cube), (weights, weight_cube)):
        combine_fits(
            file_list=file_list,
            out_cube=out_cube,
            overwrite=True,
            bounding_box=common_box,
        )

    image_header = fits.getheader(image_cube)
    weight_header = fits.getheader(weight_cube)
    assert fits.getdata(image_cube).shape == fits.getdata(weight_cube).shape
    for key in ("NAXIS1", "NAXIS2", "CRPIX1", "CRPIX2"):
        assert image_header[key] == weight_header[key]
    assert image_header["NAXIS2"] == common_box.x_span

    # The weights alone would have given the full, untrimmed grid
    assert get_common_bounding_box(file_list=weights).x_span == 8
