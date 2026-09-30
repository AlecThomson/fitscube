# Regression tests for the cube-scrambling and blank-channel bugs
from __future__ import annotations

from pathlib import Path
from typing import Any

import astropy.units as u
import numpy as np
import pytest
from astropy.io import fits
from fitscube.bounding_box import get_common_bounding_box
from fitscube.combine_fits import (
    check_matching_shapes,
    check_matching_stokes,
    combine_fits,
)
from fitscube.exceptions import (
    AxisMismatchException,
    AxisOrderException,
    IrregularSpacingException,
    ShapeMismatchException,
    StokesMismatchException,
)


def make_plane(
    path: Path,
    spec: u.Quantity,
    shape: tuple[int, ...] = (1, 1, 8, 8),
    value: float = 1.0,
    pol_outer: bool = False,
    beam: float | None = None,
    dtype: Any = np.float32,
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
    if beam is not None:
        header["BMAJ"] = beam
        header["BMIN"] = beam / 2
        header["BPA"] = 0.0
    fits.PrimaryHDU(np.full(shape, value, dtype=dtype), header=header).writeto(
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


def _drop_stokes_axis(path: Path) -> Path:
    """Rewrite a (FREQ, STOKES, DEC, RA) plane from make_plane without its Stokes axis"""
    data, header = fits.getdata(path, header=True)
    for key in ("CTYPE", "CRPIX", "CRVAL", "CDELT", "CUNIT"):
        if f"{key}4" in header:
            header[f"{key}3"] = header[f"{key}4"]
            del header[f"{key}4"]
    fits.PrimaryHDU(data[:, 0], header=header).writeto(path, overwrite=True)
    return path


def test_check_matching_stokes(file_list: list[Path]) -> None:
    for path in file_list:
        fits.setval(path, "CRVAL3", value=3.0)  # Stokes U
    assert check_matching_stokes(file_list=file_list) == (3,)


def test_check_matching_stokes_compares_codes_not_keywords(
    file_list: list[Path],
) -> None:
    """CRVAL3=2 at CRPIX3=2 is still Stokes I at the (only) first pixel"""
    fits.setval(file_list[-1], "CRVAL3", value=2.0)
    fits.setval(file_list[-1], "CRPIX3", value=2.0)
    assert check_matching_stokes(file_list=file_list) == (1,)


def test_check_matching_stokes_no_stokes_axis(
    tmp_path: Path, specs: u.Quantity
) -> None:
    file_list = [
        _drop_stokes_axis(make_plane(tmp_path / f"plane_{i}.fits", spec))
        for i, spec in enumerate(specs)
    ]
    assert check_matching_stokes(file_list=file_list) is None


def test_mismatched_stokes_raise(tmp_path: Path, file_list: list[Path]) -> None:
    """A Q plane among I planes would be mislabelled as I in the cube"""
    fits.setval(file_list[2], "CRVAL3", value=2.0)  # Stokes Q
    out_cube = tmp_path / "cube.fits"

    with pytest.raises(StokesMismatchException, match=file_list[2].name):
        combine_fits(file_list=file_list, out_cube=out_cube, overwrite=True)
    assert not out_cube.exists()


def test_missing_stokes_axis_raises(tmp_path: Path, file_list: list[Path]) -> None:
    """An input without the Stokes axis the others have cannot be mixed in"""
    _drop_stokes_axis(file_list[1])
    out_cube = tmp_path / "cube.fits"

    with pytest.raises(AxisMismatchException, match=file_list[1].name):
        combine_fits(file_list=file_list, out_cube=out_cube, overwrite=True)
    assert not out_cube.exists()


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


def test_beam_table_follows_blank_channels(tmp_path: Path, specs: u.Quantity) -> None:
    """The beam table used to be one row per file, so beams slid past the gap."""
    gapped = [specs[0], specs[2], specs[3]]
    file_list = [
        make_plane(
            tmp_path / f"gap_{i}.fits", spec, value=float(i), beam=1e-3 * (i + 1)
        )
        for i, spec in enumerate(gapped)
    ]

    out_cube = tmp_path / "cube.fits"
    combine_fits(
        file_list=file_list, out_cube=out_cube, create_blanks=True, overwrite=True
    )

    with fits.open(out_cube) as hdu_list:
        assert hdu_list[0].header["NAXIS4"] == 4
        beam_table = hdu_list["BEAMS"]
        assert beam_table.header["NCHAN"] == 4
        assert np.array_equal(beam_table.data["CHAN"], np.arange(4))
        majors = beam_table.data["BMAJ"]

    tiny = np.finfo(np.float32).tiny
    expected = (1e-3 * u.deg).to(u.arcsec).value
    assert np.isclose(majors[0], expected)
    assert np.isclose(majors[1], tiny)  # the blank channel
    assert np.isclose(majors[2], 2 * expected)
    assert np.isclose(majors[3], 3 * expected)


def test_unsorted_input_with_blanks(tmp_path: Path, specs: u.Quantity) -> None:
    """even_spacing builds the grid from the end points, so it must sort first."""
    gapped = [specs[3], specs[0], specs[2]]
    file_list = [
        make_plane(tmp_path / f"unsorted_{i}.fits", spec, value=spec.to(u.Hz).value)
        for i, spec in enumerate(gapped)
    ]

    out_cube = tmp_path / "cube.fits"
    out_specs = combine_fits(
        file_list=file_list, out_cube=out_cube, create_blanks=True, overwrite=True
    )

    assert np.allclose(out_specs.to(u.Hz).value, specs.to(u.Hz).value)
    cube = fits.getdata(out_cube)
    assert np.isnan(cube[1]).all()
    for chan in (0, 2, 3):
        assert np.allclose(cube[chan], specs[chan].to(u.Hz).value)


def test_spec_list_is_used(tmp_path: Path, file_list: list[Path]) -> None:
    """spec_list was validated and then ignored."""
    spec_list = [2e9, 2.1e9, 2.2e9, 2.3e9]
    out_specs = combine_fits(
        file_list=file_list,
        out_cube=tmp_path / "cube.fits",
        spec_list=spec_list,
        overwrite=True,
    )

    assert np.allclose(out_specs.to(u.Hz).value, spec_list)


def test_spec_file_is_used(tmp_path: Path, file_list: list[Path]) -> None:
    spec_file = tmp_path / "specs.txt"
    np.savetxt(spec_file, [2e9, 2.1e9, 2.2e9, 2.3e9])
    out_specs = combine_fits(
        file_list=file_list,
        out_cube=tmp_path / "cube.fits",
        spec_file=spec_file,
        overwrite=True,
    )

    assert np.allclose(out_specs.to(u.Hz).value, np.loadtxt(spec_file))


def test_irregular_spacing_never_drops_inputs(tmp_path: Path) -> None:
    """A grid that cannot hold every input used to write a cube missing channels."""
    rng = np.random.default_rng(0)
    irregular = np.sort(rng.uniform(1e9, 2e9, 12)) * u.Hz
    file_list = [
        make_plane(tmp_path / f"irregular_{i}.fits", spec, value=float(i))
        for i, spec in enumerate(irregular)
    ]

    with pytest.raises(IrregularSpacingException, match="would drop inputs"):
        combine_fits(
            file_list=file_list,
            out_cube=tmp_path / "cube.fits",
            create_blanks=True,
            overwrite=True,
        )

    # Without blanks the irregular axis is kept and every input is written
    out_specs = combine_fits(
        file_list=file_list, out_cube=tmp_path / "cube.fits", overwrite=True
    )
    assert np.allclose(out_specs.to(u.Hz).value, irregular.to(u.Hz).value)
    cube = fits.getdata(tmp_path / "cube.fits")
    assert np.array_equal(cube[:, 0, 0, 0], np.arange(len(file_list)))


def test_duplicate_frequencies_never_drop_inputs(
    tmp_path: Path, specs: u.Quantity
) -> None:
    """Two inputs collapsing onto one grid point would lose one of them."""
    duplicated = [specs[0], specs[0], specs[2]]
    file_list = [
        make_plane(tmp_path / f"duplicate_{i}.fits", spec, value=float(i))
        for i, spec in enumerate(duplicated)
    ]

    with pytest.raises(IrregularSpacingException, match="would drop inputs"):
        combine_fits(
            file_list=file_list,
            out_cube=tmp_path / "cube.fits",
            create_blanks=True,
            overwrite=True,
        )


def test_integer_input_is_not_cast_to_float(tmp_path: Path, specs: u.Quantity) -> None:
    """Integer BITPIX used to raise a KeyError from the float-only dtype map."""
    file_list = [
        make_plane(tmp_path / f"int_{i}.fits", spec, value=float(i), dtype=np.int16)
        for i, spec in enumerate(specs)
    ]

    out_cube = tmp_path / "cube.fits"
    combine_fits(file_list=file_list, out_cube=out_cube, overwrite=True)

    assert fits.getheader(out_cube)["BITPIX"] == 16
    cube = fits.getdata(out_cube)
    assert np.array_equal(cube[:, 0, 0, 0], np.arange(len(file_list)))


def test_blanks_need_float_output(tmp_path: Path, specs: u.Quantity) -> None:
    """NaN blanks cannot be stored in an integer cube."""
    gapped = [specs[0], specs[2], specs[3]]
    file_list = [
        make_plane(tmp_path / f"int_gap_{i}.fits", spec, dtype=np.int16)
        for i, spec in enumerate(gapped)
    ]

    with pytest.raises(ValueError, match="float_length"):
        combine_fits(
            file_list=file_list,
            out_cube=tmp_path / "cube.fits",
            create_blanks=True,
            overwrite=True,
        )

    combine_fits(
        file_list=file_list,
        out_cube=tmp_path / "cube.fits",
        create_blanks=True,
        overwrite=True,
        float_length=32,
    )
    assert np.isnan(fits.getdata(tmp_path / "cube.fits")[1]).all()


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
