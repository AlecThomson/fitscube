# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- FITS tile compression (the `fpack` format, `.fits.fz`) as `--compress-method
  tile-gzip` (lossless) and `tile-rice` (lossy, `--quantize-level`). It streams a
  plane at a time across `--max-workers` threads, needs only astropy, and lets
  single planes be read without decompressing the whole cube
- `fitscube decompress` and `fitscube.compress.decompress_cube`, to turn a
  `.fits.gz` or `.fits.fz` cube back into plain FITS a plane at a time
- Cubes decompressed from a lossy file are marked `FCLOSSY = T`, and `tile-rice`
  refuses to quantize them again unless `allow_requantize` is set
- A README section on working with compressed cubes

### Changed

- Compression lives in `fitscube.compress`; `compress_cube` and `COMPRESS_METHOD`
  are still importable from `fitscube.combine_fits`

### Fixed

- `fitscube extract` names planes from a `.fits.gz` or `.fits.fz` cube as plain
  `.fits`, rather than giving an uncompressed file a compressed suffix

## [0.2.2] - 2023-02-28

### Added

- `-o` argument to `stokescube`
- Internal API using NamedTuples to track outputs

### Fixed

- Fixed FITS index bug

## [0.2.1] - 2023-01-21

### Fixed

- Removed incorrect argument to main function in `fitscube`
