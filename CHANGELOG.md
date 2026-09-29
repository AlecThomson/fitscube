# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- The `BEAMS` table `POL` column is now the 0-based index along the cube's
  Stokes axis (always 0 for a single-Stokes cube) rather than being derived
  from the Stokes code, which made CARTA refuse to open the file

## [0.2.2] - 2023-02-28

### Added

- `-o` argument to `stokescube`
- Internal API using NamedTuples to track outputs

### Fixed

- Fixed FITS index bug

## [0.2.1] - 2023-01-21

### Fixed

- Removed incorrect argument to main function in `fitscube`
