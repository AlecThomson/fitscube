from __future__ import annotations

import importlib.metadata

import fitscube as m


def test_version():
    assert importlib.metadata.version("fitscube") == m.__version__
