from __future__ import annotations


class FITSCubeException(Exception):
    """Base container for FITSCube exceptions"""


class TargetAxisMissingException(FITSCubeException):
    """The target axis is missing in fits cube"""


class ChannelMissingException(FITSCubeException):
    """Raised when a channel can not be accessed"""
