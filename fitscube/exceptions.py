class FITSCubeException(Exception):
    """Base container for FITSCube exceptions"""
    
class FREQMissingException(FITSCubeException):
    """Missing FREQ axis in fits cube"""