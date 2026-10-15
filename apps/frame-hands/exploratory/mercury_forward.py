"""Optional synthetic native hand generator; no inference or persistent solver state."""
import ctypes
from pathlib import Path
import numpy as np


class NativeHandGenerator:
    def __init__(self,dll):
        self.lib=ctypes.CDLL(str(Path(dll).resolve()));self.pointer=ctypes.POINTER(ctypes.c_float)
        self.function=self.lib.fh_generate_hand
        self.function.argtypes=[self.pointer,self.pointer,ctypes.c_float,ctypes.c_int,self.pointer];self.function.restype=ctypes.c_int
    def generate(self,wrist,angles,side,size=.09):
        """Wrist is native internal quaternionXYZW/positionXYZ(m), angles4x3(rad).

        Angles control index/middle/ring/little proximal/PIP/DIP curls. Other
        articulation uses pinned native initial defaults. Right bone offsets
        follow native camera-X reflection; wrist orientation is not OpenXR output.
        """
        wrist=np.ascontiguousarray(wrist,dtype=np.float32);angles=np.ascontiguousarray(angles,dtype=np.float32)
        if wrist.shape!=(7,) or angles.shape!=(4,3) or side not in (0,1):raise ValueError('Invalid synthetic hand shape or side')
        output=np.zeros(182,np.float32)
        if self.function(wrist.ctypes.data_as(self.pointer),angles.ctypes.data_as(self.pointer),float(size),int(side),output.ctypes.data_as(self.pointer)):raise ValueError('Native hand generator rejected input')
        return output.reshape(26,7)
