"""Recording-only lifecycle wrapper for native Mercury size refinement."""
import ctypes
from pathlib import Path
import numpy as np

class NativeSizeRefinement:
    def __init__(self,dll):
        self.lib=ctypes.CDLL(str(Path(dll).resolve()));self.fp=ctypes.POINTER(ctypes.c_float);self.ip=ctypes.POINTER(ctypes.c_int)
        self.lib.fh_scale_create.argtypes=[];self.lib.fh_scale_create.restype=ctypes.c_void_p
        self.lib.fh_scale_destroy.argtypes=[ctypes.c_void_p];self.lib.fh_scale_destroy.restype=None
        self.lib.fh_scale_pre.argtypes=[ctypes.c_void_p,self.ip,self.ip,self.fp];self.lib.fh_scale_pre.restype=ctypes.c_int
        self.lib.fh_scale_post.argtypes=[ctypes.c_void_p,ctypes.c_float,ctypes.c_float,self.fp];self.lib.fh_scale_post.restype=ctypes.c_int
        self.lib.fh_scale_finish.argtypes=[ctypes.c_void_p,self.fp];self.lib.fh_scale_finish.restype=ctypes.c_int
        self.handle=self.lib.fh_scale_create();self.open=False;self.posts=0
        if not self.handle:raise ValueError('Native scale creation failed')
    def close(self):
        if self.handle:self.lib.fh_scale_destroy(self.handle);self.handle=None
    def __enter__(self):return self
    def __exit__(self,*args):self.close()
    @staticmethod
    def result(output):return dict(target_size=float(output[0]),size_error=float(output[1]),optimize_size=bool(output[2]),schedule_x=float(output[3]),schedule_y=float(output[4]))
    def pre(self,found,seen):
        if not self.handle or self.open:raise ValueError('Invalid native scale frame lifecycle')
        found=np.ascontiguousarray(found,dtype=np.int32);seen=np.ascontiguousarray(seen,dtype=np.int32)
        if found.shape!=(2,2) or seen.shape!=(2,):raise ValueError('Expected2views x2hands and2seen flags')
        out=np.zeros(5,np.float32)
        if self.lib.fh_scale_pre(self.handle,found.ctypes.data_as(self.ip),seen.ctypes.data_as(self.ip),out.ctypes.data_as(self.fp)):raise ValueError('Native scale pre rejected')
        self.open=True;self.posts=0;return self.result(out)
    def post(self,size,error,confidence):
        if not self.handle or not self.open or self.posts>=2:raise ValueError('Invalid native scale post lifecycle')
        confidence=np.ascontiguousarray(confidence,dtype=np.float32)
        if confidence.shape!=(2,21):raise ValueError('Expected2views x21joint confidences')
        if self.lib.fh_scale_post(self.handle,float(size),float(error),confidence.ctypes.data_as(self.fp)):raise ValueError('Native scale post rejected')
        self.posts+=1
    def finish(self):
        if not self.handle or not self.open:raise ValueError('No open native scale frame')
        out=np.zeros(5,np.float32)
        if self.lib.fh_scale_finish(self.handle,out.ctypes.data_as(self.fp)):raise ValueError('Native scale finish rejected')
        self.open=False;return self.result(out)
