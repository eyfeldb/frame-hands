"""Optional observation-only native stereo palm initialization; offline candidate."""
import ctypes
import numpy as np
from mercury_native_solver import pack_views

REASONS={0:'accepted',1:'missing_view',2:'low_palm_confidence',3:'parallel_rays',4:'depth_or_cheirality',5:'ray_mismatch',6:'degenerate_palm',7:'palm_shape_mismatch'}


def initialize_from_stereo(solver,views,size=.09):
    if not solver.handle or not solver.first:raise ValueError('Stereo initialization requires a fresh solver')
    packet=pack_views(views,solver.side);output=np.zeros(10,np.float32);function=solver.lib.fh_solver_seed_stereo
    function.argtypes=[ctypes.c_void_p,solver.pointer,ctypes.c_float,solver.pointer];function.restype=ctypes.c_int
    status=function(solver.handle,packet.ctypes.data_as(solver.pointer),size,output.ctypes.data_as(solver.pointer))
    if status<0 or status not in REASONS:raise ValueError('Stereo initializer rejected malformed input')
    if status==0:solver.first=False
    return dict(accepted=status==0,reason=REASONS[status],wrist_internal_xyzw_xyz=output[:7].tolist() if status==0 else None,max_ray_gap_m=float(output[7]),min_ray_sine=float(output[8]),palm_rms_m=float(output[9]))
