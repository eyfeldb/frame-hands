"""Offline current-observation factorial wrist/finger startup adapter."""
import ctypes
import numpy as np
from mercury_native_solver import pack_views
from mercury_stereo_init import REASONS


def initialize_observations(solver,views,stereo=False,curls=False,size=.09):
    if not solver.handle or not solver.first:raise ValueError('Observation initialization requires a fresh solver')
    packet=pack_views(views,solver.side);output=np.zeros(30,np.float32);function=solver.lib.fh_solver_seed_observations
    function.argtypes=[ctypes.c_void_p,solver.pointer,ctypes.c_float,ctypes.c_int,ctypes.c_int,solver.pointer];function.restype=ctypes.c_int
    status=function(solver.handle,packet.ctypes.data_as(solver.pointer),size,int(stereo),int(curls),output.ctypes.data_as(solver.pointer))
    if status:raise ValueError('Observation initializer rejected inputs')
    solver.first=False
    return dict(stereo_requested=bool(stereo),curls_requested=bool(curls),stereo_status=REASONS[int(output[7])] if stereo else 'not_requested',wrist_internal_xyzw_xyz=output[:7].tolist(),max_ray_gap_m=float(output[8]),palm_rms_m=float(output[9]),fingers_observed_target_angles=output[10:].reshape(4,5).tolist())
