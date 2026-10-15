"""Truth-only initialization diagnostic; never imported by recorded/live runners."""
import ctypes
import numpy as np


def seed_known_synthetic_hand(solver,wrist,angles,size=.09):
    if not solver.handle or not solver.first:raise ValueError('Synthetic seed requires a fresh solver')
    wrist=np.ascontiguousarray(wrist,dtype=np.float32);angles=np.ascontiguousarray(angles,dtype=np.float32)
    if wrist.shape!=(7,) or angles.shape!=(4,3):raise ValueError('Invalid synthetic state shape')
    function=solver.lib.fh_solver_seed_synthetic
    function.argtypes=[ctypes.c_void_p,solver.pointer,solver.pointer,ctypes.c_float];function.restype=ctypes.c_int
    if function(solver.handle,wrist.ctypes.data_as(solver.pointer),angles.ctypes.data_as(solver.pointer),size):raise ValueError('Synthetic seed rejected')
    # Suppress reset/hand_init_guess; native first_frame remains true so the first
    # fit has no temporal penalty, just like the original cold-initialized arm.
    solver.first=False
