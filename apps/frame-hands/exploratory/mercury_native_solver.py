"""Recording-only ctypes adapter for pinned native Mercury kinematics."""
import ctypes
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

ML_TO_XR=np.array([1,2,3,4,5,7,8,9,10,12,13,14,15,17,18,19,20,22,23,24,25])
FLIP=np.diag([1.,-1.,-1.,1.])

def camera_transforms(cameras):
    transforms={}
    for name in ['slam_left','slam_right']:
        xf=np.asarray(cameras[name].camera_to_world_xf).copy();xf[:3,3]*=.001
        transforms[name]=xf@FLIP
    right_from_left=np.linalg.inv(transforms['slam_right'])@transforms['slam_left']
    pose=np.r_[Rotation.from_matrix(right_from_left[:3,:3]).as_quat(),right_from_left[:3,3]].astype(np.float32)
    return pose,transforms['slam_left']


def decode_depth(heatmap):
    data=np.clip(np.asarray(heatmap,dtype=np.float32),0,1)
    if data.shape!=(21,22) or not np.isfinite(data).all():raise ValueError('Invalid relative-depth heatmaps')
    total=data.sum(axis=1);location=(data*np.arange(22,dtype=np.float32)).sum(axis=1)/np.maximum(total,1e-5)
    valid=(total>=1e-5)&(location>=0)&(location<=21)
    depth=((location+.5)/22-.5)*3;confidence=data[np.arange(21),np.clip(location.astype(int),0,21)]
    return np.where(valid,depth,0),np.where(valid,confidence,0)


def pack_views(views,side):
    if side not in (0,1):raise ValueError('Invalid anatomical side')
    values=[]
    for name in ['slam_left','slam_right']:
        view=views[name];q=Rotation.from_matrix(np.array(view['rotation'])).as_quat()
        points=np.asarray(view['points_crop'],float)/64-1
        if points.shape!=(21,2):raise ValueError('Invalid crop landmarks')
        points[:,1]*=-1
        if side:points[:,0]*=-1
        depth,confidence=decode_depth(view['depth_heatmap']);xyconfidence=np.asarray(view['confidence_xy']);curl=np.asarray(view['curls'])
        if xyconfidence.shape!=(21,) or curl.shape!=(10,):raise ValueError('Invalid confidence/curl shapes')
        values.extend([float(view['active']),*q,float(view['stereographic_radius'])])
        values.extend(np.column_stack((points,depth,xyconfidence,confidence)).ravel())
        values.extend(np.column_stack((curl[:5],np.abs(curl[5:])+.01)).ravel())
    output=np.asarray(values,dtype=np.float32)
    if output.shape!=(242,) or not np.isfinite(output).all():raise ValueError('Invalid native observation packet')
    return output


class NativeSolver:
    def __init__(self,dll,pose,side):
        self.lib=ctypes.CDLL(str(Path(dll).resolve()));pointer=ctypes.POINTER(ctypes.c_float)
        self.lib.fh_solver_create.argtypes=[pointer,ctypes.c_int];self.lib.fh_solver_create.restype=ctypes.c_void_p
        self.lib.fh_solver_destroy.argtypes=[ctypes.c_void_p];self.lib.fh_solver_destroy.restype=None
        self.lib.fh_solver_run.argtypes=[ctypes.c_void_p,pointer,ctypes.c_int,ctypes.c_float,ctypes.c_int,ctypes.c_float,ctypes.c_float,ctypes.c_float,pointer];self.lib.fh_solver_run.restype=ctypes.c_int
        pose=np.ascontiguousarray(pose,dtype=np.float32)
        if pose.shape!=(7,):raise ValueError('Expected seven-component calibrated pose')
        self.handle=self.lib.fh_solver_create(pose.ctypes.data_as(pointer),int(side))
        if not self.handle:raise ValueError('Native solver creation failed')
        self.side=side;self.first=True;self.pointer=pointer
    def transport(self,current_from_previous):
        """Re-express history after measured camera motion; never predict a hand."""
        if not self.handle or self.first:raise ValueError('No fitted native history to transport')
        xf=np.asarray(current_from_previous,float)
        if xf.shape!=(4,4) or not np.isfinite(xf).all() or not np.allclose(xf[3],[0,0,0,1],atol=1e-8):raise ValueError('Invalid camera transport')
        if not np.allclose(xf[:3,:3].T@xf[:3,:3],np.eye(3),atol=1e-6) or abs(np.linalg.det(xf[:3,:3])-1)>1e-6:raise ValueError('Camera transport must be rigid')
        try:function=self.lib.fh_solver_transport
        except AttributeError:raise ValueError('Native DLL does not support experimental history transport')
        function.argtypes=[ctypes.c_void_p,self.pointer];function.restype=ctypes.c_int
        pose=np.r_[Rotation.from_matrix(xf[:3,:3]).as_quat(),xf[:3,3]].astype(np.float32)
        if function(self.handle,pose.ctypes.data_as(self.pointer)):raise ValueError('Native camera transport rejected')
    def history(self):
        """Diagnostic native pre-fit and last-fit wrist state; no optimizer step."""
        if not self.handle or self.first:raise ValueError('No native history')
        function=self.lib.fh_solver_history;function.argtypes=[ctypes.c_void_p,self.pointer];function.restype=ctypes.c_int
        output=np.zeros(14,np.float32)
        if function(self.handle,output.ctypes.data_as(self.pointer)):raise ValueError('Native history unavailable')
        return output.reshape(2,7)
    def fingers(self):
        """Read completed index/middle/ring/little proximal/PIP/DIP curl angles."""
        if not self.handle or self.first:raise ValueError('No fitted native finger history')
        function=self.lib.fh_solver_fingers;function.argtypes=[ctypes.c_void_p,self.pointer];function.restype=ctypes.c_int
        output=np.zeros(12,np.float32)
        if function(self.handle,output.ctypes.data_as(self.pointer)):raise ValueError('Native finger history unavailable')
        return output.reshape(4,3)
    def finger_stability(self,views):
        """Inspect native curl weighting against previous fingers; never optimize.

        Return4x10: prior angles3,sum,learned average,multiplier,left curl/variance,
        right curl/variance. Valid as an upcoming-step weight only without reset.
        """
        if not self.handle or self.first:raise ValueError('No fitted native finger history')
        function=self.lib.fh_solver_curl_diagnostics;function.argtypes=[ctypes.c_void_p,self.pointer,self.pointer];function.restype=ctypes.c_int
        packet=pack_views(views,self.side);output=np.zeros(40,np.float32)
        if function(self.handle,packet.ctypes.data_as(self.pointer),output.ctypes.data_as(self.pointer)):raise ValueError('Native curl diagnostic rejected')
        return output.reshape(4,10)
    def close(self):
        if self.handle:self.lib.fh_solver_destroy(self.handle);self.handle=None
    def __enter__(self):return self
    def __exit__(self,*args):self.close()
    def run(self,views,reset=False,smoothing=2.,optimize_size=False,target_size=.09,size_error=1.,depth_weight=.01):
        if not self.handle:raise ValueError('Solver is closed')
        packet=pack_views(views,self.side);output=np.zeros(185,np.float32)
        status=self.lib.fh_solver_run(self.handle,packet.ctypes.data_as(self.pointer),int(reset or self.first),float(smoothing),int(optimize_size),float(target_size),float(size_error),float(depth_weight),output.ctypes.data_as(self.pointer))
        if status:raise ValueError('Native solver status '+str(status))
        self.first=False;joints=output[:182].reshape(26,7)
        return dict(joints_left_camera_m=joints[:,:3].tolist(),orientations_xyzw=joints[:,3:].tolist(),hand_size_m=float(output[182]),reprojection_error=float(output[183]),native_active=bool(output[184]),passes_native_reprojection_gate=bool(output[183]<=15))
