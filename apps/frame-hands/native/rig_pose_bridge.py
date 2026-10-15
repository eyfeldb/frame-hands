"""Frame-local rig snapshot to existing viewer packet conversion; standard library only.

Head poses must come from pose_service at capture_mono_ns. No camera, model,
network, input injection, or automatic XR startup is performed by this module.
"""
import math

IDENTITY=[[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,1.,0.],[0.,0.,0.,1.]]

def finite(values,length):
 if not isinstance(values,list) or len(values)!=length or any(isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) for x in values):raise ValueError('Invalid numeric array')
 return [float(x) for x in values]

def positive_integer(x):
 if type(x) is not int or x<=0:raise ValueError('Invalid positive integer')
 return x

def head_matrix(values):
 values=finite(values,12);m=[values[i*4:i*4+4] for i in range(3)]+[[0.,0.,0.,1.]]
 for i in range(3):
  for j in range(3):
   if abs(sum(m[i][k]*m[j][k] for k in range(3))-(1. if i==j else 0.))>1e-3:raise ValueError('Nonrigid head pose')
 determinant=(m[0][0]*(m[1][1]*m[2][2]-m[1][2]*m[2][1])-m[0][1]*(m[1][0]*m[2][2]-m[1][2]*m[2][0])+m[0][2]*(m[1][0]*m[2][1]-m[1][1]*m[2][0]))
 if determinant<.999:raise ValueError('Reflected head pose')
 return m

def wrist_matrix(values):
 w,x,y,z,tx,ty,tz=finite(values,7)
 if abs(w*w+x*x+y*y+z*z-1)>1e-5:raise ValueError('Nonunit wrist quaternion')
 return [[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),tx*.001],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),ty*.001],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y),tz*.001],[0.,0.,0.,1.]]

def multiply(a,b):return [[sum(a[i][k]*b[k][j] for k in range(4)) for j in range(4)] for i in range(4)]

def profile_limits(path,side):
 rows=path.read_text().splitlines();header=rows[0].split()
 if header[:4]!=['FHRIG','1','mm','left'] or len(header)!=11 or header[-3:]!=['20','21','17']:raise ValueError('Unsupported profile')
 if header[4] not in ('generic','experimental','calibrated') or int(header[5]) not in (-1,side):raise ValueError('Wrong profile side')
 limits=[finite([float(x) for x in row.split()][6:8],2) for row in rows[1:21]]
 if len(limits)!=20 or any(lo>hi for lo,hi in limits):raise ValueError('Invalid limits')
 return limits

class RigPoseBridge:
 def __init__(self,limits):
  if len(limits)!=2 or any(len(side)!=20 for side in limits):raise ValueError('Two20angleprofiles required')
  self.limits=limits;self.last_source_sequence=0;self.last_capture=0;self.sequence=0
 def convert(self,snapshot,head,now_mono_ns,wall_s):
  if snapshot.get('version')!=3 or snapshot.get('clock')!='CLOCK_MONOTONIC' or snapshot.get('space')!='capture_head_mm':raise ValueError('Wrong native protocol/clock/space')
  seq=positive_integer(snapshot['sequence']);raw=positive_integer(snapshot['capture_raw_ns']);capture=positive_integer(snapshot['capture_mono_ns']);published=positive_integer(snapshot['published_mono_ns']);now=positive_integer(now_mono_ns)
  if seq<=self.last_source_sequence or capture<=self.last_capture:return None
  if not isinstance(wall_s,(int,float)) or not math.isfinite(wall_s):raise ValueError('Invalid publish wall time')
  self.last_source_sequence=seq;self.last_capture=capture;self.sequence+=1
  result=dict(version=2,sequence=self.sequence,capture_ns=capture,capture_raw_ns=raw,published_wall_s=wall_s,age_ms=max(0,(now-capture)/1e6),context_valid=False,head_world_m=IDENTITY,hands=[],native_source_sequence=seq,unavailable_sides=[])
  if not capture<=published<=now or now-capture>250000000:
   result['reason']='invalid_or_stale_monotonic_capture';return result
  if head.get('valid') is not True:
   result['reason']='invalid_head_pose';return result
  queried=positive_integer(head.get('queried_mono_ns'))
  if head.get('capture_mono_ns')!=capture or not capture<=queried<=now or now-queried>50000000:
   result['reason']='head_timestamp_mismatch_or_old_query';return result
  historical=head_matrix(head['capture']);result['head_world_m']=head_matrix(head['current'])
  candidates=snapshot.get('hands')
  if not isinstance(candidates,list) or len(candidates)>16:raise ValueError('Invalid source set')
  by_side={0:[],1:[]};ids=set()
  for h in candidates:
   source=positive_integer(h.get('source_id'));side=h.get('side')
   if type(side) is not int or side not in (0,1) or source in ids:raise ValueError('Invalid side or duplicated source')
   ids.add(source);by_side[side].append(h)
  for side,items in by_side.items():
   if len(items)!=1:
    result['unavailable_sides'].append(dict(side=side,reason='ambiguous_side' if items else 'absent'));continue
   h=items[0]
   if h.get('status') not in ('fitted','monocular','missing_or_ambiguous_views','insufficient'):raise ValueError('Unknown source status')
   if h['status']!='fitted':result['unavailable_sides'].append(dict(side=side,reason=h['status']));continue
   angles=finite(h.get('joint_angles'),20)
   if any(v<lo-1e-8 or v>hi+1e-8 for v,(lo,hi) in zip(angles,self.limits[side])):raise ValueError('Angles outside selected anatomy')
   world=multiply(historical,wrist_matrix(h.get('wrist_head_mm')))
   result['hands'].append(dict(side=side,source_id=h['source_id'],wrist_world_m=world,joint_angles=angles))
  result['context_valid']=True
  return result
