"""Causal, bounded identity-evidence scheduling probe. No pose or image reuse.

Only observation metadata and past independent image-side scores gate reuse.
Defaults are experimental and never enabled by the accepted/live launcher.
"""
import numpy as np

class IdentitySchedule:
 def __init__(self,max_age_ns=70_000_000):self.max_age_ns=max_age_ns;self.clear()
 def clear(self):self.previous_stamp=None;self.anchor=None
 @staticmethod
 def eligible_geometry(views):
  for i,a in enumerate(views):
   p=np.asarray(a['points'],dtype=float)
   if p.shape!=(21,2) or not np.isfinite(p).all():return False
   for b in views[i+1:]:
    if a['camera']!=b['camera']:continue
    q=np.asarray(b['points'],dtype=float)
    if q.shape!=(21,2) or not np.isfinite(q).all():return False
    # Expanded boxes intersect: crossing/contact ambiguity always gets fresh evidence.
    if np.all(np.maximum(p.min(0)-8,q.min(0)-8)<=np.minimum(p.max(0)+8,q.max(0)+8)):return False
  return True
 def decide(self,observation):
  stamp=int(observation['raw_ns']);views=[v for v in observation['views'] if v['presence']>=.5]
  if self.previous_stamp is not None and not 0<stamp-self.previous_stamp<=100_000_000:self.clear()
  self.previous_stamp=stamp
  if not views:return None,'no_views'
  if not self.eligible_geometry(views):return None,'overlap_or_invalid_geometry'
  if self.anchor is None or not 0<stamp-self.anchor['stamp']<=self.max_age_ns:return None,'expired_or_missing_evidence'
  keys={(v['camera'],v['id']) for v in views}
  if len(keys)!=len(views) or keys!=set(self.anchor['views']):return None,'changed_views'
  reused=[]
  for v in views:
   old=self.anchor['views'][v['camera'],v['id']];q=old['right'];raw=float(v['right']);p=np.asarray(v['points']);prior=old['points']
   if not old['agreement'] or max(q,1-q)<.8:return None,'weak_image_evidence'
   if max(raw,1-raw)<.8 or (raw>=.5)!=(q>=.5):return None,'detector_disagrees'
   size=max(25.,float(np.linalg.norm(prior[0]-prior[9])))
   if np.max(np.linalg.norm(p[[0,5,9,13,17]]-prior[[0,5,9,13,17]],axis=1))>.25*size:return None,'palm_motion'
   reused.append(dict(v,raw_right=raw,right=q))
  return dict(observation,views=reused),'reuse'
 def observe(self,observation,rows):
  evidence={(r['camera'],r['source_id']):r for r in rows};saved={}
  for v in observation['views']:
   k=(v['camera'],v['id']);r=evidence.get(k)
   if v['presence']>=.5 and r is not None:saved[k]={'right':r['right'],'agreement':r['agreement'],'points':np.asarray(v['points']).copy()}
  self.anchor={'stamp':int(observation['raw_ns']),'views':saved}
