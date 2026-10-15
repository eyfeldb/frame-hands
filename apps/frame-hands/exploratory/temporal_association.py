"""Experimental causal per-view association. No missing hand or pose synthesis.

The default tracker is unchanged. Anatomical slots follow camera observations,
not an inherited detector ID; classifier evidence and stereo geometry constrain
acquisition. Short spatial history handles brief contradictory side scores.
"""
import copy
import itertools
import numpy as np
from frame_data import NAMES, unproject

PALM=np.array([0,5,9,13,17])

class TemporalAssociation:
    def __init__(self,cameras,retain_depth=False,reattach_prior=False):
        self.cameras=cameras
        self.retain_depth=retain_depth
        self.reattach_prior=reattach_prior
        self.tracks={}
        self.generation=1000
        self.last_debug={}
        self.ambiguous=False
        self.resolve_streak=0
        self.last_stamp=None

    def reset(self):
        # Keep generations monotonic: old model/display memory must not reconnect.
        self.tracks={}
        self.ambiguous=False
        self.resolve_streak=0
        self.last_stamp=None
        self.last_debug={}

    def _feature(self,v,camera):
        points=np.asarray(v['points'])[PALM]
        center=points.mean(0)
        scale=max(25.,float(np.linalg.norm(points[0]-points[2])))
        rays=unproject(camera,points)@camera.camera_to_world_xf[:3,:3].T
        return dict(view=v,center=center,scale=scale,rays=rays,origin=camera.camera_to_world_xf[:3,3])

    def _pair(self,a,b):
        r=a['rays'];s=b['rays'];delta=b['origin']-a['origin']
        dot=(r*s).sum(1);denom=1-dot*dot
        if np.any(denom<1e-6):return None
        ra=r@delta;sb=s@delta
        ta=(ra-dot*sb)/denom;tb=(dot*ra-sb)/denom
        if np.median(ta)<100 or np.median(tb)<100 or max(np.median(ta),np.median(tb))>2000:return None
        pa=a['origin']+r*ta[:,None];pb=b['origin']+s*tb[:,None]
        angle=np.linalg.norm(pa-pb,axis=1)/np.maximum(100,(ta+tb)*.5)
        # A poor correspondence must not become a confident stereo input.
        error=float(np.median(angle))
        if error>.08:return None
        return dict(point=((pa+pb)*.5).mean(0),error=error)

    def _cost(self,f,side,now,cam):
        v=f['view'];q=float(v['right'] if side else 1-v['right'])
        track=self.tracks.get(side)
        recent=track is not None and now-track['time']<=.65
        if not recent:
            return -np.log(max(.001,q)) if q>=.65 else float('inf')
        previous=track['views'].get(v['camera'])
        spatial=track['world'] is not None and (not self.retain_depth or now-track.get('depth_time',-1e10)<=.65)
        if previous is None or now-previous['time']>.65:
            if not self.retain_depth or not spatial:
                # A stale/new camera view lacks geometric continuity. Its label
                # alone must not beat a zero-motion established observation just
                # because that observation's instantaneous classifier flips.
                prior=.30*(-np.log(.01)) if self.reattach_prior else 0.
                return prior-np.log(max(.001,q)) if q>=.6 else float('inf')
            dt=now-track['time'];scale=f['scale'];target=None
        else:
            dt=now-previous['time'];scale=max(f['scale'],previous['scale'])
            target=previous['point']+previous['velocity']*min(dt,.10)
        if spatial:
            world=track['world']+track['velocity']*min(now-track['time'],.10)
            eye=cam.world_to_eye(world)
            if eye[2]>0:target=cam.eye_to_window(eye)
        if target is None:return float('inf')
        distance=float(np.linalg.norm(f['center']-target))/scale
        allowed=1.0+6*dt
        if distance>allowed:return float('inf')
        # Identity evidence is not an instant veto on a continuous observation.
        return 1.6*distance*distance+.30*(-np.log(max(.01,q)))

    def _prepare(self,observation,world):
        if world is None or np.shape(world)!=(4,4) or not np.isfinite(world).all():
            self.reset()
            self.last_debug=dict(assignments=[],reason='Invalid capture context')
            return None
        stamp=observation['raw_ns']
        if not isinstance(stamp,(int,np.integer)) or stamp<0:
            raise ValueError('Invalid capture timestamp')
        if self.last_stamp is not None and stamp<=self.last_stamp:
            self.last_debug=dict(assignments=[],reason='Non-increasing capture timestamp')
            return None
        self.last_stamp=stamp
        now=stamp/1e9
        self.tracks={k:v for k,v in self.tracks.items() if 0<=now-v['time']<=.65}
        if not self.tracks:
            self.ambiguous=False
            self.resolve_streak=0
        cams={n:c.copy(camera_to_world_xf=world@c.camera_to_world_xf) for n,c in self.cameras.items()}
        features={n:[] for n in NAMES}
        for v in observation['views']:
            points=np.asarray(v['points'])
            valid=(points.shape==(21,2) and np.isfinite(points).all()
                and np.isfinite(v['presence']) and .5<=v['presence']<=1
                and np.isfinite(v['right']) and 0<=v['right']<=1)
            if v['camera'] in features and valid:
                features[v['camera']].append(self._feature(v,cams[v['camera']]))
        if any(len(v)>4 for v in features.values()):raise ValueError('Unexpected candidate count')
        options=[]
        for name in NAMES:
            rows=features[name];current=[]
            for left,right in itertools.product(range(-1,len(rows)),repeat=2):
                if left>=0 and left==right:continue
                cost=0.
                for side,index in enumerate((left,right)):
                    cost+=1.8 if index<0 else self._cost(rows[index],side,now,cams[name])
                if np.isfinite(cost):current.append(((left,right),cost))
            options.append(current)
        solutions=[]
        for (a,ca),(b,cb) in itertools.product(*options):
            score=ca+cb;pairs={};valid=True
            for side in (0,1):
                if a[side]>=0 and b[side]>=0:
                    pair=self._pair(features[NAMES[0]][a[side]],features[NAMES[1]][b[side]])
                    if pair is None:valid=False;break
                    score+=(pair['error']/.025)**2
                    pairs[side]=pair
            if valid:solutions.append((score,a,b,pairs))
        solutions.sort(key=lambda s:s[0])
        return now,features,solutions

    def _apply(self,now,features,solution,margin):
        cost,a,b,pairs=solution
        selections={side:[features[n][choice[side]] for n,choice in zip(NAMES,(a,b)) if choice[side]>=0] for side in (0,1)}
        close=False
        for name in NAMES:
            rows=features[name]
            for f,g in itertools.combinations(rows,2):
                if np.linalg.norm(f['center']-g['center'])<1.8*max(f['scale'],g['scale']):close=True
        if close:
            self.ambiguous=True;self.resolve_streak=0
        if self.ambiguous and not close and all(len(selections[side])==2 for side in (0,1)):
            opposite=all(all((f['view']['right'] if side==0 else 1-f['view']['right'])>.75 for f in selections[side]) for side in (0,1))
            aligned=all(all((f['view']['right'] if side==1 else 1-f['view']['right'])>.75 for f in selections[side]) for side in (0,1))
            self.resolve_streak=(self.resolve_streak+1) if opposite else (self.resolve_streak-1) if aligned else 0
            if self.resolve_streak>=3 and set(self.tracks)=={0,1}:
                self.tracks={0:self.tracks[1],1:self.tracks[0]}
                a=(a[1],a[0]);b=(b[1],b[0]);pairs={1-side:pair for side,pair in pairs.items()}
                self.ambiguous=False;self.resolve_streak=0
            elif self.resolve_streak<=-5:
                self.ambiguous=False;self.resolve_streak=0
        assigned={};candidates={};scores={};debug=[]
        for side in (0,1):
            selected=[features[n][choice[side]] for n,choice in zip(NAMES,(a,b)) if choice[side]>=0]
            if not selected:continue
            old=self.tracks.get(side)
            if old is None:
                self.generation+=1
                old=dict(id=self.generation,views={},world=None,velocity=np.zeros(3),time=now)
            views=dict(old['views'])
            for f in selected:
                name=f['view']['camera'];prev=views.get(name);vel=np.zeros(2)
                if prev and 0<now-prev['time']<=.15:
                    vel=(f['center']-prev['point'])/(now-prev['time'])
                    speed=np.linalg.norm(vel)
                    if speed>2500:vel*=2500/speed
                    vel=.5*prev['velocity']+.5*vel
                views[name]=dict(point=f['center'],velocity=vel,scale=f['scale'],time=now)
            point=pairs.get(side,{}).get('point')
            depth_time=now if point is not None else old.get('depth_time',-1e10)
            if self.retain_depth and point is None and len(selected)==1 and old['world'] is not None and now-depth_time<=.65:
                # Association-only bearing update. The previous stereo range is
                # uncertain; never feed this point to the model or renderer.
                f=selected[0];ray=f['rays'].mean(0);ray=ray/np.linalg.norm(ray)
                distance=float(np.dot(old['world']-f['origin'],ray))
                if 100<=distance<=2000:point=f['origin']+ray*distance
            velocity=np.zeros(3)
            if point is not None and old['world'] is not None and 0<now-old['time']<=.15:
                velocity=(point-old['world'])/(now-old['time'])
                speed=np.linalg.norm(velocity)
                if speed>2000:velocity*=2000/speed
                velocity=.5*old['velocity']+.5*velocity
            self.tracks[side]=dict(id=old['id'],views=views,world=point,velocity=velocity,time=now,depth_time=depth_time)
            ident=old['id'];assigned[side]=ident
            candidates[ident]=[f['view'] for f in selected]
            scores[ident]=float(np.average([f['view']['right'] for f in selected],weights=[f['view']['presence'] for f in selected]))
            debug.append(dict(side=side,track=ident,views=[dict(camera=f['view']['camera'],source_id=f['view']['id']) for f in selected],stereo_error=pairs.get(side,{}).get('error')))
        self.last_debug=dict(assignments=debug,cost=cost,margin=margin)
        return assigned,candidates,scores

    def assign(self,observation,world):
        prepared=self._prepare(observation,world)
        if prepared is None:return {},{},{}
        now,features,solutions=prepared
        return self._apply(now,features,solutions[0],solutions[1][0]-solutions[0][0] if len(solutions)>1 else None)

    def branches(self,observation,world,limit):
        prepared=self._prepare(observation,world)
        if prepared is None:return [(0.,({}, {}, {}),self)]
        now,features,solutions=prepared
        margin=solutions[1][0]-solutions[0][0] if len(solutions)>1 else None
        children=[]
        for solution in solutions[:limit]:
            child=copy.copy(self)
            child.tracks=copy.deepcopy(self.tracks)
            result=child._apply(now,features,solution,margin)
            children.append((float(solution[0]),result,child))
        return children
