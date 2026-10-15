"""Experimental current-ray/soft-anatomy fitting. Offline prototype, no promotion.
Units mm. Measurement covariance uses an explicit assumed pixel-noise model.
"""
import numpy as np
from audit_native_lengths import PARENTS,FINGERS

def fit(measurement,covariance,lengths,pixel_sigma=3.,iterations=5):
    x=np.asarray(measurement,float).copy();cov=np.asarray(covariance,float)*pixel_sigma**2
    target=np.asarray(lengths,float)
    if x.shape!=(21,3) or cov.shape!=(21,3,3) or target.shape!=(21,) or not np.isfinite(x).all() or not np.isfinite(cov).all():raise ValueError('Invalid geometry')
    if np.any(np.linalg.eigvalsh(cov)<=0):raise ValueError('Nonpositive covariance')
    bones=[i for i in FINGERS if np.isfinite(target[i]) and target[i]>3]
    if not bones:return x,dict(iterations=0,cost=0.,active_bones=0)
    info=np.linalg.inv(cov);anchor=x.copy();D=np.zeros((63,63))
    for j in range(21):D[3*j:3*j+3,3*j:3*j+3]=info[j]
    def evaluate(y,derivatives):
        delta=y-anchor;value=float(np.einsum('ni,nij,nj->',delta,info,delta))
        if derivatives:H=D.copy();g=np.einsum('nij,nj->ni',info,delta).reshape(-1)
        for i in bones:
            par=PARENTS[i];d=y[i]-y[par];n=np.linalg.norm(d)
            if n<1e-8:return (float('inf'),None,None) if derivatives else float('inf')
            # Soft 10% anatomy uncertainty, with at least 2 mm tolerance.
            sigma=max(2.,target[i]*.1);r=(n-target[i])/sigma;value+=r*r
            if derivatives:
                J=np.zeros(63);J[3*i:3*i+3]=d/(n*sigma);J[3*par:3*par+3]=-d/(n*sigma)
                H+=np.outer(J,J);g+=J*r
        return (value,H,g) if derivatives else value
    value=evaluate(x,False);used=0
    for k in range(iterations):
        old,H,g=evaluate(x,True)
        step=np.linalg.solve(H+np.eye(63)*1e-9,-g).reshape(21,3)
        accepted=False
        for fraction in (1.,.5,.25,.125):
            trial=x+fraction*step;v=evaluate(trial,False)
            if np.isfinite(v) and v<=old:x=trial;value=v;accepted=True;break
        used=k+1
        if not accepted or np.max(np.linalg.norm(fraction*step,axis=1))<.01:break
    return x,dict(iterations=used,cost=value,active_bones=len(bones))
