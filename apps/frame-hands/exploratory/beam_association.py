"""Offline causal four-hypothesis association experiment; no future-frame lookahead.

Retains separate motion histories, never averages physical hands. Costs are the
existing experimental costs, not calibrated log probabilities. The best current
branch emits only fresh selected views. This is not a full MHT implementation.
"""
import numpy as np
from temporal_association import TemporalAssociation

class BeamAssociation:
    def __init__(self,cameras,width=4):
        if not 1<=width<=8:raise ValueError('Beam width outside bounded experiment')
        self.cameras=cameras;self.width=width;self.last_stamp=None
        self.hypotheses=[dict(cost=0.,policy=TemporalAssociation(cameras),profile=())]
        self.last_debug={};self.ambiguous=False

    def reset(self):
        generation=max(h['policy'].generation for h in self.hypotheses)
        policy=TemporalAssociation(self.cameras);policy.generation=generation
        self.hypotheses=[dict(cost=0.,policy=policy,profile=())]
        self.last_stamp=None;self.last_debug={};self.ambiguous=False

    @staticmethod
    def profile(result):
        assigned,candidates,_=result
        return tuple((side,tuple(sorted((v['camera'],v['id']) for v in candidates[ident]))) for side,ident in sorted(assigned.items()))

    def assign(self,observation,world):
        if world is None or np.shape(world)!=(4,4) or not np.isfinite(world).all():
            self.reset();self.last_debug=dict(assignments=[],reason='Invalid capture context')
            return {},{},{}
        stamp=observation['raw_ns']
        if not isinstance(stamp,(int,np.integer)) or stamp<0:raise ValueError('Invalid capture timestamp')
        if self.last_stamp is not None and stamp<=self.last_stamp:
            self.last_debug=dict(assignments=[],reason='Non-increasing capture timestamp')
            return {},{},{}
        if self.last_stamp is not None and stamp-self.last_stamp>650_000_000:self.reset()
        self.last_stamp=stamp
        children=[]
        for parent_index,parent in enumerate(self.hypotheses):
            for increment,result,policy in parent['policy'].branches(observation,world,self.width):
                profile=self.profile(result)
                children.append(dict(cost=parent['cost']+increment,policy=policy,result=result,profile=profile,
                    diversity=(profile,parent['profile']),parent=parent_index))
        children.sort(key=lambda h:h['cost'])
        kept=[];seen=set()
        for child in children:
            if child['diversity'] in seen:continue
            kept.append(child);seen.add(child['diversity'])
            if len(kept)==self.width:break
        minimum=kept[0]['cost']
        for h in kept:h['cost']-=minimum
        self.hypotheses=kept
        leader=kept[0];self.ambiguous=leader['policy'].ambiguous
        self.last_debug=dict(leader['policy'].last_debug,
            hypotheses=len(kept),leader_parent=leader['parent'],
            alternatives=[dict(relative_cost=h['cost'],profile=h['profile']) for h in kept])
        return leader['result']
