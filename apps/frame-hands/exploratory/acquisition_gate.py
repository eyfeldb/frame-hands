"""Experimental publication-only confirmation of a newly acquired hand.

No association, pose adjustment, camera or driver access. A first monocular
pose needs independent same-capture support in both cameras; established hands
may continue monocularly. This is not a probability or false-positive guarantee.
"""
from score_roi_presence import supported_views
from frame_data import NAMES


class AcquisitionGate:
    MAX_CONTINUITY_NS=650_000_000

    def __init__(self):
        self.confirmed={};self.last_stamp=None

    def step(self,stamp,valid,hands,evidence):
        if self.last_stamp is not None and stamp<=self.last_stamp:
            raise ValueError('Captures must increase')
        self.last_stamp=stamp
        if not valid:
            self.confirmed={}
            return [],[]
        if len({h['side'] for h in hands})!=len(hands) or any(h['side'] not in (0,1) for h in hands):
            raise ValueError('Expected unique anatomical slots')
        if any(r['stamp']!=stamp for r in evidence):
            raise ValueError('Evidence must belong to this capture')
        self.confirmed={s:v for s,v in self.confirmed.items() if 0<stamp-v[1]<=self.MAX_CONTINUITY_NS}
        output=[];decisions=[]
        for h in hands:
            side=h['side'];identity=h['source_id'];previous=self.confirmed.get(side)
            continued=previous is not None and previous[0]==identity
            stereo=set(NAMES).issubset({v['camera'] for v in h['source_views']})
            rows=[r for r in evidence if r['side']==side and r['source_id']==identity]
            image_stereo=set(NAMES).issubset({r['camera'] for r in supported_views(rows)})
            accepted=continued or stereo or image_stereo
            reason='continued' if continued else 'detector_stereo' if stereo else 'image_stereo' if image_stereo else 'unconfirmed_birth'
            decisions.append(dict(side=side,source_id=identity,accepted=accepted,reason=reason))
            if accepted:
                output.append(h);self.confirmed[side]=(identity,stamp)
            else:self.confirmed.pop(side,None)
        return output,decisions
