"""Native stabilized joints -> PC raw tracking space. No inference or filtering."""
import base64,math,struct
from collections import OrderedDict
HEADER=struct.Struct('<8sIIQQQ')
HAND=struct.Struct('<IIQQ63f')

def rigid(values):
    if not isinstance(values,list) or len(values)!=12 or not all(type(x) in (int,float) and math.isfinite(x) for x in values):raise ValueError('Invalid head matrix')
    rows=[values[i:i+4] for i in (0,4,8)]
    for a in range(3):
        for b in range(3):
            if abs(sum(rows[a][k]*rows[b][k] for k in range(3))-(a==b))>.002:raise ValueError('Nonrigid head matrix')
    a,b,c=rows
    det=a[0]*(b[1]*c[2]-b[2]*c[1])-a[1]*(b[0]*c[2]-b[2]*c[0])+a[2]*(b[0]*c[1]-b[1]*c[0])
    if abs(det-1)>.002:raise ValueError('Reflected head matrix')
    return rows

def map_point(point,native,pc):
    delta=[point[i]-native[i][3] for i in range(3)]
    local=[sum(native[i][j]*delta[i] for i in range(3)) for j in range(3)]
    return [sum(pc[i][j]*local[j] for j in range(3))+pc[i][3] for i in range(3)]

def unpack(reply,started,received):
    if reply.get('state')!='native':return None
    if reply.get('version')!=1:raise ValueError('Unknown native export')
    if not 0<=received-started<=40:return None
    raw=base64.b64decode(reply['packet'],validate=True)
    if len(raw)!=592:raise ValueError('Invalid model packet length')
    magic,version,size,sequence,published,source=HEADER.unpack_from(raw)
    now=reply['native_now_ms']
    if type(now)!=int or magic!=b'FHMODEL2' or version!=2 or size!=592 or not sequence:raise ValueError('Invalid model header')
    if not 0<=now-published<80 or not 0<source<=published or now-source>500:return None
    # RTT is charged in full to validity. It is never subtracted from capture age.
    render=started-(now-published)
    if received-render>=80:return None
    head=rigid(reply['head']);hands=[];identities=set()
    for side in range(2):
        identity,flags,capture,until,*points=HAND.unpack_from(raw,HEADER.size+side*HAND.size)
        if flags==0:continue
        if flags!=1 or not identity or identity in identities:raise ValueError('Invalid optical identities')
        if not 0<capture<=source or not capture<until<=capture+350:raise ValueError('Invalid native expiry')
        if not all(math.isfinite(x) and abs(x)<1000 for x in points):raise ValueError('Invalid joints')
        identities.add(identity);hands.append((side,identity,capture,until,points))
    return dict(sequence=sequence,published=published,source=source,now=now,head=head,hands=hands,render=render,
                # Head-space alignment uses the midpoint time; uncertainty <= RTT/2.
                pose_age_ms=received-(render+(received-started)/2))

def compose(native,pc):
    """Rigid transform taking native head space to PC space: pc * inverse(native), as 3 rows of 4."""
    rot=[[sum(pc[i][k]*native[j][k] for k in range(3)) for j in range(3)] for i in range(3)]   # R_pc * R_native^T
    t=[pc[i][3]-sum(rot[i][j]*native[j][3] for j in range(3)) for i in range(3)]
    return [rot[i]+[t[i]] for i in range(3)]

def orthonormalize(rows):
    r=[list(row[:3]) for row in rows]
    def norm(v):
        n=math.sqrt(sum(x*x for x in v)) or 1.;return [x/n for x in v]
    a=norm(r[0]);b=[r[1][i]-sum(r[1][k]*a[k] for k in range(3))*a[i] for i in range(3)];b=norm(b)
    c=[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]]
    return [a+[rows[0][3]],b+[rows[1][3]],c+[rows[2][3]]]

class WorldAlignment:
    """The headset and the PC track the same head, so the transform between their worlds is (nearly) constant.
    Estimating it per frame from the two head poses injects every timing error of the link into the hands;
    averaging the per-frame estimates removes that jitter. Large sustained deviations (a recentre) snap."""
    def __init__(self,alpha=.05):self.alpha=alpha;self.value=None;self.deviant_since=None
    def update(self,native,pc,now):
        fresh=compose(native,pc)
        if self.value is None:self.value=fresh;return self.value
        dt=math.sqrt(sum((fresh[i][3]-self.value[i][3])**2 for i in range(3)))
        trace=sum(fresh[i][k]*self.value[i][k] for i in range(3) for k in range(3))
        angle=math.degrees(math.acos(max(-1.,min(1.,(trace-1)/2))))
        if dt>.10 or angle>5:
            if self.deviant_since is None:self.deviant_since=now
            if now-self.deviant_since>1000:self.value=fresh;self.deviant_since=None
            return self.value
        self.deviant_since=None
        mixed=[[self.value[i][j]+self.alpha*(fresh[i][j]-self.value[i][j]) for j in range(4)] for i in range(3)]
        self.value=orthonormalize(mixed);return self.value

def map_point_world(point,world):
    return [sum(world[i][j]*point[j] for j in range(3))+world[i][3] for i in range(3)]

class Converter:
    def __init__(self):
        self.last_sequence=0;self.last_native_now=0;self.captures=OrderedDict();self.alignment=WorldAlignment()
    def stamp(self,capture,started,native_now):
        conservative=started-(native_now-capture)
        # Repeated render output may shorten, but can never renew, an observation.
        result=min(self.captures.get(capture,conservative),conservative)
        self.captures[capture]=result
        while len(self.captures)>256:self.captures.popitem(last=False)
        return result
    def convert(self,decoded,pc_head,started,received,sequence,now):
        if decoded is None or not pc_head.get('valid'):return None
        pc=rigid(pc_head['capture'])
        if decoded['now']<self.last_native_now:
            self.last_sequence=0;self.captures.clear()
        self.last_native_now=decoded['now']
        if decoded['sequence']<=self.last_sequence:return None
        if now-decoded['render']>=80:return None
        source=self.stamp(decoded['source'],started,decoded['now'])
        if not 0<source<=now or now-source>500:return None
        output=[HAND.pack(0,0,0,0,*([0.]*63)) for _ in range(2)]
        world=self.alignment.update(decoded['head'],pc,now)
        for side,identity,capture,until,points in decoded['hands']:
            mapped=self.stamp(capture,started,decoded['now']);expiry=mapped+until-capture
            if not 0<mapped<=source or now>=expiry:continue
            transformed=[]
            for index in range(0,63,3):transformed.extend(map_point_world(points[index:index+3],world))
            output[side]=HAND.pack(identity,1,mapped,expiry,*transformed)
        self.last_sequence=decoded['sequence']
        return HEADER.pack(b'FHMODEL2',2,592,sequence,decoded['render'],source)+b''.join(output)
