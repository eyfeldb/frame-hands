"""Render timestamp-matched LIVE camera overlays from a copied session (Pillow only)."""
import argparse
import json
from pathlib import Path
import struct
import sys
from PIL import Image, ImageDraw, ImageFont, ImageOps

sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from recording import SET,CAMERA,RELAY_SIZE

EDGES=[(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),(13,17),(0,17),(17,18),(18,19),(19,20)]


def audit(path):
    result={};current=None
    for line in path.read_text().splitlines():
        f=line.split()
        if not f:continue
        if f[0]=='set':current=dict(raw=int(f[1]),mono=int(f[2]),count=int(f[3]),hands={},views=[])
        elif f[0]=='hand':current['hands'][int(f[1])]=dict(right=float(f[2]),views=int(f[3]))
        elif f[0]=='view':
            if len(f)!=47:raise ValueError('Malformed view')
            coords=list(map(float,f[5:]));current['views'].append(dict(camera=f[1],id=int(f[2]),presence=float(f[3]),right=float(f[4]),points=list(zip(coords[::2],coords[1::2]))))
        elif f[0]=='end':
            if current is None or len(current['hands'])!=current['count']:raise ValueError('Incomplete audit')
            result[current['raw']]=current;current=None
    if current is not None:raise ValueError('Truncated audit')
    return result


def mapping(session,observations):
    mono={row['mono']:stamp for stamp,row in observations.items()};result={}
    with (session/'relay-trace.bin').open('rb') as stream:
        while data:=stream.read(RELAY_SIZE):
            if len(data)!=RELAY_SIZE or data[:8]!=b'FHRELAY1':raise ValueError('Bad relay trace')
            if data[56:64]!=b'FHHANDS1':continue
            capture=struct.unpack_from('<Q',data,80)[0]
            if capture in mono:result[mono[capture]]=struct.unpack_from('<Q',data,2744)[0]
    with (session/'sampler-trace.bin').open('rb') as stream:
        header=stream.read(24)
        if header[:8]!=b'FHTRACE1':raise ValueError('Bad sampler trace')
        first=struct.unpack_from('<Q',header,16)[0]
    return result,first


def selected_images(path,targets):
    with path.open('rb') as stream:
        size=path.stat().st_size
        while stream.tell()<size:
            start=stream.tell();header=stream.read(SET.size)
            if len(header)!=SET.size:raise ValueError('Truncated set')
            magic,count,total=SET.unpack(header)
            if magic!=b'FHSET01\0' or not 1<=count<=4:raise ValueError('Bad set')
            data=stream.read(CAMERA.size*count)
            if len(data)!=CAMERA.size*count:raise ValueError('Truncated camera metadata')
            cameras=[CAMERA.unpack_from(data,i*CAMERA.size) for i in range(count)]
            if total!=SET.size+CAMERA.size*count+sum(w*h for _,w,h,_,_ in cameras) or start+total>size:raise ValueError('Bad image length')
            stamp=min(t for _,_,_,t,_ in cameras)
            if stamp not in targets:stream.seek(start+total);continue
            images={}
            for name,width,height,_,_ in cameras:
                images[name.rstrip(b'\0').decode()]=Image.frombytes('L',(width,height),stream.read(width*height))
            yield stamp,images


def draw(images,observation,seconds,label='LIVE'):
    try:font=ImageFont.truetype('arial.ttf',23)
    except OSError:font=ImageFont.load_default(size=23)
    sheet=Image.new('RGB',(1056,1024))
    for column,name in enumerate(('slam_left','slam_right')):
        pixels=images[name]
        raw=pixels.convert('RGB');overlay=ImageOps.autocontrast(pixels,cutoff=1).convert('RGB');pen=ImageDraw.Draw(overlay)
        for view in observation['views']:
            if view['camera']!=name:continue
            hand=observation['hands'].get(view['id']);side=('R' if hand['right']>.5 else 'L') if hand else '?'
            color={'L':(0,240,255),'R':(255,155,0),'?':(220,220,220)}[side]
            points=view['points']
            for a,b in EDGES:pen.line([points[a],points[b]],fill=color,width=3)
            for x,y in points:pen.ellipse((x-3,y-3,x+3,y+3),fill=color)
            x,y=points[0];hand_label=f"{side} id{view['id']} p={view['presence']:.2f} modelR={view['right']:.2f}"
            pen.text((max(4,min(x,500)),max(50,min(y,pixels.height-35))),hand_label,font=font,fill=color,stroke_width=1,stroke_fill='black')
        for row,(tile,caption) in enumerate(((raw,'raw'),(overlay,label+' joints; contrast enhanced'))):
            pen=ImageDraw.Draw(tile);pen.rectangle((0,0,tile.width,42),fill='black')
            pen.text((10,5),f'{seconds:.3f}s {name} | {caption}',font=font,fill='white')
            sheet.paste(tile.resize((528,512)),(column*528,row*512))
    return sheet


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('session',type=Path)
    parser.add_argument('--sampler-ms',type=int,nargs='+',required=True)
    parser.add_argument('--replay-audit',type=Path,help='Alternative predictions; always labeled REPLAY')
    args=parser.parse_args();live=audit(args.session/'frame/live-views.txt')
    observations=audit(args.replay_audit) if args.replay_audit else live
    times,first=mapping(args.session,live)
    if not times:raise ValueError('No exact source-to-relay timestamp matches')
    targets={min(times,key=lambda stamp:abs(times[stamp]-first-ms)) for ms in args.sampler_ms}
    output=args.session/('replay-review' if args.replay_audit else 'visual-review');output.mkdir(exist_ok=True)
    index=[];begin=min(observations)
    for stamp,images in selected_images(args.session/'frame/sets.bin',targets):
        seconds=(stamp-begin)/1e9;name=f'{seconds:06.3f}.jpg'
        if stamp not in observations:raise ValueError('Selected recorded frame has no matching replay observation')
        draw(images,observations[stamp],seconds,'REPLAY' if args.replay_audit else 'LIVE').save(output/name,quality=95)
        index.append(dict(file=name,camera_seconds=seconds,sampler_ms=times[stamp]-first,hands=observations[stamp]['hands']))
    (output/'index.json').write_text(json.dumps(index,indent=2)+'\n')
    print(json.dumps(index,indent=2))


if __name__=='__main__':main()
