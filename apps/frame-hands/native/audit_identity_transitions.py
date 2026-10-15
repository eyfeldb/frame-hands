"""Describe provisional side changes from immutable full-neural detector audit rows."""
import argparse,json,sys
from pathlib import Path
from collections import Counter
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'exploratory'))
from frame_data import read_audit
from run_tracker_rig_check import sha

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for n in ('run','output'):p.add_argument('--'+n,type=Path,required=True)
 a=p.parse_args();report=json.loads((a.run/'result.json').read_text());row=report['results'][1];path=a.run/row['name']/'views.txt'
 assert sha(path)==row['views_sha256'];frames=read_audit(path);origin=min(frames);previous={};changes=[];duplicate_frames=[]
 for raw,frame in frames.items():
  sides=Counter(int(h['right']>.5) for h in frame['hands'].values())
  if any(n>1 for n in sides.values()):duplicate_frames.append(raw)
  for ident,hand in frame['hands'].items():
   score=hand['right'];old=previous.get(ident)
   if old and 0<raw-old['raw']<=100000000 and (score>.5)!=(old['score']>.5):
    views=[dict(camera=v['camera'],presence=v['presence'],right=v['right']) for v in frame['views'] if v['id']==ident]
    votes=[1 if v['right']>.85 else 0 if v['right']<.15 else -1 for v in views]
    evidence=dict(raw_ns=raw,seconds=(raw-origin)/1e9,source_id=ident,before=old['score'],after=score,stereo=hand['stereo'],views=views,strict_view_votes_agree=len(votes)>=2 and votes[0]>=0 and all(v==votes[0] for v in votes),new_score_is_latch_endpoint=min(abs(score-.01),abs(score-.99))<1e-9,duplicate_output_sides=any(n>1 for n in sides.values()))
    changes.append(evidence)
   previous[ident]=dict(raw=raw,score=score)
 result=dict(changes=changes,counts=dict(side_changes=len(changes),non_latch_endpoint=sum(not x['new_score_is_latch_endpoint'] for x in changes),during_duplicate_output=sum(x['duplicate_output_sides'] for x in changes),strict_view_votes_agree=sum(x['strict_view_votes_agree'] for x in changes),duplicate_side_frames=len(duplicate_frames)),views_sha256=sha(path),source_sha256=sha(Path(__file__)),limitations=['These are source-score changes, not independently labeled physical left/right errors.','Enabled identity latch overwrites final score with.01/.99; other values indicate provisional score on this frozen code path.','Strict-view agreement omits curl/open-hand eligibility, which is not present in this audit. This is not a reconstruction of latch state.','No relabeling, threshold change or runtime mutation.'])
 a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result['counts']))
if __name__=='__main__':main()
