#include "provisional_side.h"
#include <iostream>
#include <limits>
int main(){try{
 int checks=0;auto check=[&](bool x){++checks;if(!x)throw std::runtime_error("Provisional side check failed");};
 for(int side=0;side<2;++side){
  fhn::ProvisionalSide p;std::int64_t now=1;
  check(p.observe(side?.9:.1,side,now)&&p.right==bool(side));
  for(int i=0;i<20;++i){now+=33333333;check(p.observe(i%2?.51:.49,-1,now)&&p.right==bool(side));}
  now+=33333333;double opposite=side?.3:.7;check(p.observe(opposite,-1,now)&&p.right==bool(side));
  now+=33333333;check(p.observe(opposite,side,now)&&p.right==bool(side));
  now+=33333333;check(p.observe(opposite,1-side,now)&&p.right!=bool(side));
  auto before=p;check(!p.observe(side?.9:.1,side,now)&&p.right==before.right&&p.last==before.last);
  check(!p.observe(side?.9:.1,side,now-1)&&p.right==before.right);
  now+=33333333;check(p.observe(.5,-1,now,side)&&p.right==bool(side));
  now+=100000001;check(p.observe(opposite,-1,now)&&p.right!=bool(side));
  p.clear();check(!p.active&&p.last==0);check(p.observe(side?.9:.1,-1,now+1)&&p.right==bool(side));
 }
 fhn::ProvisionalSide p;p.observe(.5,-1,1);check(!p.right);
 p.observe(.64999,1,2);check(!p.right);p.observe(.65,1,3);check(p.right);
 p.observe(.35001,0,4);check(p.right);p.observe(.35,0,5);check(!p.right);
 for(double invalid:{-1.,2.,std::numeric_limits<double>::quiet_NaN(),std::numeric_limits<double>::infinity()}){
  bool failed=false;try{p.observe(invalid,-1,6);}catch(const std::invalid_argument&){failed=true;}check(failed);
 }
 bool failed=false;try{p.observe(.5,2,6);}catch(const std::invalid_argument&){failed=true;}check(failed);
 failed=false;try{p.observe(.5,0,0);}catch(const std::invalid_argument&){failed=true;}check(failed);
 for(int initial=0;initial<2;++initial){
  fhn::ProvisionalSide p;p.stereo_first=true;std::int64_t now=1;
  check(p.observe(initial?.6:.4,-1,now)&&p.right==bool(initial)&&!p.stereo_supported);
  // A weak initial guess is not sticky: retain original bootstrap behavior.
  now+=33333333;check(p.observe(initial?.4:.6,-1,now)&&p.right!=bool(initial)&&!p.stereo_supported);
  // Current stereo is allowed to correct even a contradictory lagging score.
  now+=33333333;check(p.observe(initial?.1:.9,initial,now)&&p.right==bool(initial)&&p.stereo_supported);
  for(int i=0;i<20;++i){now+=33333333;check(p.observe(i%2?.51:.49,-1,now)&&p.right==bool(initial));}
  now+=33333333;check(p.observe(initial?.9:.1,1-initial,now)&&p.right!=bool(initial));
  auto old=p;check(!p.observe(.5,initial,now)&&p.right==old.right&&p.stereo_supported==old.stereo_supported);
  check(!p.observe(.5,initial,now-1)&&p.last==old.last);
  now+=33333333;check(p.observe(.5,1-initial,now,initial)&&p.right==bool(initial));
  now+=100000001;check(p.observe(initial?.4:.6,-1,now)&&p.right!=bool(initial)&&!p.stereo_supported);
  p.clear();check(!p.active&&!p.stereo_supported&&p.stereo_first);
 }
 std::cout<<"checks="<<checks<<" failures=0\n";return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
