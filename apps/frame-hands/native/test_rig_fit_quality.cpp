#include "rig_fit_quality.h"
#include <fstream>
#include <iostream>
int main(int argc,char**argv){try{
 if(argc!=3)return 2;std::ifstream f(argv[1]),c(argv[2]);auto rig=fhn::HandRig::load(f);auto cameras=fhn::load_rig_cameras(c);fhn::RigPose pose;pose.side=rig.subject_side;pose.wrist.t={0,0,-450};
 for(int i=0;i<20;++i)pose.angles[i]=.5*(rig.joints[i].lower+rig.joints[i].upper);
 auto pts=rig.landmarks(pose.angles,pose.wrist,pose.side);std::array<fhn::PixelObservation,21> obs{};int count=0;
 auto check=[&](bool b){if(!b)throw std::runtime_error("Quality control failed");++count;};
 auto q=fhn::rig_fit_quality(rig,pose,cameras,obs);check(q.projectable&&q.all.observations==0&&q.all.rmse_px()==0);
 for(int i=0;i<20;++i)for(int v=0;v<2;++v){obs[i].present[v]=true;check(cameras[v].project(pts[i],obs[i].point[v]));}
 auto original=pose;q=fhn::rig_fit_quality(rig,pose,cameras,obs);check(q.projectable&&q.all.observations==40&&q.palm.observations==10&&q.fingers.observations==30&&q.all.rmse_px()==0);
 for(auto& o:obs)for(int v=0;v<2;++v){o.point[v].x+=3;o.point[v].y+=4;o.sigma[v]=2;}
 q=fhn::rig_fit_quality(rig,pose,cameras,obs);check(std::abs(q.all.rmse_px()-5)<1e-9&&std::abs(q.palm.rmse_px()-5)<1e-9&&std::abs(q.fingers.rmse_px()-5)<1e-9&&std::abs(q.all.rmse_normalized()-2.5)<1e-9);
 obs[0].present[1]=false;q=fhn::rig_fit_quality(rig,pose,cameras,obs);check(q.all.observations==39&&q.palm.observations==10&&q.fingers.observations==29);
 auto missing=obs;missing[0].point[1].x=std::numeric_limits<double>::quiet_NaN();q=fhn::rig_fit_quality(rig,pose,cameras,missing);check(q.projectable&&q.all.observations==39);
 obs[0].sigma[0]=0;bool threw=false;try{fhn::rig_fit_quality(rig,pose,cameras,obs);}catch(const std::invalid_argument&){threw=true;}check(threw);obs[0].sigma[0]=2;
 pose.wrist.t.z=450;q=fhn::rig_fit_quality(rig,pose,cameras,obs);check(!q.projectable&&q.all.observations==0);
 pose=original;q=fhn::rig_fit_quality(rig,pose,cameras,obs);check(pose.angles==original.angles&&pose.wrist.t.z==original.wrist.t.z);
 obs[0].point[0].x=1e300;threw=false;try{fhn::rig_fit_quality(rig,pose,cameras,obs);}catch(const std::invalid_argument&){threw=true;}check(threw);
 std::cout<<"controls="<<count<<" failures=0\n";return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
