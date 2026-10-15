#include "finger_correspondence.h"
#include "pose_bank.h"
#include <fstream>
#include <iostream>
int main(int argc,char**argv){try{
 if(argc!=4)return 2;std::ifstream f(argv[1]),c(argv[2]),b(argv[3]);auto rig=fhn::HandRig::load(f);auto cameras=fhn::load_rig_cameras(c);auto bank=fhn::PoseBank::load(b);int checks=0;
 auto check=[&](bool ok){if(!ok)throw std::runtime_error("Correspondence invariant failed");++checks;};
 fhn::RigPose pose;pose.side=rig.subject_side;pose.angles=bank.angles[0];pose.wrist.t={0,-70,-450};auto points=rig.landmarks(pose.angles,pose.wrist,pose.side);
 std::array<fhn::PixelObservation,21> obs{};
 for(int i=0;i<21;++i)for(int v=0;v<2;++v){check(cameras[v].project(points[i],obs[i].point[v]));obs[i].present[v]=i!=20;obs[i].sigma[v]=2.+i*.1+v*.2;}
 for(int v=0;v<2;++v){auto swapped=fhn::swap_index_middle_view(obs,v);auto restored=fhn::swap_index_middle_view(swapped,v);
  for(int i=0;i<21;++i)for(int k=0;k<2;++k){check((restored[i].point[k]-obs[i].point[k]).len()==0&&restored[i].sigma[k]==obs[i].sigma[k]&&restored[i].present[k]==obs[i].present[k]);if(k!=v)check((swapped[i].point[k]-obs[i].point[k]).len()==0&&swapped[i].sigma[k]==obs[i].sigma[k]);}}
 bool rejected=false;try{fhn::swap_index_middle_view(obs,2);}catch(const std::invalid_argument&){rejected=true;}check(rejected);
 auto result=fhn::fit_index_middle_hypotheses(rig,pose,cameras,obs);check(result.chosen==-1&&result.fit.cost==0&&result.fit.pose.angles==pose.angles&&!result.eligible[0]&&!result.eligible[1]);
 for(auto& o:obs)o.present[1]=false;result=fhn::fit_index_middle_hypotheses(rig,pose,cameras,obs);check(!result.fit.sufficient&&result.chosen==-1);
 obs[8].present[1]=true;result=fhn::fit_index_middle_hypotheses(rig,pose,cameras,obs);check(!result.eligible[0]&&!result.eligible[1]);
 std::cout<<"controls="<<checks<<" failures=0\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
