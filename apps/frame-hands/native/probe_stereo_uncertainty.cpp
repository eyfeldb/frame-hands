#include "stereo_disagreement.h"
#include "pose_bank.h"
#include <fstream>
#include <iomanip>
#include <iostream>
int main(int argc,char**argv){try{
 if(argc!=5)return 2;std::ifstream f(argv[1]),c(argv[2]),b(argv[3]);if(std::ifstream(argv[4]).good())return 3;auto rig=fhn::HandRig::load(f);auto cams=fhn::load_rig_cameras(c);auto bank=fhn::PoseBank::load(b);bank.validate(rig);std::ofstream out(argv[4]);out<<std::setprecision(17);
 for(int pose_index=0;pose_index<int(bank.angles.size());++pose_index)for(int mode=0;mode<3;++mode){
  fhn::RigPose truth;truth.side=rig.subject_side;truth.angles=bank.angles[pose_index];truth.wrist.t={0,-70,-450};auto points=rig.landmarks(truth.angles,truth.wrist,truth.side);std::array<fhn::PixelObservation,21> observations{};
  for(int i=0;i<21;++i)for(int v=0;v<2;++v){observations[i].present[v]=i!=20;if(!cams[v].project(points[i],observations[i].point[v]))throw std::runtime_error("Bad truth projection");if(mode==1){observations[i].point[v].x+=2*std::sin(double(i*3+v+1));observations[i].point[v].y+=2*std::cos(double(i*2+v+1));}}
  // Index/middle chains in one camera receive the other's semantic labels.
  if(mode==2)for(auto indices:std::array<std::pair<int,int>,4>{{{8,11},{9,12},{10,13},{1,2}}})std::swap(observations[indices.first].point[1],observations[indices.second].point[1]);
  auto weighted=observations;int changed=0;for(auto&o:weighted){double factor=fhn::stereo_uncertainty_factor(fhn::stereo_disagreement(cams,o),o);changed+=factor>1;for(auto&s:o.sigma)s*=factor;}
  auto seed=truth;seed.wrist.t.x+=3;for(int i=0;i<20;++i)seed.angles[i]=std::clamp(seed.angles[i]+.05,rig.joints[i].lower,rig.joints[i].upper);
  auto control=fhn::fit_rig_pixels(rig,seed,cams,observations);auto candidate=fhn::fit_rig_pixels(rig,seed,cams,weighted);if(!control.sufficient||!candidate.sufficient)throw std::runtime_error("Fit failed");
  out<<pose_index<<' '<<mode<<' '<<changed;
  for(const auto* result:std::array<const fhn::PixelFitResult*,2>{&control,&candidate}){auto fitted=rig.landmarks(result->pose.angles,result->pose.wrist,truth.side);double square=0,max=0;for(int i=0;i<21;++i){double e=(fitted[i]-points[i]).len();square+=e*e;max=std::max(max,e);}out<<' '<<std::sqrt(square/21)<<' '<<max;}
  out<<'\n';if(!changed&&(control.pose.angles!=candidate.pose.angles||control.pose.wrist.q.w!=candidate.pose.wrist.q.w||control.cost!=candidate.cost))throw std::runtime_error("No-change control differs");
 }
 std::cout<<"synthetic_cases="<<bank.angles.size()*3<<'\n';return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
