#include "finger_correspondence.h"
#include "pose_bank.h"
#include <fstream>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <random>
int main(int argc,char**argv){try{
 if(argc!=6)return 2;std::ifstream l(argv[1]),r(argv[2]),c(argv[3]),b(argv[4]);if(std::ifstream(argv[5]).good())return 3;
 std::array<fhn::HandRig,2> rigs{fhn::HandRig::load(l),fhn::HandRig::load(r)};auto cameras=fhn::load_rig_cameras(c);auto bank=fhn::PoseBank::load(b);std::ofstream out(argv[5]);out<<std::setprecision(17);int count=0;
 std::mt19937 rng(42083);std::normal_distribution<double> noise(0,2);
 for(int family=0;family<2;++family)for(int side=0;side<2;++side)for(int mode=0;mode<5;++mode){
  std::array<fhn::RigPose,2> previous{};
  for(int frame=0;frame<(family?90:int(bank.angles.size()));++frame){
   const auto& rig=rigs[side];fhn::RigPose truth;truth.side=side;truth.wrist.t={0,-70,-450};truth.angles=bank.angles[frame%bank.angles.size()];
   if(family){double blend=frame<45?std::clamp((frame-20)/5.,0.,1.):1-std::clamp((frame-60)/5.,0.,1.);
    for(int j=0;j<20;++j){int finger=j/4,k=j%4;double curl=finger==1?blend:finger==2?1-blend:.8;
     double value=k==0?0:k==1?curl:k==2?1.4*curl:.9*curl;truth.angles[j]=std::clamp(value,rig.joints[j].lower,rig.joints[j].upper);}
    truth.wrist.q=fhn::rig_exp({.1*std::sin(frame*.06),.15*std::cos(frame*.04),.08*std::sin(frame*.05)});
    truth.wrist.t.x=10*std::sin(frame*.1);
   }
   auto points=rig.landmarks(truth.angles,truth.wrist,side);std::array<fhn::PixelObservation,21> pixels{};
   for(int i=0;i<21;++i)for(int v=0;v<2;++v){if(!cameras[v].project(points[i],pixels[i].point[v]))throw std::runtime_error("Truth behind camera");pixels[i].present[v]=i!=20;
    if(mode==1){pixels[i].point[v].x+=noise(rng);pixels[i].point[v].y+=noise(rng);}}
   bool corruption=mode>=2&&(!family||(frame>=12&&frame<35));
   if(corruption){if(mode==2||mode==4)pixels=fhn::swap_index_middle_view(pixels,0);if(mode==3||mode==4)pixels=fhn::swap_index_middle_view(pixels,1);}
   auto seed=truth;if(!family){seed.wrist.t.x+=3;for(int j=0;j<20;++j)seed.angles[j]=std::clamp(seed.angles[j]+.05,rig.joints[j].lower,rig.joints[j].upper);}
   auto baseline=fhn::fit_rig_pixels(rig,family&&frame?previous[0]:seed,cameras,pixels);
   auto start=std::chrono::steady_clock::now();auto candidate=fhn::fit_index_middle_hypotheses(rig,family&&frame?previous[1]:seed,cameras,pixels);double us=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count();
   if(!baseline.sufficient||!candidate.fit.sufficient)throw std::runtime_error("Unexpected missing fit");
   out<<family<<' '<<side<<' '<<mode<<' '<<frame<<' '<<corruption<<' '<<candidate.chosen<<' '<<candidate.original_disagreement<<' '<<candidate.alternative_disagreement[0]<<' '<<candidate.alternative_disagreement[1]<<' '<<us;
   for(const auto* fit:std::array<const fhn::PixelFitResult*,2>{&baseline,&candidate.fit}){auto p=rig.landmarks(fit->pose.angles,fit->pose.wrist,side);double square=0,max=0;for(int i=0;i<21;++i){double d=(p[i]-points[i]).len();square+=d*d;max=std::max(max,d);}out<<' '<<std::sqrt(square/21)<<' '<<max<<' '<<fit->cost;}
   // Emit truth/control/candidate geometry for independent metric reconstruction.
   for(const auto& p:std::array<fhn::RigPose,3>{truth,baseline.pose,candidate.fit.pose}){out<<' '<<p.wrist.q.w<<' '<<p.wrist.q.x<<' '<<p.wrist.q.y<<' '<<p.wrist.q.z<<' '<<p.wrist.t.x<<' '<<p.wrist.t.y<<' '<<p.wrist.t.z;for(auto angle:p.angles)out<<' '<<angle;}
   out<<'\n';previous={baseline.pose,candidate.fit.pose};++count;
  }
 }
 if(!out.good())return 4;std::cout<<"synthetic_cases="<<count<<'\n';return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
