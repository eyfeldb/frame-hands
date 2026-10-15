#include "pose_bank.h"
#include "rig_fit_quality.h"
#include <chrono>
#include <fstream>
#include <iostream>
#include <iomanip>
int main(int argc,char**argv){try{
 if(argc!=7)return 2;std::ifstream l(argv[1]),r(argv[2]),c(argv[3]),bank_file(argv[4]),input(argv[5]);if(std::ifstream(argv[6]).good())return 3;
 std::array<fhn::HandRig,2> rigs{fhn::HandRig::load(l),fhn::HandRig::load(r)};auto cameras=fhn::load_rig_cameras(c);auto bank=fhn::PoseBank::load(bank_file);std::ofstream out(argv[6]);out<<std::setprecision(17);fhn::RigPose seed;int count=0;
 while(input>>seed.side){if(seed.side<0||seed.side>1)return 4;
  if(!(input>>seed.wrist.q.w>>seed.wrist.q.x>>seed.wrist.q.y>>seed.wrist.q.z>>seed.wrist.t.x>>seed.wrist.t.y>>seed.wrist.t.z))return 4;
  for(auto& x:seed.angles)if(!(input>>x))return 4;std::array<fhn::PixelObservation,21> obs{};
  for(auto& o:obs)for(int v=0;v<2;++v){int present;if(!(input>>present>>o.point[v].x>>o.point[v].y>>o.sigma[v])||present<0||present>1)return 4;o.present[v]=present;}
  auto start=std::chrono::steady_clock::now();const auto& rig=rigs[seed.side];auto result=fhn::fit_pose_bank(bank,rig,seed,cameras,obs);double us=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count();
  if(!result.fit.sufficient||result.fit.cost>result.control.cost)throw std::runtime_error("Bank comparison failed");
  auto exact=fhn::pose_bank_cost(rig,result.fit.pose,seed,cameras,obs);if(std::abs(exact-result.fit.cost)>1e-8*std::max(1.,std::abs(exact)))throw std::runtime_error("Cost contract mismatch");
  out<<count<<' '<<us<<' '<<result.chosen<<' '<<result.fits<<' '<<result.control.cost<<' '<<result.fit.cost;
  for(int k=0;k<2;++k)out<<' '<<result.attempted[k]<<' '<<result.costs[k];
  for(const auto* p:std::array<const fhn::RigPose*,3>{&seed,&result.control.pose,&result.fit.pose}){auto q=fhn::rig_fit_quality(rig,*p,cameras,obs);if(!q.projectable)throw std::runtime_error("Behind camera");
   out<<' '<<q.all.rmse_px()<<' '<<q.palm.rmse_px()<<' '<<q.fingers.rmse_px()<<' '<<p->wrist.q.w<<' '<<p->wrist.q.x<<' '<<p->wrist.q.y<<' '<<p->wrist.q.z<<' '<<p->wrist.t.x<<' '<<p->wrist.t.y<<' '<<p->wrist.t.z;
   for(double angle:p->angles)out<<' '<<angle;for(const auto& point:rig.landmarks(p->angles,p->wrist,p->side))out<<' '<<point.x<<' '<<point.y<<' '<<point.z;
  }
  out<<'\n';++count;
 }
 if(!count||!input.eof()||!out.good())return 5;std::cout<<"bank_poses="<<count<<'\n';return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
