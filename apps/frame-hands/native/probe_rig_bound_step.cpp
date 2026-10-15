#include "rig_stage.h"
#include <fstream>
#include <iostream>
#include <iomanip>
#include <chrono>
int main(int argc,char**argv){try{
 if(argc!=7)return 2;std::ifstream generic(argv[1]),left(argv[2]),right(argv[3]),calibration(argv[4]),input(argv[5]);
 if(!generic||!left||!right||!calibration||!input||std::ifstream(argv[6]).good())return 3;
 const auto reference=fhn::HandRig::load(generic);std::array<fhn::HandRig,2> rigs{fhn::HandRig::load(left),fhn::HandRig::load(right)};
 const auto cameras=fhn::load_rig_cameras(calibration);fhn::RigStage stage;std::ofstream out(argv[6]);out<<std::setprecision(17);
 std::int64_t stamp;int source,side,count=0;
 while(input>>stamp){if(!(input>>source>>side)||side<0||side>1)return 4;
  std::array<fhn::RigObservation,21> points{};std::array<fhn::PixelObservation,21> pixels{};
  for(auto& p:points){int present;if(!(input>>present>>p.point.x>>p.point.y>>p.point.z)||present<0||present>1)return 4;p.present=present;}
  for(auto& p:pixels)for(int v=0;v<2;++v){int present;if(!(input>>present>>p.point[v].x>>p.point[v].y>>p.sigma[v])||present<0||present>1)return 4;p.present[v]=present;}
  const auto baseline=stage.fit(reference,rigs[side],side,points,cameras,pixels,stamp,source);
  if(!baseline.initialized)return 8;
  // Freeze baseline history and selected seed: isolate bounds, not temporal branching.
  auto seed=baseline.initial.pose;
  // Recover selected history seed from the previous baseline pose only when selected.
  static std::array<fhn::RigPose,2> previous{};
  if(baseline.refined.used_history)seed.angles=previous[side].angles;
  previous[side]=baseline.refined.fit.pose;
  auto run=[&](const fhn::RigPose& start,bool bounds){auto begin=std::chrono::steady_clock::now();
   auto fit=fhn::fit_rig_pixels(rigs[side],start,cameras,pixels,{},0,&baseline.initial.pose.angles,bounds);
   auto us=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-begin).count();
   return std::make_pair(fit,us);};
  auto control=run(seed,false),candidate=run(seed,true),continued=run(baseline.refined.fit.pose,false);
  // Same initial seed/prior must reproduce the stage's selected completed fit.
  if(std::abs(control.first.cost-baseline.refined.fit.cost)>1e-7)return 9;
  out<<stamp<<' '<<source<<' '<<side;
  for(const auto* item:{&control,&candidate,&continued}){
   const auto& r=item->first;out<<' '<<r.cost<<' '<<r.observed_rmse_px<<' '<<r.iterations<<' '<<r.updates<<' '<<item->second;
   for(auto angle:r.pose.angles)out<<' '<<angle;
   for(auto point:rigs[side].landmarks(r.pose.angles,r.pose.wrist,side))out<<' '<<point.x<<' '<<point.y<<' '<<point.z;
  }
  out<<'\n';++count;
 }
 if(!input.eof()||!out.good()||!count)return 5;std::cout<<"hands="<<count<<'\n';return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 7;}}
