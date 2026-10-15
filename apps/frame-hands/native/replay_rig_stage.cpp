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
  auto begin=std::chrono::steady_clock::now();const auto result=stage.fit(reference,rigs[side],side,points,cameras,pixels,stamp,source);
  double us=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-begin).count();
  const auto& r=result.refined.fit;const auto& pose=r.pose;
  out<<result.initialized<<' '<<us<<' '<<result.initial.cost<<' '<<r.cost<<' '<<r.observed_rmse_px<<' '<<result.refined.used_history<<' '<<result.refined.fit_count;
  const auto& w=result.seed.wrist;out<<' '<<w.q.w<<' '<<w.q.x<<' '<<w.q.y<<' '<<w.q.z<<' '<<w.t.x<<' '<<w.t.y<<' '<<w.t.z;
  out<<' '<<pose.wrist.q.w<<' '<<pose.wrist.q.x<<' '<<pose.wrist.q.y<<' '<<pose.wrist.q.z<<' '<<pose.wrist.t.x<<' '<<pose.wrist.t.y<<' '<<pose.wrist.t.z;
  for(auto a:pose.angles)out<<' '<<a;
  if(result.initialized){for(auto p:rigs[side].landmarks(pose.angles,pose.wrist,side))out<<' '<<p.x<<' '<<p.y<<' '<<p.z;}
  else for(int i=0;i<63;++i)out<<" 0";
  out<<'\n';++count;
 }
 if(!input.eof()||!out.good()||!count)return 5;std::cout<<"hands="<<count<<'\n';return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 7;}}
