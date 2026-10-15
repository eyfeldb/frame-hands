#include "rig_stage.h"
#include "whole_pose_fitter.h"
#include <fstream>
#include <iostream>
#include <iomanip>
#include <chrono>
int main(int argc,char**argv){try{
 if(argc!=8)return 2;std::ifstream generic(argv[1]),left(argv[2]),right(argv[3]),calibration(argv[4]),input(argv[5]),heads(argv[6]);
 if(!generic||!left||!right||!calibration||!input||!heads||std::ifstream(argv[7]).good())return 3;
 const auto reference=fhn::HandRig::load(generic);std::array<fhn::HandRig,2> rigs{fhn::HandRig::load(left),fhn::HandRig::load(right)};
 const auto cameras=fhn::load_rig_cameras(calibration);fhn::RigStage stage;std::ofstream out(argv[7]);out<<std::setprecision(17);
 fhn::WholePoseFitter history;
 std::int64_t stamp;int source,side,count=0;
 while(input>>stamp){if(!(input>>source>>side)||side<0||side>1)return 4;
  std::array<fhn::RigObservation,21> points{};std::array<fhn::PixelObservation,21> pixels{};
  for(auto& p:points){int present;if(!(input>>present>>p.point.x>>p.point.y>>p.point.z)||present<0||present>1)return 4;p.present=present;}
  for(auto& p:pixels)for(int v=0;v<2;++v){int present;if(!(input>>present>>p.point[v].x>>p.point[v].y>>p.sigma[v])||present<0||present>1)return 4;p.present[v]=present;}
  std::int64_t raw;int hsource,hside,valid,continuous;fhn::Rigid head;
  if(!(heads>>raw>>hsource>>hside>>valid>>continuous>>head.q.w>>head.q.x>>head.q.y>>head.q.z>>head.t.x>>head.t.y>>head.t.z)||raw!=stamp||hsource!=source||hside!=side)return 6;
  const auto baseline=stage.fit(reference,rigs[side],side,points,cameras,pixels,stamp,source);if(!baseline.initialized)return 8;
  auto began=std::chrono::steady_clock::now();
  auto candidate=history.refine(rigs[side],baseline.initial.pose,baseline.refined.fit,cameras,points,pixels,stamp,source,valid?&head:nullptr,continuous);
  double us=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-began).count();
  const auto& selected=candidate.fit;bool eligible=candidate.eligible,used=candidate.used;double cost=candidate.alternate_cost;
  out<<stamp<<' '<<source<<' '<<side<<' '<<eligible<<' '<<used<<' '<<us<<' '<<cost;
  for(const auto* fit:std::array<const fhn::PixelFitResult*,2>{&baseline.refined.fit,&selected}){
   out<<' '<<fit->cost<<' '<<fit->observed_rmse_px;auto p=fit->pose;out<<' '<<p.wrist.q.w<<' '<<p.wrist.q.x<<' '<<p.wrist.q.y<<' '<<p.wrist.q.z<<' '<<p.wrist.t.x<<' '<<p.wrist.t.y<<' '<<p.wrist.t.z;
   for(auto angle:p.angles)out<<' '<<angle;
   for(auto point:rigs[side].landmarks(p.angles,p.wrist,side))out<<' '<<point.x<<' '<<point.y<<' '<<point.z;
  }
  out<<'\n';++count;
 }
 std::string extra;if(!input.eof()||heads>>extra||!out.good()||!count)return 5;std::cout<<"hands="<<count<<'\n';return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 7;}}
