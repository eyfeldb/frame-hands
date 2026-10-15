#include "rig_fit_quality.h"
#include <fstream>
#include <iostream>
#include <iomanip>
int main(int argc,char**argv){try{
 if(argc!=6)return 2;std::ifstream left(argv[1]),right(argv[2]),cal(argv[3]),input(argv[4]);if(std::ifstream(argv[5]).good())return 3;
 std::array<fhn::HandRig,2> rigs{fhn::HandRig::load(left),fhn::HandRig::load(right)};auto cameras=fhn::load_rig_cameras(cal);
 std::ofstream out(argv[5]);out<<std::setprecision(17);fhn::RigPose pose;int count=0;
 while(input>>pose.side){if(pose.side<0||pose.side>1)return 4;
  if(!(input>>pose.wrist.q.w>>pose.wrist.q.x>>pose.wrist.q.y>>pose.wrist.q.z>>pose.wrist.t.x>>pose.wrist.t.y>>pose.wrist.t.z))return 4;
  for(auto& x:pose.angles)if(!(input>>x))return 4;std::array<fhn::PixelObservation,21> obs{};
  for(auto& o:obs)for(int v=0;v<2;++v){int present;if(!(input>>present>>o.point[v].x>>o.point[v].y>>o.sigma[v])||present<0||present>1)return 4;o.present[v]=present;}
  const auto q=fhn::rig_fit_quality(rigs[pose.side],pose,cameras,obs);out<<q.projectable;
  for(const auto* part:std::array<const fhn::RigQualityPart*,3>{&q.all,&q.palm,&q.fingers})out<<' '<<part->observations<<' '<<part->rmse_px()<<' '<<part->rmse_normalized();out<<'\n';++count;
 }
 if(!count||!input.eof()||!out.good())return 5;std::cout<<"quality_poses="<<count<<'\n';return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
