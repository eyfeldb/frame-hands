#ifdef NDEBUG
#undef NDEBUG
#endif
#include "rig_snapshot.h"
#include <cassert>
#include <fstream>
#include <iostream>
#include <limits>
int main(int argc,char**argv){
 if(argc!=2||std::ifstream(argv[1]).good())return 2;
 fhn::TrackerRigOutput h;h.source=7;h.side=0;h.stamp=9000000000;h.status="fitted";h.result.initialized=true;
 h.result.refined.fit.pose.wrist.t={100,200,-500};
 auto serialize=[&](const fhn::TrackerRigOutput& v){return fhn::rig_snapshot(1,9000000000,1000000000,1020000000,{v});};
 const auto valid=serialize(h);assert(valid.find("CLOCK_MONOTONIC")!=std::string::npos);
 for(int mode=0;mode<6;++mode){auto bad=h;bool refused=false;
  if(mode==0)bad.stamp++;
  if(mode==1)bad.side=1;
  if(mode==2)bad.status="unknown";
  if(mode==3)bad.result.initialized=false;
  if(mode==4)bad.result.refined.fit.pose.wrist.q.w=2;
  if(mode==5)bad.result.refined.fit.pose.angles[0]=std::numeric_limits<double>::quiet_NaN();
  try{serialize(bad);}catch(const std::invalid_argument&){refused=true;}assert(refused);
 }
 bool refused=false;try{fhn::rig_snapshot(1,9,20,10,{});}catch(const std::invalid_argument&){refused=true;}assert(refused);
 std::ofstream output(argv[1]);output<<valid<<fhn::rig_snapshot(2,9033333333,1033333333,1053333333,{});output.close();assert(output.good());
 std::cout<<"PASS snapshot clocks, pose/status coherence, finite angles, rigid wrist and empty frame serialization\n";
}
