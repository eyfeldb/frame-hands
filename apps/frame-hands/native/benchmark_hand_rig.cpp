#include "hand_rig.h"
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>
struct Pose { int side; fhn::Rigid wrist; std::array<double,20> angles; };
int main(int argc,char** argv){try{
 if(argc!=3)return 2;std::ifstream file(argv[1]),input(argv[2]);if(!file||!input)return 3;
 const auto rig=fhn::HandRig::load(file);std::vector<Pose> poses;Pose p;
 while(input>>p.side){if(!(input>>p.wrist.q.w>>p.wrist.q.x>>p.wrist.q.y>>p.wrist.q.z>>p.wrist.t.x>>p.wrist.t.y>>p.wrist.t.z))return 4;for(double& a:p.angles)if(!(input>>a))return 4;poses.push_back(p);if(poses.size()>4096)return 5;}
 if(!input.eof()||poses.empty())return 6;
 volatile double checksum=0;std::vector<double> us;us.reserve(20*poses.size());
 using Clock=std::chrono::steady_clock;
 for(const auto& row:poses){auto out=rig.landmarks(row.angles,row.wrist,row.side);checksum+=out[0].x;}
 const auto start=Clock::now();
 for(int repeat=0;repeat<20;++repeat)for(const auto& row:poses){
  auto t=Clock::now();auto out=rig.landmarks(row.angles,row.wrist,row.side);auto end=Clock::now();checksum+=out[(repeat+row.side)%21].x;
  us.push_back(std::chrono::duration<double,std::micro>(end-t).count());
  if(std::chrono::duration<double>(end-start).count()>10)return 8;
 }
 std::sort(us.begin(),us.end());double mean=0;for(double x:us)mean+=x/us.size();
 std::cout<<std::setprecision(12)<<"{\"poses\":"<<poses.size()<<",\"calls\":"<<us.size()<<",\"median_us\":"<<us[us.size()/2]<<",\"p95_us\":"<<us[(us.size()-1)*95/100]<<",\"max_us\":"<<us.back()<<",\"mean_us\":"<<mean<<",\"checksum\":"<<checksum<<"}\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 7;}}
