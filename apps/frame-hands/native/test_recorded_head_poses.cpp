#include "recorded_head_poses.h"
#include <fstream>
#include <iostream>
int main(int argc,char**argv){try{
 int checks=0;auto check=[&](bool b){++checks;if(!b)throw std::runtime_error("Recorded head control failed");};
 const std::string header="FHHEAD 1 CLOCK_MONOTONIC raw_world_mm "+std::string(64,'a')+"\n";
 std::istringstream good(header+"100 90 1 1 0 0 0 10 20 30\n200 180 0 1 0 0 0 0 0 0\n");fhn::RecordedHeadPoses poses(good);
 check(poses.size()==2);check(!poses.at(100).usable(100));check(poses.at(100).usable(100,true));check(!poses.at(101).usable(101,true));check(!poses.at(200).usable(200,true));check(poses.at(100).query_ns==0&&poses.at(100).returned_ns==0);check(poses.at(100).head.t.x==10);
 auto rejects=[&](const std::string& text){bool refused=false;try{std::istringstream stream(text);fhn::RecordedHeadPoses bad(stream);}catch(const std::invalid_argument&){refused=true;}check(refused);};
 rejects(header);rejects("FHHEAD 1 CLOCK_MONOTONIC_RAW raw_world_mm "+std::string(64,'a')+"\n");
 for(const auto& line:{"-1 90 1 1 0 0 0 0 0 0","100 -1 1 1 0 0 0 0 0 0","100 90 2 1 0 0 0 0 0 0","100 90 1 2 0 0 0 0 0 0","100 90 1 1 0 0 0 nan 0 0","100.5 90 1 1 0 0 0 0 0 0","100 90 1 1 0 0 0 0 0 0 extra","100 90 1 1 0 0 0 0 0"})rejects(header+line+"\n");
 rejects(header+"100 90 1 1 0 0 0 0 0 0\n100 91 1 1 0 0 0 0 0 0\n");rejects(header+"100 90 1 1 0 0 0 0 0 0\n101 89 1 1 0 0 0 0 0 0\n");
 if(argc==2){std::ifstream input(argv[1]);fhn::RecordedHeadPoses real(input);check(real.size()==1804);std::cout<<"recorded_frames="<<real.size()<<'\n';}
 std::cout<<"controls="<<checks<<" failures=0\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
