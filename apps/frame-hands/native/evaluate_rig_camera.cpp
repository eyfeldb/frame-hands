#include "fit_rig_pixels.h"
#include <fstream>
#include <iomanip>
#include <iostream>
int main(int argc,char** argv){try{if(argc!=4)return 2;std::ifstream calibration(argv[1]),input(argv[2]);if(!calibration||!input||std::ifstream(argv[3]).good())return 3;auto cams=fhn::load_rig_cameras(calibration);std::ofstream output(argv[3]);output<<std::setprecision(17);fhn::V p;unsigned count=0;
while(input>>p.x){if(!(input>>p.y>>p.z))return 5;for(auto& c:cams){fhn::V uv;if(!c.project(p,uv))throw std::invalid_argument("Point behind camera");output<<uv.x<<' '<<uv.y<<' ';}output<<'\n';++count;}if(!input.eof()||!output.good()||!count)return 6;return 0;}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 7;}}
