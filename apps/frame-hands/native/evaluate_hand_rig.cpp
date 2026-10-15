// Portable C++ evaluator. Profile and saved poses only; no camera, Torch or XR.
#include "hand_rig.h"
#include <fstream>
#include <iomanip>
#include <iostream>
int main(int argc,char** argv){
 try{
    if(argc!=4)return 2;std::ifstream profile(argv[1]),input(argv[2]);if(!profile||!input||std::ifstream(argv[3]).good())return 3;
    const auto rig=fhn::HandRig::load(profile);std::ofstream output(argv[3]);if(!output)return 4;output<<std::setprecision(17);
    int side;unsigned count=0;
    while(input>>side){fhn::Rigid w;std::array<double,20>a{};
        if(!(input>>w.q.w>>w.q.x>>w.q.y>>w.q.z>>w.t.x>>w.t.y>>w.t.z))return 5;
        for(auto& v:a)if(!(input>>v))return 5;
        auto points=rig.landmarks(a,w,side);output<<side;for(auto p:points)output<<' '<<p.x<<' '<<p.y<<' '<<p.z;output<<'\n';++count;
    }
    if(!input.eof()||!output.good()||!count)return 6;std::cout<<"poses="<<count<<"\n";return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 7;}
}
