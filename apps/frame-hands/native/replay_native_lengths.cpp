// Saved world-space observations only. No OpenVR, cameras, sockets or input.
#include "stable_lengths.h"
#include <fstream>
#include <iomanip>
int main(int argc,char** argv){
    if(argc!=3)return 2;
    std::ifstream in(argv[1]); if(!in)return 3;
    // Caller creates a fresh private directory; refuse to overwrite an existing file.
    if(std::ifstream(argv[2]).good())return 4;
    std::ofstream out(argv[2]); if(!out)return 5;
    out<<std::setprecision(17);
    fhs::Stability controls[2];fhn::StableLengths candidates[2];
    uint64_t t,last=0;uint32_t id;int side,good;
    while(in>>t>>id>>side>>good){
        if(t<last || side<0 || side>1 || (good!=0 && good!=1))return 6;last=t;
        std::array<fhs::V,21> raw{},candidate{};
        for(auto& v:raw)if(!(in>>v.x>>v.y>>v.z))return 7;
        auto& b=controls[side];auto& c=candidates[side];bool accepted=b.update(raw,id,t,t);
        bool applied=false;
        if(accepted)applied=c.apply(raw,b.value,id,t,good,candidate);
        out<<t<<' '<<id<<' '<<side<<' '<<accepted<<' '<<applied<<' '<<c.count();
        if(accepted)for(const auto& points:{b.value,candidate})for(auto v:points)out<<' '<<v.x<<' '<<v.y<<' '<<v.z;
        out<<'\n';
    }
    if(!in.eof())return 8;return out.good()?0:9;
}
