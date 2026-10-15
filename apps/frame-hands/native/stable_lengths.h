#pragma once
#include "../pc-input/stability.h"
#include <vector>
namespace fhn {
// Experimental postprocess only; does not create observations or choose identity.
// Calibration samples must be fresh stereo with externally checked reprojection.
struct StableLengths {
    uint32_t id=0; uint64_t stamp=0;
    std::array<std::vector<double>,21> samples;
    std::array<double,21> fixed{};
    static bool finger(int i){return i>0 && i!=1 && i!=5 && i!=9 && i!=13 && i!=17;}
    void reset(){id=0;stamp=0;fixed={};for(auto& v:samples)v.clear();}
    static double median(std::vector<double> x){std::sort(x.begin(),x.end());return x[x.size()/2];}
    bool apply(const std::array<fhs::V,21>& raw,const std::array<fhs::V,21>& filtered,
               uint32_t next,uint64_t capture,bool good,std::array<fhs::V,21>& out){
        out=filtered;
        if(!next || !fhs::Stability::geometry(raw) || !fhs::Stability::geometry(filtered))return false;
        if(id && capture<=stamp)return false;
        if(id!=next || capture-stamp>fhs::Stability::hold_ms){reset();id=next;}
        stamp=capture;
        for(int i=1;i<21;++i){
            if(!finger(i))continue;
            int parent=fhs::Stability::parent[i];
            if(!fixed[i] && good){
                auto& v=samples[i];v.push_back((raw[i]-raw[parent]).len());
                if(v.size()>15)v.erase(v.begin());
                if(v.size()==15){
                    double m=median(v);std::vector<double> deviations;
                    for(double x:v)deviations.push_back(std::abs(x-m));
                    // Robust variation gate; first qualifying window, no future data.
                    if(median(deviations)<=.10*m)fixed[i]=m;
                }
            }
            double length=fixed[i]?fixed[i]:(filtered[i]-filtered[parent]).len();
            out[i]=out[parent]+(filtered[i]-filtered[parent]).unit()*length;
        }
        if(!fhs::Stability::geometry(out)){out=filtered;return false;}
        return true;
    }
    int count()const{int n=0;for(double x:fixed)n+=x>0;return n;}
};
}
