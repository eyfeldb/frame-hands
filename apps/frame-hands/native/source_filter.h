#pragma once
#include "../pc-input/math.h"
#include "../recovery/stereo_sanity.h"
#include <cstdint>
namespace fhn {
// Isolated source-stage candidate. Head-relative metres; capture time nanoseconds.
// Baseline reproduces Tracker::smooth; finger mode changes only local adaptation.
struct SourceFilter {
    std::array<fhs::V,21> value{},finger_velocity{};
    fhs::V palm_velocity{};int64_t stamp=0;unsigned frames=0,guard_resets=0;
    static bool plausible(const std::array<fhs::V,21>& p){
        std::array<double,3> q[21];for(int i=0;i<21;++i)q[i]={p[i].x,p[i].y,p[i].z};
        return frame_rescue::plausible(q);
    }
    static fhs::V palm(const std::array<fhs::V,21>& p){return (p[0]+p[5]+p[9]+p[13]+p[17])*.2;}
    void reset(){stamp=0;frames=0;palm_velocity={};finger_velocity={};}
    bool update(const std::array<fhs::V,21>& raw,int64_t t,bool finger_aware,bool guard=true){
        if(frames && t<=stamp)return false;
        for(auto p:raw)if(!p.finite())return false;
        double dt=double(t-stamp)/1e9;stamp=t;++frames;
        if(frames<=1||dt<=0||dt>.3){value=raw;palm_velocity={};finger_velocity={};return true;}
        auto alpha=[dt](double hz){return 1/(1+1/(2*3.14159265358979323846*hz*dt));};
        auto before=value;
        fhs::V d=(palm(raw)-palm(before))*(1/dt);
        palm_velocity=palm_velocity+(d-palm_velocity)*alpha(1.5);
        double cutoff=2.+30.*palm_velocity.len(),a=alpha(cutoff);
        for(int i=0;i<21;++i)value[i]=before[i]+(raw[i]-before[i])*a;
        if(finger_aware)for(int first:{1,6,10,14,18}){
            const int anchor=first==1?0:first-1,last=first==1?4:first+2;
            double speed=0;
            for(int i=first;i<=last;++i){
                auto old_local=before[i]-before[anchor],new_local=raw[i]-raw[anchor];
                auto v=(new_local-old_local)*(1/dt);
                finger_velocity[i]=finger_velocity[i]+(v-finger_velocity[i])*alpha(1.5);
                speed=std::max(speed,finger_velocity[i].len());
            }
            double fa=alpha(std::max(cutoff,2.+30.*speed));
            for(int i=first;i<=last;++i){
                auto old_local=before[i]-before[anchor],new_local=raw[i]-raw[anchor];
                value[i]=value[anchor]+old_local+(new_local-old_local)*fa;
            }
        }
        if(guard&&plausible(raw)&&!plausible(value)){value=raw;++guard_resets;}
        return true;
    }
};
}
