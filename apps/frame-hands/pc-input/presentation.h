#pragma once
#include "skeleton.h"
namespace fhs {
inline Q blend_rotation(Q a,Q b,double t){
    double dot=a.w*b.w+a.x*b.x+a.y*b.y+a.z*b.z;
    if(dot<0){b={-b.w,-b.x,-b.y,-b.z};dot=-dot;}
    double x=1-t,y=t;
    if(dot<.9995){double angle=std::acos(std::clamp(dot,0.,1.));x=std::sin((1-t)*angle)/std::sin(angle);y=std::sin(t*angle)/std::sin(angle);}
    return Q{x*a.w+y*b.w,x*a.x+y*b.x,x*a.y+y*b.y,x*a.z+y*b.z}.unit();
}
inline V blend_segment(V a,V b,double t){
    if(a.len()<1e-9||b.len()<1e-9)return a*(1-t)+b*t;
    return blend_rotation({},swing(a,b),t).rotate(a.unit())*(a.len()*(1-t)+b.len()*t);
}
// Presentation only: smooth at driver cadence, never predict or renew tracking.
// 20 ms response adds latency; availability is decided by Stability before this.
struct Presentation {
    uint32_t id=0;uint64_t stamp=0,time=0;Skeleton value{};
    void reset(){*this={};}
    Skeleton update(const Skeleton& target,uint32_t identity,uint64_t capture,uint64_t now){
        if(!id||id!=identity||capture<stamp||capture-stamp>80||now<time||now-time>80){
            id=identity;stamp=capture;time=now;value=target;return value;
        }
        double t=1-std::exp(-double(now-time)/20.);time=now;stamp=capture;
        value.position=value.position*(1-t)+target.position*t;
        value.rotation=blend_rotation(value.rotation,target.rotation,t);
        for(int b=0;b<26;++b){
            value.bones[b].p=blend_segment(value.bones[b].p,target.bones[b].p,t);
            value.bones[b].q=blend_rotation(value.bones[b].q,target.bones[b].q,t);
        }
        auto model=forward(value.bones);
        for(int f=0;f<5;++f){auto knuckle=model[4+5*f];value.bones[26+f]={model[0].q.inv().rotate(knuckle.p-model[0].p),(model[0].q.inv()*knuckle.q).unit()};}
        return value;
    }
};
}
