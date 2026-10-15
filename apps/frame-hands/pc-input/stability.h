#pragma once
#include "protocol.h"

namespace fhs {
// Diagnostic candidate only. World-space input; no extrapolation or input synthesis.
// Observation time, never render time, controls filtering and the short hold budget.
struct Stability {
    static constexpr uint64_t hold_ms=80;
    static constexpr int parent[21]={0,0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19};
    uint32_t id=0;
    uint64_t stamp=0,observed=0,pending_stamp=0;
    unsigned rejected=0,accepted=0,fallbacks=0;
    std::array<V,21> value{},previous{},pending{};
    std::array<double,21> speed{};
    void reset(){id=0;stamp=observed=pending_stamp=0;speed={};}
    static double alpha(double hz,double dt){double r=6.283185307179586*hz*dt;return r/(1+r);}
    static bool geometry(const std::array<V,21>& p){
        // Translate before validation: the bridge range limit is head-relative,
        // whereas this module receives world-space positions.
        Hand h;h.id=1;h.flags=Present;
        for(int i=0;i<21;++i){V v=p[i]-p[0];if(!p[i].finite())return false;h.points[i][0]=float(v.x);h.points[i][1]=float(v.y);h.points[i][2]=float(v.z);}
        return valid_hand(h);
    }
    static double distance(const std::array<V,21>& a,const std::array<V,21>& b){
        double d=0;for(int i=0;i<21;++i)d=std::max(d,(a[i]-b[i]).len());return d;
    }
    bool update(const std::array<V,21>& p,uint32_t next,uint64_t capture,uint64_t now){
        if(!next||capture>now||now-capture>hold_ms||!geometry(p))return false;
        if(id&&capture<=observed)return false; // duplicates cannot renew the hold
        if(id!=next||!id||capture-stamp>hold_ms){
            reset();id=next;stamp=observed=capture;value=previous=p;++accepted;return true;
        }
        double dt=double(capture-stamp)/1000.;observed=capture;
        // Quarantine a spatial discontinuity for one observation. A second nearby
        // observation confirms real rapid relocation; returning to the old track
        // discards the spike. This threshold is a hypothesis, not a speed limit.
        bool jump=distance(previous,p)>.08+4.*dt;
        if(jump&&!(pending_stamp&&capture-pending_stamp<=50&&distance(pending,p)<.06+4.*double(capture-pending_stamp)/1000.)){
            pending=p;pending_stamp=capture;++rejected;return false;
        }
        pending_stamp=0;
        auto before=value;
        double da=alpha(1.5,dt);
        speed[0]+=( (p[0]-previous[0]).len()/dt-speed[0])*da;
        double a=alpha(3.+20.*speed[0],dt);
        value[0]=before[0]*(1-a)+p[0]*a;
        for(int i=1;i<21;++i){
            V old=before[i]-before[parent[i]],b=p[i]-p[parent[i]];
            V raw=previous[i]-previous[parent[i]];
            double angle=std::acos(std::clamp(raw.unit().dot(b.unit()),-1.,1.));
            speed[i]+=(angle/dt-speed[i])*da;
            double blend=alpha(3.+2.*speed[i],dt);
            // Quaternion swing interpolation also handles opposite directions;
            // Cartesian interpolation would collapse the bone to zero length.
            Q q=swing(old,b);if(q.w<0)q={-q.w,-q.x,-q.y,-q.z};
            double half=std::acos(std::clamp(q.w,-1.,1.));
            Q part;
            if(half>1e-8){double k=std::sin(half*blend)/std::sin(half);part={std::cos(half*blend),q.x*k,q.y*k,q.z*k};}
            double length=old.len()+(b.len()-old.len())*alpha(2.,dt);
            value[i]=value[parent[i]]+part.rotate(old.unit())*length;
        }
        if(!geometry(value)){value=p;speed={};++fallbacks;}
        previous=p;stamp=capture;++accepted;return true;
    }
    bool visible(uint64_t now)const{return id&&now>=stamp&&now-stamp<=hold_ms;}
};
}
