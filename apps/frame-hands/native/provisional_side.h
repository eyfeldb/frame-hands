#pragma once
#include <cmath>
#include <cstdint>
#include <stdexcept>
namespace fhn {
// Side decision only. Never changes neural scores, source association or freshness.
// Weak scores retain a continuous source's previous side. A strong identity latch
// has priority; otherwise a change needs current qualified stereo agreement.
struct ProvisionalSide {
    bool active=false,right=false;
    // Opt-in experiment: bootstrap from original score until qualified stereo.
    bool stereo_first=false,stereo_supported=false;
    std::int64_t last=0;
    void clear(){active=false;stereo_supported=false;last=0;}
    bool observe(double score,int stereo_vote,std::int64_t now,int strong=-1){
        if(!std::isfinite(score)||score<0||score>1||stereo_vote < -1||stereo_vote>1||strong < -1||strong>1||now<=0)
            throw std::invalid_argument("Invalid provisional side evidence");
        if(now<=last)return false;
        if(last&&now-last>100000000){active=false;stereo_supported=false;}
        last=now;
        if(strong>=0){right=bool(strong);active=true;stereo_supported=true;return true;}
        if(stereo_first){
            if(stereo_vote>=0){right=bool(stereo_vote);stereo_supported=true;}
            else if(!stereo_supported)right=score>.5;
            active=true;return true;
        }
        if(!active){right=score>.5;active=true;return true;}
        if(!right&&score>=.65&&stereo_vote==1)right=true;
        else if(right&&score<=.35&&stereo_vote==0)right=false;
        return true;
    }
};
}
