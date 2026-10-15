#pragma once
#include <cmath>
#include <cstdint>
namespace fhnav {
// Controller use to navigate/dismiss the dashboard must not undo its Hands
// selection. The final dashboard sample also covers the dismissal event.
struct TakeoverWindow {
    uint64_t epoch=0, settle=0;
    bool update(uint64_t sequence,bool dashboard,uint64_t now){
        const bool reset=sequence!=epoch||dashboard;
        if(reset){epoch=sequence;settle=now+300000000;}
        return reset;
    }
    bool ready(uint64_t now)const{return now>=settle;}
};
// Selecting Hands consumes the activation click. Observe a stable release before
// allowing a subsequent press; one neutral component sample is insufficient.
struct ButtonGate {
    bool armed=false;
    uint64_t last=0, release_since=0, armed_at=0;
    static constexpr uint64_t release_ns=200000000;
    void reset(){armed=false;last=release_since=armed_at=0;}
    bool sample(bool available,uint64_t buttons,uint64_t now){
        if(!available){reset();return false;}
        if(!armed){
            if(buttons)release_since=0;
            else if(!release_since)release_since=now;
            else if(now-release_since>=release_ns){armed=true;armed_at=now;}
            last=buttons;
            return false;
        }
        bool press=(buttons&~last)!=0;last=buttons;return press;
    }
    bool fresh_event(uint64_t now,float age_seconds,uint64_t settle)const{
        if(!armed||!std::isfinite(age_seconds)||age_seconds<0)return false;
        const double age=double(age_seconds)*1000000000.;
        if(age>double(now))return false;
        const uint64_t occurred=now-uint64_t(age);
        return occurred>=armed_at&&occurred>=settle;
    }
};
}
