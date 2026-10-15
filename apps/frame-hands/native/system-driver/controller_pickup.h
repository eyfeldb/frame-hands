#pragma once
#include <cmath>
#include <cstdint>
namespace fhnav {
// Pick-up detector for a physical controller, from its tracked pose only (2026-10-12). While a PC game is
// streaming, SteamVR delivers button presses only to the focused application, so the button-based takeover in
// ButtonGate cannot see them; device poses are always readable. A controller counts as picked up when it has
// rested (speed below rest_speed for rest_ns) and then moves faster than lift_speed for lift_ns AND is at least
// lift_distance away from where it rested. A nudge on the desk fails the distance test; a quick bump fails the
// duration test; a controller already in motion when Hands were selected must rest first.
struct PickupGate {
    static constexpr double rest_speed=0.05,lift_speed=0.25,lift_distance=0.10;
    static constexpr uint64_t rest_ns=1500000000ull,lift_ns=300000000ull;
    bool have_last=false,resting=false;double lx=0,ly=0,lz=0,rx=0,ry=0,rz=0;uint64_t last_t=0,still_since=0,moving_since=0;
    void reset(){have_last=false;resting=false;still_since=0;moving_since=0;}
    bool sample(bool tracked,double x,double y,double z,uint64_t now){
        if(!tracked){reset();return false;}
        if(!have_last||now<=last_t){have_last=true;lx=x;ly=y;lz=z;last_t=now;return false;}
        const double dt=double(now-last_t)*1e-9;if(dt<0.004)return false;
        const double speed=std::sqrt((x-lx)*(x-lx)+(y-ly)*(y-ly)+(z-lz)*(z-lz))/dt;
        lx=x;ly=y;lz=z;last_t=now;
        if(speed<rest_speed){
            if(!still_since)still_since=now;
            if(now-still_since>=rest_ns){resting=true;rx=x;ry=y;rz=z;}
            moving_since=0;return false;
        }
        still_since=0;
        if(!resting)return false;
        if(speed>lift_speed){if(!moving_since)moving_since=now;}
        else{moving_since=0;return false;}
        const double away=std::sqrt((x-rx)*(x-rx)+(y-ry)*(y-ry)+(z-rz)*(z-rz));
        if(now-moving_since>=lift_ns&&away>=lift_distance){resting=false;moving_since=0;return true;}
        return false;
    }
};
}
