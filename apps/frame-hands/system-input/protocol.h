#pragma once
#include <cmath>
#include <cstdint>
#include <cstring>
#include <ctime>

namespace fh {
inline uint64_t now_ns() { timespec t{}; clock_gettime(CLOCK_MONOTONIC,&t); return uint64_t(t.tv_sec)*1000000000ull+t.tv_nsec; }
constexpr uint64_t expiry_ns=150000000;
enum : uint32_t { Visible=1, Click=2, Claim=4 };
struct Packet {
    uint32_t magic=0x46485031,version=1,size=sizeof(Packet),flags=0;
    uint64_t sequence=0,sent_ns=0;
    double position[3]{},quaternion[4]{1,0,0,0}; // raw universe; quaternion w,x,y,z
};
static_assert(sizeof(Packet)==88,"wire layout");
inline bool fresh(uint64_t stamp,uint64_t now,uint64_t limit=expiry_ns) {
    return stamp && stamp<=now && now-stamp<=limit;
}
inline bool valid(const Packet& p,uint64_t now) {
    if(p.magic!=0x46485031 || p.version!=1 || p.size!=sizeof p || (p.flags&~7u) || !p.sequence || !fresh(p.sent_ns,now))return false;
    if((p.flags&(Click|Claim)) && !(p.flags&Visible))return false;
    for(double x:p.position)if(!std::isfinite(x)||std::abs(x)>100)return false;
    double norm=0;for(double x:p.quaternion){if(!std::isfinite(x))return false;norm+=x*x;}
    return norm>.98 && norm<1.02;
}
// Reject old/reordered packets; a held click cannot outlive its last source sample.
struct Mailbox {
    Packet packet{}; uint64_t last_sequence=0;bool require_release=true;
    bool accept(const Packet& p,uint64_t now) {
        if(!valid(p,now)||p.sequence<=last_sequence)return false;
        packet=p;last_sequence=p.sequence;return true;
    }
    Packet read(uint64_t now,bool head_valid) {
        Packet p=packet;
        if(!head_valid || !fresh(p.sent_ns,now)){p.flags=0;require_release=true;}
        else if(!(p.flags&Click))require_release=false;
        if(require_release)p.flags&=~uint32_t(Click);
        return p;
    }
};
// Need a fresh open hand before every new interaction after a loss/ID change.
struct PinchGate {
    uint32_t id=0;bool armed=false,down=false;
    bool update(bool valid,uint32_t hand_id,bool pinching) {
        if(!valid || !hand_id || id!=hand_id){id=valid?hand_id:0;armed=false;down=false;}
        if(!valid || !hand_id)return false;
        if(!pinching){armed=true;down=false;}else down=armed;
        return down;
    }
};
// Contact from in front, then retract to rearm. Invalid samples never hold input.
struct TouchGate {
    bool armed=false,down=false;uint32_t id=0;
    bool update(bool valid,uint32_t hand_id,double gap) {
        if(!valid || !std::isfinite(gap)||gap<-.025||gap>.09||id!=hand_id) {
            armed=false;down=false;id=valid?hand_id:0;
        }
        if(!valid || !hand_id || !std::isfinite(gap)||gap<-.025||gap>.09)return false;
        if(gap>=.018){armed=true;down=false;}
        else if(armed && gap<=.003)down=true;
        return down;
    }
};
// SteamVR binds a newly connected device asynchronously. Claim the laser after
// a short bind window and hold the non-click action across several VR frames.
struct ClaimPulse {
    uint64_t connected_at=0;
    bool update(bool visible,uint64_t now) {
        if(!visible){connected_at=0;return false;}
        if(!connected_at||now<connected_at)connected_at=now;
        auto age=now-connected_at;
        return age>=300000000&&age<380000000;
    }
};
}
