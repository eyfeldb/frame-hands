#pragma once
#include "math.h"
#include <cstdint>
#include <cstring>
namespace fhs {
constexpr uint64_t expiry_ms=150;
enum : uint32_t { Present=1, Stereo=2, Controls=1 };
struct Hand { uint32_t id=0,flags=0; float points[21][3]{}; };
struct Packet {
    char magic[8]={'F','H','P','C','V','R','1',0};
    uint32_t version=1,size=568;
    uint64_t sequence=0,sent_ms=0,capture_ms=0;
    uint32_t options=0,reserved=0;
    Hand hands[2]{};
};
static_assert(sizeof(Hand)==260&&sizeof(Packet)==568,"wire layout");
inline bool valid_envelope(const Packet& p,uint64_t now){
    if(std::memcmp(p.magic,"FHPCVR1",8)||p.version!=1||p.size!=sizeof p||!p.sequence||p.reserved||p.options&~Controls)return false;
    if(p.capture_ms>p.sent_ms||p.sent_ms>now||now-p.capture_ms>expiry_ms||now-p.sent_ms>expiry_ms)return false;
    return true;
}
enum class HandIssue { None, Flags, Identity, Nonfinite, Range, Palm, Segment };
inline const char* hand_issue_name(HandIssue reason){
    switch(reason){
        case HandIssue::None:return "valid"; case HandIssue::Flags:return "flags";
        case HandIssue::Identity:return "identity"; case HandIssue::Nonfinite:return "nonfinite";
        case HandIssue::Range:return "range"; case HandIssue::Palm:return "degenerate_palm";
        case HandIssue::Segment:return "segment_length";
    }
    return "unknown";
}
inline HandIssue hand_issue(const Hand& h){
        if(h.flags&~(Present|Stereo))return HandIssue::Flags;
        if(!(h.flags&Present))return HandIssue::None;
        if(!h.id)return HandIssue::Identity;
        for(auto& a:h.points){V v{a[0],a[1],a[2]};if(!v.finite())return HandIssue::Nonfinite;if(v.len()>2.5)return HandIssue::Range;}
        auto at=[&](int i){return V{h.points[i][0],h.points[i][1],h.points[i][2]};};
        if((at(9)-at(0)).len()<.025||(at(5)-at(17)).len()<.015||(at(9)-at(0)).cross(at(5)-at(17)).len()<.0003)return HandIssue::Palm;
        constexpr int parents[21]={0,0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19};
        for(int i=1;i<21;++i){auto n=(at(i)-at(parents[i])).len();if(n<.003||n>.16)return HandIssue::Segment;}
    return HandIssue::None;
}
inline bool valid_hand(const Hand& h){return hand_issue(h)==HandIssue::None;}
inline bool valid(const Packet& p,uint64_t now){
    if(!valid_envelope(p,now))return false;
    for(const auto& h:p.hands)if(!valid_hand(h))return false;
    if((p.hands[0].flags&Present)&&(p.hands[1].flags&Present)&&p.hands[0].id==p.hands[1].id)return false;
    return true;
}
// A repeat never renews the capture timestamp or brings an expired hand back.
struct Mailbox {
    Packet packet{};uint64_t last_sequence=0;
    bool accept(const Packet& p,uint64_t now){
        if(!valid_envelope(p,now)||p.sequence<=last_sequence)return false;
        packet=p;
        for(auto& h:packet.hands)if(!valid_hand(h))h={};
        if((packet.hands[0].flags&Present)&&(packet.hands[1].flags&Present)&&packet.hands[0].id==packet.hands[1].id)packet.hands[0]=packet.hands[1]={};
        last_sequence=p.sequence;return true;
    }
    bool fresh(uint64_t now)const{return last_sequence&&valid(packet,now);}
};
}
