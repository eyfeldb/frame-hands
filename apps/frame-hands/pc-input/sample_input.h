#pragma once
#include "protocol.h"
namespace fhs {
// Use the same independent-hand mailbox as vrserver. A bad left hand must not
// remove the right white/source skeleton from the diagnostic view.
struct SampleInput {
    Mailbox mailbox;
    uint64_t observations=0,duplicate_identity=0,absent[2]{},issues[2][7]{};
    HandIssue last_issue[2]{};
    uint32_t last_id[2]{},last_flags[2]{},bad_segments[2]{};
    bool last_duplicate=false;
    bool observe(const Packet& packet,uint64_t now){
        if(!mailbox.accept(packet,now))return false;
        ++observations;
        last_duplicate=(packet.hands[0].flags&Present)&&(packet.hands[1].flags&Present)&&packet.hands[0].id==packet.hands[1].id;
        duplicate_identity+=last_duplicate;
        for(int side=0;side<2;++side){
            const auto& h=packet.hands[side];last_issue[side]=hand_issue(h);
            last_id[side]=h.id;last_flags[side]=h.flags;bad_segments[side]=0;
            if(last_issue[side]==HandIssue::Segment){
                constexpr int parent[21]={0,0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19};
                for(int i=1;i<21;++i){auto a=h.points[i],b=h.points[parent[i]];
                    double length=(V{a[0],a[1],a[2]}-V{b[0],b[1],b[2]}).len();
                    if(length<.003||length>.16)bad_segments[side]|=1u<<i;
                }
            }
            ++issues[side][int(last_issue[side])];
            absent[side]+=!(packet.hands[side].flags&Present);
        }
        return true;
    }
};
}
