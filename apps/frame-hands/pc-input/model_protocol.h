#pragma once
#include "protocol.h"
namespace fhs {
#pragma pack(push,1)
struct ModelHand { uint32_t id=0,flags=0; uint64_t capture_ms=0,until_ms=0; float points[21][3]{}; };
struct ModelPacket {
    char magic[8]={'F','H','M','O','D','E','L','2'};uint32_t version=2,size=592;
    uint64_t sequence=0,render_ms=0,source_ms=0;ModelHand hands[2]{};
};
#pragma pack(pop)
static_assert(sizeof(ModelHand)==276&&sizeof(ModelPacket)==592,"model wire layout");
inline bool model_envelope(const ModelPacket& p,uint64_t now){
    return !std::memcmp(p.magic,"FHMODEL2",8)&&p.version==2&&p.size==sizeof p&&p.sequence&&p.render_ms<=now&&p.source_ms<=now&&now-p.render_ms<=80&&now-p.source_ms<=500;
}
inline bool model_hand(const ModelHand& h,uint64_t source){
    if(h.flags&~Present)return false;
    if(!(h.flags&Present))return true;
    if(!h.id||h.capture_ms>source||h.until_ms<h.capture_ms||h.until_ms-h.capture_ms>350)return false;
    Hand relative;relative.id=h.id;relative.flags=Present;
    for(int i=0;i<21;++i)for(int j=0;j<3;++j){
        if(!std::isfinite(h.points[i][j])||std::abs(h.points[i][j])>1000)return false;
        relative.points[i][j]=h.points[i][j]-h.points[0][j];
    }
    return valid_hand(relative);
}
struct ModelMailbox {
    ModelPacket packet{};uint64_t last_sequence=0;
    bool accept(const ModelPacket& p,uint64_t now){
        if(!model_envelope(p,now)||p.sequence<=last_sequence)return false;
        packet=p;
        for(auto& h:packet.hands)if(!model_hand(h,p.source_ms))h={};
        if((packet.hands[0].flags&Present)&&(packet.hands[1].flags&Present)&&packet.hands[0].id==packet.hands[1].id)packet.hands[0]=packet.hands[1]={};
        last_sequence=p.sequence;return true;
    }
    bool fresh(uint64_t now)const{return last_sequence&&model_envelope(packet,now);}
};
}
