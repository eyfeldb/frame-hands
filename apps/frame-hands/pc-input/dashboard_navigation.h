#pragma once
#include "model_protocol.h"
namespace fhs {
struct DashboardContext {char magic[8]={'F','H','U','I','C','T','X','1'};uint64_t stamp_ms=0;uint32_t enabled=0,reserved=0;};
static_assert(sizeof(DashboardContext)==24);
inline bool dashboard_context(const DashboardContext& c,uint64_t now){return !std::memcmp(c.magic,"FHUICTX1",8)&&c.enabled==1&&!c.reserved&&c.stamp_ms&&c.stamp_ms<=now&&now-c.stamp_ms<=100;}
struct DashboardPinch {
    uint32_t left=0,right=0;uint64_t capture=0,began=0;bool armed=false,down=false;
    void reset(){*this={};}
    bool update(const ModelPacket& p,uint64_t now,bool enabled){
        if(!enabled||!model_envelope(p,now)){reset();return false;}
        for(const auto& h:p.hands)if(!(h.flags&Present)||!model_hand(h,p.source_ms)||now<h.capture_ms||now>=h.until_ms||now-h.capture_ms>100){reset();return false;}
        const auto& h=p.hands[0]; // left pinch clicks the right-hand pointer
        if(left!=h.id||right!=p.hands[1].id||h.capture_ms<capture){reset();left=h.id;right=p.hands[1].id;}
        if(h.capture_ms==capture)return down; // render/prediction frames cannot create a new press
        capture=h.capture_ms;
        auto distance=[&](int a,int b){V d{h.points[a][0]-h.points[b][0],h.points[a][1]-h.points[b][1],h.points[a][2]-h.points[b][2]};return d.len();};
        double width=distance(5,17),gap=distance(4,8);
        if(!std::isfinite(width)||width<.035||width>.13){reset();return false;}
        // Thresholds from the wearer's 2026-10-10 recording (fitted tips, hand width ~63 mm): real thumb-index
        // contact measured 6-14 mm (pressed: 6-9), a relaxed half-closed hand 16-18 mm for under 0.2 s. Close
        // at 20% of the hand width (11-16 mm) held for 120 ms; a frame drifting between close and open keeps
        // the hold running instead of restarting it, so fingertip jitter around the threshold cannot block a
        // real pinch. The wearer still saw false clicks at 24%/100 ms, hence the tighter band.
        const double close=std::clamp(width*.20,.011,.016),open=std::clamp(width*.50,.030,.045);
        if(gap>=open){armed=true;down=false;began=0;}
        else if(armed&&gap<=close){if(!began)began=capture;else if(capture-began>=120)down=true;}
        return down;
    }
};
}
