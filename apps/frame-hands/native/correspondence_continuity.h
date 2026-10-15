#pragma once
#include "finger_correspondence.h"
#include <cstdint>
namespace fhn {
// Short-lived correspondence memory, not a pose/presence hold. Strict evidence
// alone can start it. Optional relative evidence may renew an existing selection.
class CorrespondenceMemory {
    int source_=-1,side_=-1,camera_=-1;bool relative_renewal_=false;
    std::int64_t last_stamp_=0,last_selected_=0,last_strong_=0;
    static constexpr std::int64_t lifetime=100000000;
    void validate(std::int64_t stamp,int source,int side)const{
        if(stamp<=last_stamp_||source<0||side<0||side>1)throw std::invalid_argument("Invalid correspondence lifecycle");
    }
public:
    CorrespondenceMemory()=default;
    explicit CorrespondenceMemory(bool relative_renewal):relative_renewal_(relative_renewal){}
    void set_relative_renewal(bool enabled){if(enabled!=relative_renewal_){clear();relative_renewal_=enabled;}}
    // Historical trace name: with renewal enabled this is latest evidence, not necessarily strict admission.
    void clear(){source_=side_=camera_=-1;last_selected_=last_strong_=0;}
    std::int64_t strong_stamp()const{return last_strong_;}
    int permit(std::int64_t stamp,int source,int side,bool continuous,const CorrespondenceResult& strict)const{
        validate(stamp,source,side);
        if(!continuous||strict.chosen!=-1||!strict.control.sufficient||camera_<0||source!=source_||side!=side_||stamp-last_selected_>lifetime||stamp-last_strong_>lifetime)return -1;
        const double original=strict.original_disagreement,alternative=strict.alternative_disagreement[camera_];
        // Consistent original evidence releases immediately. Do not retry a strict
        // alternative that already lost, or a geometrically worse interpretation.
        if(!std::isfinite(original)||!std::isfinite(alternative)||original<=3||alternative<=3||alternative>=original)return -1;
        return camera_;
    }
    void commit(std::int64_t stamp,int source,int side,bool continuous,
                const CorrespondenceResult& strict,const CorrespondenceResult& selected){
        validate(stamp,source,side);
        const int camera=selected.chosen;
        bool strong=camera>=0&&camera<2&&strict.chosen==camera&&strict.control.sufficient&&std::isfinite(strict.original_disagreement)&&std::isfinite(strict.alternative_disagreement[camera])&&strict.eligible[camera]&&strict.alternative_disagreement[camera]<=3&&strict.original_disagreement>3;
        bool continued=camera>=0&&camera<2&&!strong&&permit(stamp,source,side,continuous,strict)==camera;
        if(camera<-1||camera>1||(camera>=0&&!strong&&!continued))throw std::invalid_argument("Unjustified correspondence selection");
        last_stamp_=stamp;
        if(!continuous||camera<0||!selected.fit.sufficient){clear();return;}
        source_=source;side_=side;camera_=camera;last_selected_=stamp;
        if(strong||(relative_renewal_&&continued&&strict.original_disagreement-strict.alternative_disagreement[camera]>=3))last_strong_=stamp;
    }
};
struct CorrespondenceContinuation {CorrespondenceResult result{};bool continued=false,renewed=false;std::int64_t evidence_age=0,strong_stamp=0;};
inline CorrespondenceContinuation continue_correspondence(CorrespondenceMemory& memory,
    const HandRig& rig,const RigPose& seed,const CorrespondenceResult& strict,
    const std::array<RigCamera,2>& cameras,const std::array<PixelObservation,21>& pixels,
    const std::array<double,20>& prior,std::int64_t stamp,int source,bool continuous){
    CorrespondenceContinuation out;out.result=strict;
    int candidate=memory.permit(stamp,source,seed.side,continuous,strict);
    if(candidate>=0){auto alternate=swap_index_middle_view(pixels,candidate);auto fit=fit_rig_pixels(rig,seed,cameras,alternate,{},0,&prior);
        out.result.alternative_cost[candidate]=fit.cost;
        if(fit.sufficient&&fit.cost<strict.fit.cost){out.result.fit=fit;out.result.chosen=candidate;out.continued=true;out.evidence_age=stamp-memory.strong_stamp();}}
    memory.commit(stamp,source,seed.side,continuous,strict,out.result);out.strong_stamp=memory.strong_stamp();out.renewed=out.continued&&out.strong_stamp==stamp;return out;
}
}
