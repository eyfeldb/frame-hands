#pragma once
#include "skeleton.h"
namespace fhs {
// Offline experiment: preserve measured joint positions and transport the
// unmeasured bone roll from the preceding observation. No position smoothing.
// This can accumulate roll on complex paths; it is not a validated skinning model.
struct RetargetContinuity {
    uint32_t id=0;uint64_t stamp=0;
    std::array<Transform,31> previous{};Skeleton cached{};
    Skeleton update(const std::array<V,21>& points,int side,uint32_t identity,uint64_t capture){
        if(id==identity&&stamp==capture)return cached;
        auto result=retarget(points,side);auto model=forward(result.bones);
        if(id==identity&&stamp&&capture>stamp&&capture-stamp<=80){
            for(int bone=2;bone<26;++bone){
                bool tip=bone==5||bone==10||bone==15||bone==20||bone==25;
                if(tip)model[bone].q=model[bone-1].q;
                else {
                    V old_direction=previous[bone+1].p-previous[bone].p;
                    V direction=model[bone+1].p-model[bone].p;
                    model[bone].q=(swing(old_direction,direction)*previous[bone].q).unit();
                }
            }
            for(int f=0;f<5;++f)model[26+f]=model[4+5*f];
            for(int bone=0;bone<31;++bone){int p=parents[bone];result.bones[bone]=p<0?model[bone]:Transform{model[p].q.inv().rotate(model[bone].p-model[p].p),(model[p].q.inv()*model[bone].q).unit()};}
        }
        previous=model;id=identity;stamp=capture;cached=result;return result;
    }
};
}
