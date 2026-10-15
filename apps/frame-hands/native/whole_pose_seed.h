#pragma once
#include "fit_hand_rig.h"
namespace fhn {
// Experimental seed only. Caller establishes fresh stereo, same source/side and age.
// World/head translations are millimeters; no visibility or output hold is implied.
inline bool rig_is_rigid(const Rigid& x){double n=x.q.w*x.q.w+x.q.x*x.q.x+x.q.y*x.q.y+x.q.z*x.q.z;
    return x.t.finite()&&std::isfinite(n)&&std::abs(n-1.)<1e-6;}
inline Rigid rig_inverse(const Rigid& p){auto q=p.q.inv();return{q,q.rotate(p.t*-1.)};}
inline bool whole_pose_seed(const HandRig& rig,const RigPose& previous_world,
                           const Rigid& current_head,const std::array<RigObservation,21>& observations,
                           RigPose& seed){
    if(!rig_is_rigid(current_head)||!rig_is_rigid(previous_world.wrist))throw std::invalid_argument("Nonrigid history/head transform");
    constexpr std::array<int,5> indices{5,8,11,14,17};
    V observed{};for(int i:indices){observations[i].validate();if(!observations[i].present)return false;observed=observed+observations[i].point*.2;}
    RigPose result=previous_world;result.wrist=rig_inverse(current_head).then(previous_world.wrist);
    result.wrist.q=result.wrist.q.unit();
    auto landmarks=rig.landmarks(result.angles,result.wrist,result.side);V predicted{};
    for(int i:indices)predicted=predicted+landmarks[i]*.2;
    result.wrist.t=result.wrist.t+observed-predicted;seed=result;return true;
}
}
