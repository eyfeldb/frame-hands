#pragma once
#include "fit_rig_pixels.h"
namespace fhn {
// Reprojection diagnostics only. Agreement with detector observations is neither
// visibility nor calibrated confidence. No pose, availability or input mutation.
struct RigQualityPart {
    int observations=0;double squared_pixels=0,squared_normalized=0;
    double rmse_px()const{return observations?std::sqrt(squared_pixels/observations):0.;}
    double rmse_normalized()const{return observations?std::sqrt(squared_normalized/observations):0.;}
};
struct RigFitQuality {bool projectable=true;RigQualityPart all{},palm{},fingers{};};
inline RigFitQuality rig_fit_quality(const HandRig& rig,const RigPose& pose,
    const std::array<RigCamera,2>& cameras,const std::array<PixelObservation,21>& observations){
    for(const auto& c:cameras)c.validate();for(const auto& o:observations)o.validate();
    const auto points=rig.landmarks(pose.angles,pose.wrist,pose.side);RigFitQuality out;
    for(int i=0;i<21;++i)for(int v=0;v<2;++v)if(observations[i].present[v]){
        V uv;if(!cameras[v].project(points[i],uv)){out.projectable=false;continue;}
        const auto d=uv-V{observations[i].point[v].x,observations[i].point[v].y,0};double e=d.dot(d);
        double normalized=e/(observations[i].sigma[v]*observations[i].sigma[v]);
        if(!std::isfinite(e)||!std::isfinite(normalized))throw std::invalid_argument("Unbounded quality residual");
        bool palm=i==5||i==8||i==11||i==14||i==17;
        for(auto* part:std::array<RigQualityPart*,2>{&out.all,palm?&out.palm:&out.fingers}){
            ++part->observations;part->squared_pixels+=e;part->squared_normalized+=normalized;
            if(!std::isfinite(part->squared_pixels)||!std::isfinite(part->squared_normalized))throw std::invalid_argument("Unbounded accumulated quality");}
    }
    return out;
}
}
