#pragma once
#include "stereo_disagreement.h"
namespace fhn {
// Experimental correspondence hypotheses only. No presence/identity/input state.
constexpr std::array<std::pair<int,int>,4> index_middle_pairs{{{8,11},{9,12},{10,13},{1,2}}};
inline std::array<PixelObservation,21> swap_index_middle_view(const std::array<PixelObservation,21>& original,int camera){
    if(camera<0||camera>1)throw std::invalid_argument("Invalid correspondence camera");
    auto out=original;
    for(auto pair:index_middle_pairs){auto& a=out[pair.first];auto& b=out[pair.second];
        std::swap(a.point[camera],b.point[camera]);std::swap(a.sigma[camera],b.sigma[camera]);std::swap(a.present[camera],b.present[camera]);}
    return out;
}
inline double index_middle_disagreement(const std::array<RigCamera,2>& cameras,const std::array<PixelObservation,21>& pixels){
    double square=0;
    for(auto pair:index_middle_pairs)for(int i:{pair.first,pair.second}){
        auto d=stereo_disagreement(cameras,pixels[i]);if(!d.valid)return std::numeric_limits<double>::infinity();
        double z=d.rms_px/std::hypot(pixels[i].sigma[0],pixels[i].sigma[1]);square+=z*z;
    }
    return std::sqrt(square/8.);
}
struct CorrespondenceResult {
    PixelFitResult control{},fit{};int chosen=-1;
    double original_disagreement=0;
    std::array<double,2> alternative_disagreement{},alternative_cost{};
    std::array<bool,2> eligible{};
};
inline CorrespondenceResult refine_index_middle_hypotheses(const HandRig& rig,const RigPose& seed,const PixelFitResult& baseline,
    const std::array<RigCamera,2>& cameras,const std::array<PixelObservation,21>& pixels,const std::array<double,20>& prior){
    CorrespondenceResult out;out.control=baseline;out.fit=out.control;
    out.original_disagreement=index_middle_disagreement(cameras,pixels);
    out.alternative_cost.fill(std::numeric_limits<double>::infinity());
    out.alternative_disagreement.fill(std::numeric_limits<double>::infinity());
    // Reuse the existing 3-sigma robust threshold. Missing geometry is not ambiguity evidence.
    if(!out.control.sufficient||!std::isfinite(out.original_disagreement)||out.original_disagreement<=3)return out;
    for(int v=0;v<2;++v){auto alternate=swap_index_middle_view(pixels,v);
        double disagreement=index_middle_disagreement(cameras,alternate);out.alternative_disagreement[v]=disagreement;
        if(!std::isfinite(disagreement)||disagreement>3)continue;
        out.eligible[v]=true;auto fitted=fit_rig_pixels(rig,seed,cameras,alternate,{},0,&prior);
        out.alternative_cost[v]=fitted.cost;
        // Different correspondence hypotheses: lower cost is only a selection hypothesis,
        // never proof of anatomical truth, tracking confidence or unmodified-pixel accuracy.
        if(fitted.sufficient&&fitted.cost<out.fit.cost){out.fit=fitted;out.chosen=v;}
    }
    return out;
}
inline CorrespondenceResult fit_index_middle_hypotheses(const HandRig& rig,const RigPose& seed,
    const std::array<RigCamera,2>& cameras,const std::array<PixelObservation,21>& pixels){
    auto baseline=fit_rig_pixels(rig,seed,cameras,pixels);
    return refine_index_middle_hypotheses(rig,seed,baseline,cameras,pixels,seed.angles);
}
}
