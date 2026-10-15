#pragma once
#include "fit_rig_pixels.h"
#include <vector>
#include <algorithm>
namespace fhn {
// Complete-pose starting guesses, not output templates or a confidence model.
struct PoseBank {
    std::string model_sha,training_sha;std::vector<std::array<double,20>> angles;
    static PoseBank load(std::istream& input){PoseBank bank;std::string magic,extra;int version=0,count=0,joints=0;
        if(!(input>>magic>>version>>count>>joints>>bank.model_sha>>bank.training_sha)||magic!="FHPOSEBANK"||version!=1||count<1||count>32||joints!=20||!HandRig::hash(bank.model_sha)||!HandRig::hash(bank.training_sha))throw std::invalid_argument("Invalid pose bank header");
        bank.angles.resize(count);for(auto& a:bank.angles)for(auto& x:a)if(!(input>>x)||!std::isfinite(x))throw std::invalid_argument("Invalid pose bank row");
        if(input>>extra)throw std::invalid_argument("Trailing pose bank data");return bank;}
    void validate(const HandRig& rig)const{
        if(model_sha!=rig.model_sha||!HandRig::hash(training_sha)||angles.empty()||angles.size()>32)throw std::invalid_argument("Pose bank/model mismatch");
        for(const auto& a:angles)for(int i=0;i<20;++i)if(!std::isfinite(a[i])||a[i]<rig.joints[i].lower||a[i]>rig.joints[i].upper)throw std::invalid_argument("Pose bank outside anatomy limits");
    }
};
inline double pose_bank_cost(const HandRig& rig,const RigPose& pose,const RigPose& reference,
    const std::array<RigCamera,2>& cameras,const std::array<PixelObservation,21>& observations){
    const auto points=rig.landmarks(pose.angles,pose.wrist,pose.side);double value=0;
    for(int i=0;i<21;++i){double square=0;
        for(int v=0;v<2;++v)if(observations[i].present[v]){V uv;if(!cameras[v].project(points[i],uv))return std::numeric_limits<double>::infinity();
            auto d=(uv-V{observations[i].point[v].x,observations[i].point[v].y,0})*(1/observations[i].sigma[v]);square+=d.dot(d);}
        double norm=std::sqrt(square);value+=norm<=3?.5*square:3*(norm-1.5);
    }
    for(int i=0;i<20;++i){double d=pose.angles[i]-reference.angles[i];value+=.005*d*d;}return value;
}
struct PoseBankResult {PixelFitResult control{},fit{};int chosen=-1,fits=0;std::array<int,2> attempted{-1,-1};std::array<double,2> costs{};};
inline PoseBankResult fit_pose_bank(const PoseBank& bank,const HandRig& rig,const RigPose& seed,
    const std::array<RigCamera,2>& cameras,const std::array<PixelObservation,21>& observations){
    bank.validate(rig);PoseBankResult out;out.control=fit_rig_pixels(rig,seed,cameras,observations);out.fit=out.control;
    if(!out.control.sufficient)return out;
    std::vector<std::pair<double,int>> ranked;
    for(int i=0;i<int(bank.angles.size());++i){auto alternative=seed;alternative.angles=bank.angles[i];double cost=pose_bank_cost(rig,alternative,seed,cameras,observations);if(std::isfinite(cost))ranked.emplace_back(cost,i);}
    std::sort(ranked.begin(),ranked.end());
    for(int k=0;k<std::min(2,int(ranked.size()));++k){int i=ranked[k].second;auto alternative=seed;alternative.angles=bank.angles[i];
        auto fitted=fit_rig_pixels(rig,alternative,cameras,observations,{},0,&seed.angles);out.attempted[k]=i;out.costs[k]=fitted.cost;++out.fits;
        if(fitted.sufficient&&fitted.cost<out.fit.cost){out.fit=fitted;out.chosen=i;}
    }
    return out;
}
}
