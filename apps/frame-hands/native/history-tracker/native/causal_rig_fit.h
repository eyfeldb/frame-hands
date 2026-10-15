#pragma once
#include "fit_rig_pixels.h"
#include <cstdint>
namespace fhn {
// Causal initialization experiment. No output hold, smoothing, prediction or identity inference.
// Retain only local joint angles: a previous head-space wrist cannot be reused after head motion.
struct CausalRigResult {
    PixelFitResult fit{};
    bool history_eligible=false,used_history=false;
    double fresh_seed_cost=0,history_seed_cost=0;
    int fit_count=1;
};
class CausalRigFitter {
    struct History {bool valid=false;int source_id=-1;std::int64_t stamp=0;std::array<double,20> angles{};};
    std::array<History,2> history_{};
    static double score(const HandRig& rig,const RigPose& pose,const RigPose& fresh,
                        const std::array<RigCamera,2>& cameras,const std::array<PixelObservation,21>& observations,
                        RigFitOptions options){
        const auto points=rig.landmarks(pose.angles,pose.wrist,pose.side);double value=0;
        for(int i=0;i<21;++i){double square=0;
            for(int v=0;v<2;++v)if(observations[i].present[v]){V pixel;if(!cameras[v].project(points[i],pixel))return std::numeric_limits<double>::infinity();
                const auto d=(pixel-V{observations[i].point[v].x,observations[i].point[v].y,0})*(1/observations[i].sigma[v]);square+=d.dot(d);}
            const double n=std::sqrt(square);value+=n<=options.huber?.5*square:options.huber*(n-.5*options.huber);
        }
        for(int i=0;i<20;++i){double d=pose.angles[i]-fresh.angles[i];value+=.5*options.angle_prior*d*d;}
        if(options.translation_prior>0){V d=pose.wrist.t-options.translation_reference;value+=.5*options.translation_prior*d.dot(d);}
        return value;
    }
public:
    void clear(int side){if(side<0||side>1)throw std::invalid_argument("Invalid side");history_[side]={};}
    CausalRigResult fit(const HandRig& rig,const RigPose& fresh,const std::array<RigCamera,2>& cameras,
                        const std::array<PixelObservation,21>& observations,std::int64_t stamp,int source_id,
                        RigFitOptions options={},bool compare_final=false){
        if(fresh.side<0||fresh.side>1||stamp<=0||source_id<0)throw std::invalid_argument("Invalid causal metadata");
        auto& h=history_[fresh.side];
        if(stamp<=h.stamp)throw std::invalid_argument("Nonmonotonic hand timestamp");
        for(const auto& o:observations)o.validate();for(const auto& c:cameras)c.validate();
        CausalRigResult out;RigPose selected=fresh;const int stereo=observed_landmarks(observations);
        out.fresh_seed_cost=score(rig,fresh,fresh,cameras,observations,options);out.history_seed_cost=out.fresh_seed_cost;
        out.history_eligible=h.valid&&h.source_id==source_id&&stamp-h.stamp<=100000000&&stereo>=6;
        if(out.history_eligible){auto warm=fresh;warm.angles=h.angles;
            out.history_seed_cost=score(rig,warm,fresh,cameras,observations,options);
            if(std::isfinite(out.history_seed_cost)&&out.history_seed_cost<out.fresh_seed_cost){selected=warm;out.used_history=true;}
        }
        // A failed projection is a rejected observation, not a fatal service error.
        if(!std::isfinite(out.used_history?out.history_seed_cost:out.fresh_seed_cost)){
            out.fit.pose=fresh;h={false,source_id,stamp,{}};return out;
        }
        // Same prior in both branches: isolate initialization from temporal regularization.
        // FRAME_HANDS_RIG_ANGLE_SIGMA_DEG: temporal continuity instead. The previous fitted angles of this track
        // are the prior reference with weight 1/sigma^2, and the fresh-seed control only wins when clearly
        // cheaper (20% margin), so near-equal minima do not alternate frame to frame.
        const double sigma=angle_sigma_rad();
        const bool temporal=sigma>0&&h.valid&&h.source_id==source_id&&stamp-h.stamp<=100000000;
        RigFitOptions fit_options=options;const std::array<double,20>* reference=&fresh.angles;
        std::array<double,20> previous{};
        if(temporal){previous=h.angles;fit_options.angle_prior=1/(sigma*sigma);reference=&previous;}
        out.fit=fit_rig_pixels(rig,selected,cameras,observations,fit_options,0,reference);
        if(compare_final&&out.used_history&&std::isfinite(out.fresh_seed_cost)){
            auto control=temporal?fit_rig_pixels(rig,fresh,cameras,observations,fit_options,0,reference):fit_rig_pixels(rig,fresh,cameras,observations,options);out.fit_count=2;
            if(control.cost<=out.fit.cost*(temporal?.8:1.)){out.fit=control;out.used_history=false;}
        }
        h={out.fit.sufficient,source_id,stamp,out.fit.pose.angles};return out;
    }
};
}
