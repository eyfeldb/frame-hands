#pragma once
#include "fit_rig_pixels.h"
#include "whole_pose_seed.h"
#include <cstdint>
namespace fhn {
struct WholePoseResult {PixelFitResult fit{};bool eligible=false,used=false;double alternate_cost=0;};
// Optional initializer, not a presence tracker or renderer. Fresh baseline remains
// a competitor. Capture-time head transforms must share the same world reference.
class WholePoseFitter {
    struct History {bool valid=false;int source=-1;std::int64_t stamp=0;RigPose world{};};
    std::array<History,2> history_{};
    std::array<std::int64_t,2> last_stamp_{};
public:
    // Call on missing/ambiguous side, source disappearance or reference-space reset.
    // Timestamp guards deliberately survive clearing history.
    void clear(int side){if(side<0||side>1)throw std::invalid_argument("Invalid history side");history_[side]={};}
    void clear(){for(int side=0;side<2;++side)clear(side);}
    WholePoseResult refine(const HandRig& rig,const RigPose& fresh,const PixelFitResult& baseline,
        const std::array<RigCamera,2>& cameras,const std::array<RigObservation,21>& points,
        const std::array<PixelObservation,21>& pixels,std::int64_t stamp,int source,
        const Rigid* capture_head,bool continuous=true,RigFitOptions options={}){
        int side=fresh.side;
        if(side<0||side>1||baseline.pose.side!=side||source<0||stamp<=0||stamp<=last_stamp_[side])throw std::invalid_argument("Invalid whole-pose metadata");
        if(capture_head&&!rig_is_rigid(*capture_head))throw std::invalid_argument("Nonrigid capture head");
        for(const auto& p:points)p.validate();for(const auto& p:pixels)p.validate();
        WholePoseResult out;out.fit=baseline;out.alternate_cost=baseline.cost;
        const int stereo=observed_landmarks(pixels);   // stereo pairs, or any-view landmarks in mono mode
        bool palm=true;for(int i:{5,8,11,14,17})palm=palm&&points[i].present;
        bool fresh_valid=baseline.sufficient&&stereo>=6&&palm;
        if(fresh_valid){
            if(!std::isfinite(baseline.cost)||!std::isfinite(baseline.observed_rmse_px))throw std::invalid_argument("Nonfinite baseline result");
            rig.landmarks(baseline.pose.angles,baseline.pose.wrist,side);
        }
        auto previous=history_[side];
        out.eligible=capture_head&&continuous&&fresh_valid&&previous.valid&&previous.source==source&&stamp-previous.stamp<=100000000;
        if(out.eligible){RigPose seed;
            out.eligible=whole_pose_seed(rig,previous.world,*capture_head,points,seed);
            // History can put an observed fingertip behind a camera even when
            // the current baseline is valid. Reject only this optional seed.
            if(out.eligible){auto projected=rig.landmarks(seed.angles,seed.wrist,side);
                for(int i=0;i<21;++i)for(int view=0;view<2;++view)if(pixels[i].present[view]){
                    V uv;if(!cameras[view].project(projected[i],uv))out.eligible=false;}}
            if(out.eligible){auto alternate=fit_rig_pixels(rig,seed,cameras,pixels,options,0,&fresh.angles);   // same cost terms as the baseline
                out.alternate_cost=alternate.cost;
                if(alternate.sufficient&&alternate.cost<baseline.cost){out.fit=alternate;out.used=true;}}
        }
        // Mutate only after validating/fitting. No failed/stale update refreshes history.
        last_stamp_[side]=stamp;if(history_[1-side].source==source)history_[1-side]={};history_[side]={};
        if(capture_head&&fresh_valid){auto& next=history_[side];next.valid=true;next.source=source;next.stamp=stamp;
            next.world=out.fit.pose;next.world.wrist=capture_head->then(out.fit.pose.wrist);next.world.wrist.q=next.world.wrist.q.unit();}
        return out;
    }
};
}
