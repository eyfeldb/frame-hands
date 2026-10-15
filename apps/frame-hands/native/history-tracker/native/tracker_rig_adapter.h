#pragma once
#include "rig_stage.h"
#include "whole_pose_fitter.h"
#include "capture_head_pose.h"
#include <map>
#include <string>
#include <vector>
#include <set>
namespace fhn {
// Independent pose output alongside the original tracker. Never writes into the
// detector's crop/identity/filter state. Templates use the actual tracker structs.
struct TrackerRigOutput {
    int source=0,side=0;std::int64_t stamp=0;std::string status;
    RigStageResult result{};bool whole_eligible=false,whole_used=false;double whole_alternate_cost=0,whole_baseline_cost=0;
    bool mono=false;   // fitted from a single camera view (FRAME_HANDS_RIG_MONO=1)
};
class TrackerRigAdapter {
    struct Track {int side=-1;RigStage stage;WholePoseFitter whole;};
    bool whole_enabled_=false,allow_recorded_=false;std::uint64_t head_generation_=0;
    HandRig reference_;std::array<HandRig,2> anatomy_;std::array<RigCamera,2> cameras_;
    std::map<int,Track> tracks_;std::vector<TrackerRigOutput> outputs_;std::int64_t last_stamp_=0;
public:
    template<class Camera> static RigCamera convert_camera(const Camera& c){
        RigCamera result;
        for(int r=0;r<3;++r)result.rotation[r]={c.R[0][r],c.R[1][r],c.R[2][r]};
        V origin{c.origin[0]*1000,c.origin[1]*1000,c.origin[2]*1000};
        result.translation={-result.rotation[0].dot(origin),-result.rotation[1].dot(origin),-result.rotation[2].dot(origin)};
        result.fx=c.fx;result.fy=c.fy;result.cx=c.cx;result.cy=c.cy;
        for(int i=0;i<4;++i)result.k[i]=c.k[i];result.validate();return result;
    }
    TrackerRigAdapter(HandRig reference,std::array<HandRig,2> anatomy,std::array<RigCamera,2> cameras,bool whole_enabled=false,bool allow_recorded=false)
        :whole_enabled_(whole_enabled),allow_recorded_(allow_recorded),reference_(std::move(reference)),anatomy_(std::move(anatomy)),cameras_(std::move(cameras)){
        reference_.validate();
        for(int i=0;i<2;++i){const auto& r=anatomy_[i];r.validate();if(r.subject_side>=0&&r.subject_side!=i)throw std::invalid_argument("Wrong profile side");}
        for(auto& c:cameras_)c.validate();
    }
    const std::vector<TrackerRigOutput>& outputs()const{return outputs_;}
    template<class Hand,class Seen> void update(const std::vector<const Hand*>& hands,const std::vector<Seen>& fresh,std::int64_t stamp,const CaptureHeadPose* head=nullptr){
        if(stamp<=last_stamp_)throw std::invalid_argument("Nonmonotonic tracker update");
        last_stamp_=stamp;outputs_.clear();std::set<int> present;
        const bool head_valid=whole_enabled_&&head&&head->usable(stamp,allow_recorded_);
        if(whole_enabled_){if(!head_valid||head_generation_!=head->generation)for(auto& entry:tracks_)entry.second.whole.clear();
            head_generation_=head_valid?head->generation:0;}
        std::array<int,2> side_count{};for(const auto* h:hands)if(h)++side_count[h->right()?1:0];
        for(const auto* h:hands){
            if(!h||h->id<=0||!present.insert(h->id).second)throw std::invalid_argument("Invalid duplicate tracker source");
            const int side=h->right()?1:0;TrackerRigOutput output;output.source=h->id;output.side=side;output.stamp=stamp;
            // Single-view hands are refused unless the opt-in mono fit is enabled: the tracker's
            // single-view 3D (model depth scaled by the learned hand size) then seeds the rig and the
            // fit minimizes reprojection in the one camera that saw the hand.
            const bool mono=h->nviews<2;
            if(mono&&!mono_fit_enabled()){tracks_.erase(h->id);output.status="monocular";outputs_.push_back(output);continue;}
            std::array<const Seen*,2> views{};bool ambiguous=false;int seen=0;
            for(const auto& v:fresh)if(v.hand==h->id){int i=v.cam=="slam_left"?0:(v.cam=="slam_right"?1:-1);
                if(i>=0){if(views[i])ambiguous=true;views[i]=&v;}}
            for(const auto* v:views)seen+=v!=nullptr;
            if(ambiguous||(mono?seen!=1:seen!=2)){tracks_.erase(h->id);output.status="missing_or_ambiguous_views";outputs_.push_back(output);continue;}
            output.mono=mono;
            std::array<RigObservation,21> points{};std::array<PixelObservation,21> pixels{};
            for(int m=0;m<21;++m){int r=HandRig::mediapipe_to_rig[m];if(r<0)continue;
                points[r].present=true;points[r].point={h->pts[m][0]*1000,h->pts[m][1]*1000,h->pts[m][2]*1000};
                for(int v=0;v<2;++v){if(!views[v])continue;pixels[r].present[v]=true;pixels[r].point[v]={views[v]->lm.pts[m][0],views[v]->lm.pts[m][1],0};}}
            auto& track=tracks_[h->id];if(track.side!=side){track.stage=RigStage{};track.whole=WholePoseFitter{};track.side=side;}
            output.result=track.stage.fit(reference_,anatomy_[side],side,points,cameras_,pixels,stamp,h->id);
            if(whole_enabled_&&output.result.initialized){
                auto full=track.whole.refine(anatomy_[side],output.result.initial.pose,output.result.refined.fit,cameras_,points,pixels,stamp,h->id,
                    head_valid&&side_count[side]==1?&head->head:nullptr,true,output.result.options);
                output.whole_baseline_cost=output.result.refined.fit.cost;output.result.refined.fit=full.fit;output.whole_eligible=full.eligible;output.whole_used=full.used;output.whole_alternate_cost=full.alternate_cost;
            }else if(whole_enabled_)track.whole.clear();
            output.status=output.result.initialized?"fitted":"insufficient";outputs_.push_back(std::move(output));
        }
        // A missing source invalidates angle history immediately, even on short gaps.
        for(auto it=tracks_.begin();it!=tracks_.end();)if(!present.count(it->first))it=tracks_.erase(it);else ++it;
    }
};
}
