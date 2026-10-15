#pragma once
#include "rig_snapshot.h"
#include "native_head_pose.h"
#include "recorded_head_poses.h"
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
namespace fhn {
// Opt-in, isolated experiment. Invalid requested configuration fails at startup.
class TrackerRigHook {
    std::unique_ptr<TrackerRigAdapter> adapter_;
    std::ofstream trace_;
    std::unique_ptr<RecordedHeadPoses> recorded_heads_;
    std::unique_ptr<NativeHeadPose> head_source_;std::int64_t capture_raw_=0;std::uint64_t capture_mono_=0;
    std::string snapshot_;std::uint64_t sequence_=0;std::int64_t raw_stamp_=0;
public:
    template<class Camera> void configure(const std::map<std::string,Camera>& cameras){
        const char* root=std::getenv("FRAME_HANDS_RIG_DIR");if(!root||!*root)return;
        const char* path=std::getenv("FRAME_HANDS_RIG_TRACE");
        if(!path||!*path)throw std::invalid_argument("Rig experiment requires a trace path");
        if(const char* snapshot=std::getenv("FRAME_HANDS_RIG_SNAPSHOT")){snapshot_=snapshot;if(!snapshot_.empty()&&std::ifstream(snapshot_).good())throw std::invalid_argument("Snapshot path already exists");}
        std::string dir=root;std::ifstream generic(dir+"/generic.fhrig"),left(dir+"/side0.fhrig"),right(dir+"/side1.fhrig");
        if(!generic||!left||!right||(std::string(path)!="/dev/null"&&std::ifstream(path).good()))throw std::invalid_argument("Invalid rig profile or existing trace");
        if(!cameras.count("slam_left")||!cameras.count("slam_right"))throw std::invalid_argument("Missing calibrated forward cameras");
        auto reference=HandRig::load(generic);std::array<HandRig,2> anatomy{HandRig::load(left),HandRig::load(right)};
        std::array<RigCamera,2> converted{TrackerRigAdapter::convert_camera(cameras.at("slam_left")),TrackerRigAdapter::convert_camera(cameras.at("slam_right"))};
        bool whole=false;
        if(const char* value=std::getenv("FRAME_HANDS_RIG_WHOLE_POSE")){if(std::string(value)!="1")throw std::invalid_argument("Invalid whole-pose option");whole=true;}
        if(const char* value=std::getenv("FRAME_HANDS_RIG_MONO")){const std::string s(value);if(s!="0"&&s!="1")throw std::invalid_argument("Invalid mono-fit option");mono_fit_enabled()=s=="1";}
        if(const char* value=std::getenv("FRAME_HANDS_RIG_MONO_SIGMA_MM")){const double s=std::atof(value);if(!(s>=1&&s<=1000))throw std::invalid_argument("Invalid mono depth prior sigma");mono_translation_sigma_mm()=s;}
        if(const char* value=std::getenv("FRAME_HANDS_RIG_STEREO_SIGMA_MM")){const double s=std::atof(value);if(!(s==0||(s>=1&&s<=1000)))throw std::invalid_argument("Invalid stereo wrist prior sigma");stereo_translation_sigma_mm()=s;}
        if(const char* value=std::getenv("FRAME_HANDS_RIG_ANCHOR_LEAK")){const double s=std::atof(value);if(!(s>=0&&s<=1))throw std::invalid_argument("Invalid anchor leak");mono_anchor_leak()=s;}
        if(const char* value=std::getenv("FRAME_HANDS_RIG_ANCHOR_CAP_MM")){const double s=std::atof(value);if(!(s>=1&&s<=1000))throw std::invalid_argument("Invalid anchor cap");mono_anchor_cap_mm()=s;}
        if(const char* value=std::getenv("FRAME_HANDS_RIG_ANGLE_SIGMA_DEG")){const double s=std::atof(value);if(!(s==0||(s>=0.5&&s<=90)))throw std::invalid_argument("Invalid angle prior sigma");angle_sigma_rad()=s*3.14159265358979323846/180;}
        const char* saved=std::getenv("FRAME_HANDS_RIG_REPLAY_HEADS");
        if(saved){if(!whole||!*saved||!snapshot_.empty()||std::getenv("FRAME_HANDS_RIG_OPENVR"))throw std::invalid_argument("Replay head mode conflicts with live context");
            std::ifstream file(saved);if(!file)throw std::invalid_argument("Head recording unavailable");recorded_heads_=std::make_unique<RecordedHeadPoses>(file);}
        else if(whole){const char* library=std::getenv("FRAME_HANDS_RIG_OPENVR");head_source_=std::make_unique<NativeHeadPose>(library);}
        adapter_=std::make_unique<TrackerRigAdapter>(reference,anatomy,converted,whole,bool(recorded_heads_));
        trace_.open(path);if(!trace_)throw std::runtime_error("Cannot create rig trace");trace_.precision(17);
    }
    // Only the live capture loop knows the raw->monotonic conversion. Replay does
    // not call this and cannot pretend a saved raw timestamp is a fresh capture.
    void publish(std::int64_t raw,std::uint64_t capture_mono,std::uint64_t now_mono){
        if(recorded_heads_)throw std::invalid_argument("Recorded heads cannot publish live snapshots");
        if(!adapter_||snapshot_.empty())return;
        if(raw!=raw_stamp_)throw std::invalid_argument("Snapshot must match current tracker frame");
        const auto payload=rig_snapshot(sequence_+1,raw,capture_mono,now_mono,outputs());
        const auto temporary=snapshot_+".tmp";
        {std::ofstream file(temporary,std::ios::binary|std::ios::trunc);file<<payload;file.close();if(!file)throw std::runtime_error("Snapshot write failed");}
        if(std::rename(temporary.c_str(),snapshot_.c_str()))throw std::runtime_error("Snapshot atomic replacement failed");
        ++sequence_;
    }
    void capture_context(std::int64_t raw,std::uint64_t mono){if(recorded_heads_)throw std::invalid_argument("Recorded heads cannot enter live capture");capture_raw_=raw;capture_mono_=mono;}
    bool enabled()const{return bool(adapter_);}
    const std::vector<TrackerRigOutput>& outputs()const{static const std::vector<TrackerRigOutput> empty;return adapter_?adapter_->outputs():empty;}
    template<class Hand,class Seen> void update(const std::vector<const Hand*>& hands,const std::vector<Seen>& views,std::int64_t stamp){
        if(!adapter_)return;CaptureHeadPose head;
        if(recorded_heads_)head=recorded_heads_->at(stamp);
        if(head_source_&&capture_raw_==stamp)head=head_source_->query(stamp,capture_mono_);
        adapter_->update(hands,views,stamp,(head_source_||recorded_heads_)?&head:nullptr);raw_stamp_=stamp;
        if(head_source_||recorded_heads_)trace_<<(recorded_heads_?"head_recorded ":"head ")<<stamp<<' '<<head.capture_ns<<' '<<head.query_ns<<' '<<head.returned_ns<<' '<<head.generation<<' '<<head.valid<<' '<<head.head.q.w<<' '<<head.head.q.x<<' '<<head.head.q.y<<' '<<head.head.q.z<<' '<<head.head.t.x<<' '<<head.head.t.y<<' '<<head.head.t.z<<'\n';
        // Every frame has a marker, including empty frames, to preserve loss evidence.
        trace_<<"frame "<<stamp<<' '<<outputs().size()<<'\n';
        for(const auto& o:outputs()){
            trace_<<"rig "<<o.source<<' '<<o.side<<' '<<o.status;
            if(o.result.initialized){const auto& f=o.result.refined.fit;const auto& p=f.pose;
                trace_<<' '<<f.cost<<' '<<f.observed_rmse_px<<' '<<p.wrist.q.w<<' '<<p.wrist.q.x<<' '<<p.wrist.q.y<<' '<<p.wrist.q.z<<' '<<p.wrist.t.x<<' '<<p.wrist.t.y<<' '<<p.wrist.t.z;
                for(double angle:p.angles)trace_<<' '<<angle;}
            trace_<<'\n';
            if(o.mono)trace_<<"mono "<<o.source<<' '<<o.result.initialized<<'\n';
            if(head_source_||recorded_heads_)trace_<<"whole "<<o.source<<' '<<o.whole_eligible<<' '<<o.whole_used<<' '<<o.whole_baseline_cost<<' '<<o.whole_alternate_cost<<'\n';
        }
        if(!trace_)throw std::runtime_error("Rig trace write failed");
    }
};
}
