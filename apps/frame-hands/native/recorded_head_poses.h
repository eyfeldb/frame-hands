#pragma once
#include "capture_head_pose.h"
#include <charconv>
#include <map>
#include <sstream>
namespace fhn {
class RecordedHeadPoses {
    std::map<std::int64_t,CaptureHeadPose> poses_;
    static std::uint64_t stamp(const std::string& s){std::uint64_t n=0;auto p=std::from_chars(s.data(),s.data()+s.size(),n);
        if(s.empty()||p.ec!=std::errc{}||p.ptr!=s.data()+s.size()||!n)throw std::invalid_argument("Invalid recorded timestamp");return n;}
public:
    std::string recording_sha256;
    explicit RecordedHeadPoses(std::istream& input){
        std::string magic,clock,space,extra;int version=0;
        if(!(input>>magic>>version>>clock>>space>>recording_sha256)||magic!="FHHEAD"||version!=1||clock!="CLOCK_MONOTONIC"||space!="raw_world_mm"||!HandRig::hash(recording_sha256))throw std::invalid_argument("Invalid recorded head header");
        std::int64_t last_raw=0;std::uint64_t last_mono=0;std::string line;std::getline(input,line);if(line.find_first_not_of(" \t\r")!=std::string::npos)throw std::invalid_argument("Trailing head header");
        while(std::getline(input,line)){
            if(poses_.size()>=100000)throw std::invalid_argument("Unbounded head recording");
            std::istringstream row(line);std::string raw,mono;int valid=-1;CaptureHeadPose p;
            if(!(row>>raw>>mono>>valid>>p.head.q.w>>p.head.q.x>>p.head.q.y>>p.head.q.z>>p.head.t.x>>p.head.t.y>>p.head.t.z)||row>>extra)throw std::invalid_argument("Malformed recorded head row");
            auto raw_value=stamp(raw);p.capture_ns=stamp(mono);
            if(raw_value>INT64_MAX||std::int64_t(raw_value)<=last_raw||p.capture_ns<=last_mono||valid<0||valid>1||!rig_is_rigid(p.head))throw std::invalid_argument("Invalid recorded head evidence");
            p.raw_ns=std::int64_t(raw_value);p.generation=1;p.recorded=true;p.valid=valid;
            poses_.emplace(p.raw_ns,p);last_raw=p.raw_ns;last_mono=p.capture_ns;
        }
        if(!input.eof()||poses_.empty())throw std::invalid_argument("Empty or failed head recording");
    }
    CaptureHeadPose at(std::int64_t raw)const{auto it=poses_.find(raw);if(it!=poses_.end())return it->second;CaptureHeadPose p;p.raw_ns=raw;p.recorded=true;return p;}
    std::size_t size()const{return poses_.size();}
};
}
