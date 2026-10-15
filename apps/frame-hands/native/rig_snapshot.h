#pragma once
#include "tracker_rig_adapter.h"
#include <sstream>
#include <iomanip>
namespace fhn {
// Camera/raw time is an identity key only. Consumers must request head pose using
// capture_mono_ns, never stamp from TrackerRigOutput.
inline std::string rig_snapshot(std::uint64_t sequence,std::int64_t raw_ns,std::uint64_t capture_mono_ns,
    std::uint64_t published_mono_ns,const std::vector<TrackerRigOutput>& outputs){
    if(!sequence||raw_ns<=0||!capture_mono_ns||capture_mono_ns>published_mono_ns)throw std::invalid_argument("Invalid snapshot clocks");
    std::ostringstream out;out<<std::setprecision(17);
    out<<"{\"version\":3,\"clock\":\"CLOCK_MONOTONIC\",\"space\":\"capture_head_mm\",\"sequence\":"<<sequence
       <<",\"capture_raw_ns\":"<<raw_ns<<",\"capture_mono_ns\":"<<capture_mono_ns<<",\"published_mono_ns\":"<<published_mono_ns<<",\"hands\":[";
    bool first=true;
    for(const auto& h:outputs){
        if(h.stamp!=raw_ns||h.source<=0||h.side<0||h.side>1)throw std::invalid_argument("Snapshot/source mismatch");
        if(h.status!="fitted"&&h.status!="monocular"&&h.status!="missing_or_ambiguous_views"&&h.status!="insufficient")throw std::invalid_argument("Unknown rig status");
        if(h.result.initialized!=(h.status=="fitted"))throw std::invalid_argument("Inconsistent rig availability");
        if(!first)out<<',';first=false;
        out<<"{\"source_id\":"<<h.source<<",\"side\":"<<h.side<<",\"status\":\""<<h.status<<"\"";
        if(h.result.initialized){const auto& p=h.result.refined.fit.pose;const auto& w=p.wrist;
            if(p.side!=h.side||!w.t.finite())throw std::invalid_argument("Invalid fitted snapshot");
            const double n=w.q.w*w.q.w+w.q.x*w.q.x+w.q.y*w.q.y+w.q.z*w.q.z;
            if(!std::isfinite(n)||std::abs(n-1)>1e-5)throw std::invalid_argument("Invalid snapshot rotation");
            out<<",\"wrist_head_mm\":["<<w.q.w<<','<<w.q.x<<','<<w.q.y<<','<<w.q.z<<','<<w.t.x<<','<<w.t.y<<','<<w.t.z<<"],\"joint_angles\":[";
            for(int i=0;i<20;++i){if(!std::isfinite(p.angles[i]))throw std::invalid_argument("Invalid snapshot angle");if(i)out<<',';out<<p.angles[i];}out<<']';
        }
        out<<'}';
    }
    out<<"]}\n";return out.str();
}
}
