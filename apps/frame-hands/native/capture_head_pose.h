#pragma once
#include "whole_pose_seed.h"
#include <cstdint>
namespace fhn {
// Requested-time runtime estimate, not a guarantee of historical sensor sampling.
// head is raw-tracking-world-from-capture-head in millimeters.
struct CaptureHeadPose {
    std::int64_t raw_ns=0;std::uint64_t capture_ns=0,query_ns=0,returned_ns=0,generation=0;
    bool valid=false,recorded=false;Rigid head{};
    bool usable(std::int64_t raw,bool allow_recorded=false)const{
        if(!valid||raw<=0||raw_ns!=raw||!capture_ns||!generation||!rig_is_rigid(head))return false;
        if(recorded)return allow_recorded&&query_ns==0&&returned_ns==0;
        return capture_ns<=query_ns&&query_ns<=returned_ns&&returned_ns-capture_ns<=250000000;
    }
};
inline bool head_request_age(std::uint64_t capture,std::uint64_t now,float& seconds){
    if(!capture||capture>now||now-capture>250000000)return false;seconds=float(double(now-capture)*1e-9);return true;
}
inline bool head_from_matrix(const double* m,Rigid& head){
    for(int i=0;i<12;++i)if(!std::isfinite(m[i]))return false;
    V x{m[0],m[4],m[8]},y{m[1],m[5],m[9]},z{m[2],m[6],m[10]};
    if(std::abs(x.len()-1)>1e-4||std::abs(y.len()-1)>1e-4||std::abs(z.len()-1)>1e-4||
       std::abs(x.dot(y))>1e-4||std::abs(x.dot(z))>1e-4||std::abs(y.dot(z))>1e-4||x.dot(y.cross(z))<.9999)return false;
    head={fhs::basis(x,y,z),{m[3]*1000,m[7]*1000,m[11]*1000}};return rig_is_rigid(head);
}
}
