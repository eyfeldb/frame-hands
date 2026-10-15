#pragma once
#include "../pc-input/math.h"
#include <array>
#include <cctype>
#include <istream>
#include <stdexcept>
#include <string>
namespace fhn {
using fhs::V;using fhs::Q;
struct Rigid {
    Q q{};V t{};
    V apply(V p)const{return q.rotate(p)+t;}
    Rigid then(const Rigid& child)const{return{q*child.q,apply(child.t)};}
};
struct RigJoint {V axis{},pivot{};double lower=0,upper=0;};
struct HandRig {
    std::array<RigJoint,20> joints{};
    std::array<V,21> rest{};
    std::array<std::array<double,17>,21> weights{};
    std::string model_sha,calibration_sha;bool calibrated=false,experimental=false;int subject_side=-1;
    // Missing thumb CMC is deliberately not manufactured from a different landmark.
    static constexpr std::array<int,21> mediapipe_to_rig={5,-1,6,7,0,8,9,10,1,11,12,13,2,14,15,16,3,17,18,19,4};
    static bool hash(const std::string& s){if(s.size()!=64)return false;for(unsigned char c:s)if(!std::isxdigit(c))return false;return true;}
    void validate()const{
        if(!hash(model_sha)||subject_side< -1||subject_side>1)throw std::invalid_argument("Invalid rig provenance or side");
        if((calibrated||experimental)?(!hash(calibration_sha)||subject_side<0):(calibration_sha!="-"||subject_side!=-1))throw std::invalid_argument("Inconsistent calibration status");
        for(const auto& j:joints){
            if(!j.axis.finite()||!j.pivot.finite()||std::abs(j.axis.len()-1)>1e-5||j.pivot.len()>1000||!std::isfinite(j.lower)||!std::isfinite(j.upper)||j.lower>j.upper||std::abs(j.lower)>6.284||std::abs(j.upper)>6.284)throw std::invalid_argument("Invalid joint geometry or limits");
        }
        for(int i=0;i<21;++i){if(!rest[i].finite()||rest[i].len()>1000)throw std::invalid_argument("Invalid landmark");
            double sum=0;for(double w:weights[i]){if(!std::isfinite(w)||w<0||w>1)throw std::invalid_argument("Invalid skinning weight");sum+=w;}
            if(std::abs(sum-1)>1e-5)throw std::invalid_argument("Skinning weights must sum to one");
        }
    }
    static HandRig load(std::istream& input){
        std::string magic,units,side,kind,extra;int version=0,nj=0,nl=0,nf=0;HandRig r;
        if(!(input>>magic>>version>>units>>side>>kind>>r.subject_side>>r.model_sha>>r.calibration_sha>>nj>>nl>>nf)||magic!="FHRIG"||version!=1||units!="mm"||side!="left"||(kind!="generic"&&kind!="calibrated"&&kind!="experimental")||nj!=20||nl!=21||nf!=17)throw std::invalid_argument("Unsupported rig header");
        r.calibrated=kind=="calibrated";r.experimental=kind=="experimental";
        for(auto& j:r.joints)if(!(input>>j.axis.x>>j.axis.y>>j.axis.z>>j.pivot.x>>j.pivot.y>>j.pivot.z>>j.lower>>j.upper))throw std::invalid_argument("Truncated joint profile");
        for(int i=0;i<21;++i){if(!(input>>r.rest[i].x>>r.rest[i].y>>r.rest[i].z))throw std::invalid_argument("Truncated landmark profile");for(double& w:r.weights[i])if(!(input>>w))throw std::invalid_argument("Truncated weights");}
        if(input>>extra)throw std::invalid_argument("Unexpected trailing rig data");r.validate();return r;
    }
    std::array<V,21> landmarks(const std::array<double,20>& angles,const Rigid& wrist,int side)const{
        if(side<0||side>1||((calibrated||experimental)&&side!=subject_side))throw std::invalid_argument("Wrong anatomical side");
        double qnorm=std::sqrt(wrist.q.w*wrist.q.w+wrist.q.x*wrist.q.x+wrist.q.y*wrist.q.y+wrist.q.z*wrist.q.z);
        if(!std::isfinite(qnorm)||std::abs(qnorm-1)>1e-6||!wrist.t.finite())throw std::invalid_argument("Nonrigid wrist");
        std::array<Rigid,17> frames{};
        for(int finger=0;finger<5;++finger){Rigid cumulative;
            for(int k=0;k<4;++k){int i=4*finger+k;const auto& j=joints[i];double angle=angles[i];
                if(!std::isfinite(angle)||angle<j.lower-1e-8||angle>j.upper+1e-8)throw std::invalid_argument("Pose outside joint limits");
                double norm=j.axis.len(),half=angle*norm*.5,s=std::sin(half)/norm;
                Q q{std::cos(half),j.axis.x*s,j.axis.y*s,j.axis.z*s};Rigid local{q,j.pivot-q.rotate(j.pivot)};
                cumulative=cumulative.then(local);if(k>=1)frames[2+3*finger+k-1]=cumulative;
            }
        }
        std::array<V,21> result{};
        for(int i=0;i<21;++i){V point{};double mass=0;for(int b=0;b<17;++b)if(weights[i][b]){point=point+frames[b].apply(rest[i])*weights[i][b];mass+=weights[i][b];}
            if(side)point.x=-point.x;result[i]=wrist.q.rotate(point)+wrist.t*mass;if(!result[i].finite())throw std::runtime_error("Nonfinite rig output");}
        return result;
    }
};
}
