// Recording-only C ABI wrapper around pinned, unmodified Mercury solver sources.
#include "kine_lm/lm_interface.hpp"
#include "kine_lm/lm_defines.hpp"
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <new>
using namespace xrt::tracking::hand::mercury;
namespace lm = xrt::tracking::hand::mercury::lm;
extern "C" void u_log(const char *, int, const char *, u_logging_level level, const char *format, ...) {
    if (level < U_LOGGING_WARN) return;
    va_list args; va_start(args,format); std::vfprintf(stderr,format,args); va_end(args); std::fputc('\n',stderr);
}
struct Solver { xrt_pose calibration; bool right; lm::KinematicHandLM *state=nullptr;
    Solver(xrt_pose p,bool r):calibration(p),right(r){reset();}
    void reset(){if(state){lm::optimizer_destroy(&state);state=nullptr;}lm::optimizer_create(calibration,right,U_LOGGING_WARN,&state);}
    ~Solver(){if(state)lm::optimizer_destroy(&state);}
};
#define API extern "C" __declspec(dllexport)
API void *fh_solver_create(const float *pose, int side) {
    if(!pose || (side!=0 && side!=1))return nullptr;
    for(int i=0;i<7;i++)if(!std::isfinite(pose[i]))return nullptr;
    float n=0;for(int i=0;i<4;i++)n+=pose[i]*pose[i];if(std::fabs(n-1)>0.001f)return nullptr;
    xrt_pose p{};p.orientation={pose[0],pose[1],pose[2],pose[3]};p.position={pose[4],pose[5],pose[6]};
    try{return new Solver(p,side!=0);}catch(...){return nullptr;}
}
API void fh_solver_destroy(void *handle){delete static_cast<Solver *>(handle);}
// Input:2views x( active,quaternionXYZW,radius,21x[xy,relativeDepth,confidenceXY,confidenceDepth],5x[curl,variance])=242floats.
// Output:26x[positionXYZ,orientationXYZW], handSize,reprojectionError,active =185floats.
API int fh_solver_run(void *handle,const float *data,int reset,float smoothing,int optimize_size,float target_size,float size_error,float depth,float *output) {
    if(!handle||!data||!output)return -1;
    for(int i=0;i<242;i++)if(!std::isfinite(data[i]))return -1;
    if(!std::isfinite(smoothing)||!std::isfinite(target_size)||!std::isfinite(size_error)||!std::isfinite(depth)||target_size<.06f||target_size>.12f||smoothing<0||size_error<0||depth<0)return -1;
    one_frame_input input{};int index=0;int active=0;
    for(auto &view:input.views){view.active=data[index++]!=0;active+=view.active;view.look_dir={data[index],data[index+1],data[index+2],data[index+3]};index+=4;view.stereographic_radius=data[index++];
        float norm=view.look_dir.x*view.look_dir.x+view.look_dir.y*view.look_dir.y+view.look_dir.z*view.look_dir.z+view.look_dir.w*view.look_dir.w;
        if(view.active && (view.stereographic_radius<=0 || std::fabs(norm-1)>0.001f))return -1;
        for(auto &p:view.keypoints_in_scaled_stereographic){p.pos_2d={data[index],data[index+1]};p.depth_relative_to_midpxm=data[index+2];p.confidence_xy=data[index+3];p.confidence_depth=data[index+4];index+=5;if(p.confidence_xy<0 || p.confidence_depth<0)return -1;}
        for(auto &c:view.curls){c.value=data[index++];c.variance=data[index++];if(c.variance<=0)return -1;}
    }
    if(!active)return -1;
    try{auto *solver=static_cast<Solver *>(handle);if(reset)solver->reset();xrt_hand_joint_set hand{};float size=0,error=0;
        lm::optimizer_run(solver->state,input,reset!=0,smoothing,optimize_size!=0,target_size,size_error,depth,hand,size,error);
        int n=0;for(const auto &joint:hand.values.hand_joint_set_default){const auto &p=joint.relation.pose;output[n++]=p.position.x;output[n++]=p.position.y;output[n++]=p.position.z;output[n++]=p.orientation.x;output[n++]=p.orientation.y;output[n++]=p.orientation.z;output[n++]=p.orientation.w;}
        output[n++]=size;output[n++]=error;output[n++]=hand.is_active?1.f:0.f;
        if(n!=185)return -4;for(int i=0;i<n;i++)if(!std::isfinite(output[i]))return -3;return 0;
    }catch(...){return -2;}
}

// Optional experimental re-expression of existing state in the current left-camera
// coordinates. Pinned native optimizer sources and their fit equations are unchanged.
API int fh_solver_transport(void *handle,const float *pose) {
    if(!handle || !pose)return -1;
    for(int i=0;i<7;i++)if(!std::isfinite(pose[i]))return -1;
    float norm=0;for(int i=0;i<4;i++)norm+=pose[i]*pose[i];
    if(std::fabs(norm-1)>0.001f)return -1;
    auto *solver=static_cast<Solver *>(handle);auto &state=*solver->state;
    if(state.first_frame)return -1;
    if(pose[0]==0 && pose[1]==0 && pose[2]==0 && pose[3]==1 && pose[4]==0 && pose[5]==0 && pose[6]==0)return 0;
    // optimizer_run packs zero post-transforms before returning. Only transport
    // that completed state; do not rotate an in-progress optimization variable.
    if(state.last_frame.wrist_post_location.norm()>1e-7f || state.last_frame.wrist_post_orientation_aax.norm()>1e-7f)return -1;
    Eigen::Quaternionf rotation(pose[3],pose[0],pose[1],pose[2]);
    Eigen::Vector3f translation(pose[4],pose[5],pose[6]);
    auto position=[&](auto &p){Eigen::Vector3f v=rotation*Eigen::Vector3f(p.x,p.y,p.z)+translation;p.x=v.x();p.y=v.y();p.z=v.z();};
    position(state.this_frame_pre_position);position(state.last_frame.wrist_final_location);
    // Native right-hand bone offsets are reflected in camera X after rotation.
    // Therefore their internal rotation transforms by F*R*F, F=diag(-1,1,1).
    Eigen::Quaternionf internal=rotation;
    if(solver->right){internal.y()*=-1;internal.z()*=-1;}
    auto orientation=[&](auto &q){Eigen::Quaternionf r=internal*Eigen::Quaternionf(q.w,q.x,q.y,q.z);r.normalize();q.x=r.x();q.y=r.y();q.z=r.z();q.w=r.w();};
    orientation(state.this_frame_pre_rotation);orientation(state.last_frame.wrist_final_orientation);
    return 0;
}

// Diagnostic snapshot for transport invariants, without running another fit.
API int fh_solver_history(void *handle,float *output) {
    if(!handle || !output)return -1;
    const auto &s=*static_cast<Solver *>(handle)->state;if(s.first_frame)return -1;
    int n=0;
    auto pose=[&](const auto &p,const auto &q){output[n++]=p.x;output[n++]=p.y;output[n++]=p.z;output[n++]=q.x;output[n++]=q.y;output[n++]=q.z;output[n++]=q.w;};
    pose(s.this_frame_pre_position,s.this_frame_pre_rotation);pose(s.last_frame.wrist_final_location,s.last_frame.wrist_final_orientation);
    return 0;
}

// Read-only audit of the existing curl-dependent stability multiplier. These
// declarations refer to the pinned native implementations, not replacements.
namespace xrt::tracking::hand::mercury::lm {
float get_avg_curl_value(const one_frame_input &, const int);
HandScalar calc_stability_curl_multiplier(const OptimizerFinger<HandScalar> &, HandScalar);
}
API int fh_solver_fingers(void *handle,float *output) {
    if(!handle || !output)return -1;
    const auto &s=*static_cast<Solver *>(handle)->state;if(s.first_frame)return -1;
    int n=0;for(const auto &f:s.last_frame.finger){output[n++]=f.proximal_swing.x;output[n++]=f.rots[0];output[n++]=f.rots[1];}
    return n==12?0:-2;
}
// Per index/middle/ring/little finger: prior angles3, sum, learned average,
// native stability multiplier, left curl/variance, right curl/variance.
// Caller must use this BEFORE a non-reset optimizer step. No state is changed.
API int fh_solver_curl_diagnostics(void *handle,const float *packet,float *output) {
    if(!handle || !packet || !output)return -1;
    const auto &s=*static_cast<Solver *>(handle)->state;if(s.first_frame)return -1;
    one_frame_input input{};int active=0;
    for(int v=0;v<2;v++){
        const float *data=packet+121*v;
        if(!std::isfinite(data[0]))return -1;
        input.views[v].active=data[0]!=0;active+=input.views[v].active;
        for(int i=0;i<5;i++){
            float value=data[111+2*i],variance=data[112+2*i];
            if(!std::isfinite(value)||!std::isfinite(variance)||variance<=0)return -1;
            input.views[v].curls[i]={value,variance};
        }
    }
    if(!active)return -1;
    int n=0;for(int i=0;i<4;i++){
        const auto &f=s.last_frame.finger[i];float average=lm::get_avg_curl_value(input,i+1);
        output[n++]=f.proximal_swing.x;output[n++]=f.rots[0];output[n++]=f.rots[1];
        output[n++]=f.proximal_swing.x+f.rots[0]+f.rots[1];output[n++]=average;
        output[n++]=lm::calc_stability_curl_multiplier(f,average);
        for(int v=0;v<2;v++){output[n++]=input.views[v].curls[i+1].value;output[n++]=input.views[v].curls[i+1].variance;}
    }
    return n==40?0:-2;
}

#ifdef FH_SYNTHETIC_DIAGNOSTICS
extern "C" int fh_seed_synthetic_state(void *,const float *,const float *,float);
API int fh_solver_seed_synthetic(void *handle,const float *wrist,const float *angles,float size) {
    if(!handle)return -1;
    return fh_seed_synthetic_state(static_cast<Solver *>(handle)->state,wrist,angles,size);
}
#endif

#ifdef FH_STEREO_INITIALIZER
extern "C" int fh_seed_stereo_state(void *,const float *,float,float *);
API int fh_solver_seed_stereo(void *handle,const float *packet,float size,float *diagnostic) {
    if(!handle)return -1;
    return fh_seed_stereo_state(static_cast<Solver *>(handle)->state,packet,size,diagnostic);
}
#endif

#ifdef FH_CURL_INITIALIZER
extern "C" int fh_seed_observation_state(void *,const float *,float,int,int,float *);
API int fh_solver_seed_observations(void *handle,const float *packet,float size,int stereo,int curls,float *diagnostic) {
    if(!handle)return -1;
    return fh_seed_observation_state(static_cast<Solver *>(handle)->state,packet,size,stereo,curls,diagnostic);
}
#endif
