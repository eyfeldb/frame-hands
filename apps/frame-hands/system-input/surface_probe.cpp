// Read-only comparison of raw/standing gaze and native overlay intersection.
#include <openvr.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <thread>
vr::HmdVector3_t point(const vr::HmdMatrix34_t& m,vr::HmdVector3_t v){vr::HmdVector3_t r{};for(int i=0;i<3;++i)r.v[i]=m.m[i][3]+m.m[i][0]*v.v[0]+m.m[i][1]*v.v[1]+m.m[i][2]*v.v[2];return r;}
vr::HmdVector3_t inverse_point(const vr::HmdMatrix34_t& m,vr::HmdVector3_t v){vr::HmdVector3_t r{};for(int i=0;i<3;++i)for(int j=0;j<3;++j)r.v[i]+=m.m[j][i]*(v.v[j]-m.m[j][3]);return r;}
double distance(vr::HmdVector3_t a,vr::HmdVector3_t b){double s=0;for(int i=0;i<3;++i)s+=(a.v[i]-b.v[i])*(a.v[i]-b.v[i]);return std::sqrt(s);}
bool hit(vr::IVROverlay* overlay,vr::VROverlayHandle_t handle,vr::ETrackingUniverseOrigin origin,vr::HmdVector3_t source,vr::HmdVector3_t target){
    vr::VROverlayIntersectionParams_t p{};p.eOrigin=origin;p.vSource=source;double n=distance(source,target);if(n<.0001)return false;
    for(int k=0;k<3;++k)p.vDirection.v[k]=(target.v[k]-source.v[k])/n;
    vr::VROverlayIntersectionResults_t r{};return overlay->ComputeOverlayIntersection(handle,&p,&r);
}
int main(int argc,char** argv){
    if(argc!=2)return 2;
    vr::EVRInitError err;auto sys=vr::VR_Init(&err,vr::VRApplication_Overlay);if(err||!sys)return 1;
    auto input=vr::VRInput();auto overlay=vr::VROverlay();vr::VRActionSetHandle_t set=0;vr::VRActionHandle_t action=0;
    if(input->SetActionManifestPath(argv[1])||input->GetActionSetHandle("/actions/main",&set)||input->GetActionHandle("/actions/main/in/gaze",&action)){vr::VR_Shutdown();return 2;}
    vr::VRActiveActionSet_t active{};active.ulActionSet=set;
    // A known transparent, noninteractive surface separates API origin support
    // from eye accuracy. Off-head/idle results remain inconclusive.
    vr::VROverlayHandle_t control=0;
    if(!overlay->CreateOverlay("framehands.surface.control","Frame Hands hidden geometry control",&control)){
        uint32_t pixel=0xffffffff;vr::HmdMatrix34_t m{};m.m[0][0]=m.m[1][1]=m.m[2][2]=1;m.m[1][3]=1;m.m[2][3]=-1;
        auto a=overlay->SetOverlayRaw(control,&pixel,1,1,4);auto b=overlay->SetOverlayWidthInMeters(control,1);auto c=overlay->SetOverlayTransformAbsolute(control,vr::TrackingUniverseStanding,&m);
        overlay->SetOverlayAlpha(control,0);overlay->ShowOverlay(control);
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        vr::HmdVector3_t source{{0,1,0}},target{{0,1,-1}};auto convert=sys->GetRawZeroPoseToStandingAbsoluteTrackingPose();
        bool standing_hit=hit(overlay,control,vr::TrackingUniverseStanding,source,target);
        bool raw_hit=hit(overlay,control,vr::TrackingUniverseRawAndUncalibrated,inverse_point(convert,source),inverse_point(convert,target));
        std::printf("{\"transparent_control\":true,\"setup_errors\":[%d,%d,%d],\"standing_hit\":%s,\"raw_hit\":%s}\n",int(a),int(b),int(c),standing_hit?"true":"false",raw_hit?"true":"false");
        overlay->DestroyOverlay(control);
    }
    const char* keys[]={"valve.steam.gamepadui.keyboard","valve.steam.gamepadui.main","system.systemui"};
    vr::VROverlayHandle_t handles[3]{};int raw_hits[3]{},standing_hits[3]{},center_hits[3]{},valid=0,head=0;
    double max_origin_error=0,max_target_error=0;
    for(int i=0;i<3;++i){
        auto find=overlay->FindOverlay(keys[i],&handles[i]);vr::VROverlayTransformType type=vr::VROverlayTransform_Invalid;auto te=overlay->GetOverlayTransformType(handles[i],&type);
        float width=0,aspect=0,curve=0;uint32_t w=0,h=0;overlay->GetOverlayWidthInMeters(handles[i],&width);overlay->GetOverlayTexelAspect(handles[i],&aspect);overlay->GetOverlayCurvature(handles[i],&curve);overlay->GetOverlayTextureSize(handles[i],&w,&h);
        vr::HmdMatrix34_t center{};auto ce=overlay->GetTransformForOverlayCoordinates(handles[i],vr::TrackingUniverseStanding,{{.5,.5}},&center);
        std::printf("{\"overlay\":\"%s\",\"find_error\":%d,\"visible\":%s,\"transform_error\":%d,\"transform_type\":%d,\"width_m\":%.5f,\"texel_aspect\":%.5f,\"curve\":%.5f,\"texture_size\":[%u,%u],\"center_error\":%d,\"center\":[%.5f,%.5f,%.5f]}\n",keys[i],int(find),overlay->IsOverlayVisible(handles[i])?"true":"false",int(te),int(type),width,aspect,curve,w,h,int(ce),center.m[0][3],center.m[1][3],center.m[2][3]);
    }
    auto end=std::chrono::steady_clock::now()+std::chrono::seconds(3);
    while(std::chrono::steady_clock::now()<end){
        input->UpdateActionState(&active,sizeof active,1);vr::VREyeTrackingData_t raw{},standing{};
        auto re=input->GetEyeTrackingDataRelativeToNow(action,vr::TrackingUniverseRawAndUncalibrated,0,&raw,sizeof raw);
        auto se=input->GetEyeTrackingDataRelativeToNow(action,vr::TrackingUniverseStanding,0,&standing,sizeof standing);
        vr::TrackedDevicePose_t pose{};sys->GetDeviceToAbsoluteTrackingPose(vr::TrackingUniverseStanding,0,&pose,1);head+=pose.bPoseIsValid;
        if(!re&&!se&&raw.bActive&&raw.bValid&&raw.bTracked&&standing.bActive&&standing.bValid&&standing.bTracked){
            ++valid;auto convert=sys->GetRawZeroPoseToStandingAbsoluteTrackingPose();
            max_origin_error=std::max(max_origin_error,distance(point(convert,raw.vGazeOrigin),standing.vGazeOrigin));
            max_target_error=std::max(max_target_error,distance(point(convert,raw.vGazeTarget),standing.vGazeTarget));
            for(int i=0;i<3;++i){
                raw_hits[i]+=hit(overlay,handles[i],vr::TrackingUniverseRawAndUncalibrated,raw.vGazeOrigin,raw.vGazeTarget);
                standing_hits[i]+=hit(overlay,handles[i],vr::TrackingUniverseStanding,standing.vGazeOrigin,standing.vGazeTarget);
                vr::HmdMatrix34_t center{};
                if(!overlay->GetTransformForOverlayCoordinates(handles[i],vr::TrackingUniverseStanding,{{.5,.5}},&center)){
                    vr::HmdVector3_t p{{center.m[0][3],center.m[1][3],center.m[2][3]}};
                    center_hits[i]+=hit(overlay,handles[i],vr::TrackingUniverseStanding,standing.vGazeOrigin,p);
                }
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    std::printf("{\"valid_eye_pairs\":%d,\"valid_head_samples\":%d,\"raw_to_standing_origin_max_error_m\":%.6f,\"raw_to_standing_target_max_error_m\":%.6f,\"real_input_sent\":false}\n",valid,head,max_origin_error,max_target_error);
    for(int i=0;i<3;++i)std::printf("{\"overlay\":\"%s\",\"raw_gaze_hits\":%d,\"standing_gaze_hits\":%d,\"standing_center_hits\":%d}\n",keys[i],raw_hits[i],standing_hits[i],center_hits[i]);
    vr::VR_Shutdown();
}
