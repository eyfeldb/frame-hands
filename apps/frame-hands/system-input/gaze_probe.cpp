// Read-only public eye-input probe. No camera, controller, overlay or click created.
#include "openvr.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <thread>
int main(int argc,char** argv) {
    if(argc!=2)return 2;
    vr::EVRInitError error=vr::VRInitError_None;
    auto system=vr::VR_Init(&error,vr::VRApplication_Overlay);
    if(error || !system){std::printf("{\"init_error\":%d}\n",int(error));return 1;}
    auto input=vr::VRInput();
    auto manifest=input->SetActionManifestPath(argv[1]);
    vr::VRActionSetHandle_t set=0;vr::VRActionHandle_t action=0;
    auto set_error=input->GetActionSetHandle("/actions/main",&set);
    auto action_error=input->GetActionHandle("/actions/main/in/gaze",&action);
    vr::VRActiveActionSet_t active{};active.ulActionSet=set;
    unsigned samples=0,eye_active=0,valid=0,tracked=0,foveation=0,head_tracked=0;
    int update_error=0,eye_error=0;
    double lo[3]={1e9,1e9,1e9},hi[3]={-1e9,-1e9,-1e9};
    auto end=std::chrono::steady_clock::now()+std::chrono::seconds(3);
    while(std::chrono::steady_clock::now()<end){
        update_error=int(input->UpdateActionState(&active,sizeof(active),1));
        vr::VREyeTrackingData_t eyes{};
        eye_error=int(input->GetEyeTrackingDataRelativeToNow(action,vr::TrackingUniverseStanding,0,&eyes,sizeof(eyes)));
        ++samples;eye_active+=eyes.bActive;valid+=eyes.bValid;tracked+=eyes.bTracked;
        if(!eye_error && eyes.bActive && eyes.bValid && eyes.bTracked){
            double direction[3],length=0;
            for(int k=0;k<3;++k){direction[k]=eyes.vGazeTarget.v[k]-eyes.vGazeOrigin.v[k];length+=direction[k]*direction[k];}
            length=std::sqrt(length);
            if(std::isfinite(length)&&length>0.0001)for(int k=0;k<3;++k){lo[k]=std::min(lo[k],direction[k]/length);hi[k]=std::max(hi[k],direction[k]/length);}
        }
        vr::HmdVector2_t left{},right{};
        foveation+=system->GetEyeTrackedFoveationCenter(&left,&right);
        vr::TrackedDevicePose_t head{};
        system->GetDeviceToAbsoluteTrackingPose(vr::TrackingUniverseStanding,0,&head,1);
        head_tracked+=head.bPoseIsValid && head.bDeviceIsConnected;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    std::printf("{\"manifest_error\":%d,\"set_error\":%d,\"action_error\":%d,\"update_error\":%d,\"eye_error\":%d,\"samples\":%u,\"active\":%u,\"valid\":%u,\"tracked\":%u,\"foveation_available\":%u,\"head_tracked\":%u,\"direction_range\":[%.6f,%.6f,%.6f],\"input_injected\":false}\n",
        int(manifest),int(set_error),int(action_error),update_error,eye_error,samples,eye_active,valid,tracked,foveation,head_tracked,
        tracked?hi[0]-lo[0]:0,tracked?hi[1]-lo[1]:0,tracked?hi[2]-lo[2]:0);
    vr::VR_Shutdown();return 0;
}
