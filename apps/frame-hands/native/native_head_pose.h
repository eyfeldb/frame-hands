#pragma once
#include "capture_head_pose.h"
#include <stdexcept>
#if defined(FRAME_HANDS_WITH_OPENVR) && !defined(_WIN32)
#include "../system-input/vendor/openvr.h"
#include <dlfcn.h>
#include <ctime>
#endif
namespace fhn {
// Opt-in Frame-local OpenVR Background client. No scene, camera or input creation.
// API calls are synchronous: elapsed time is recorded/rejected if stale, not timed out.
class NativeHeadPose {
#if defined(FRAME_HANDS_WITH_OPENVR) && !defined(_WIN32)
    void* library_=nullptr;void(*shutdown_)()=nullptr;vr::IVRSystem* system_=nullptr;
    std::uint64_t generation_=1;bool stopped_=false;
    static std::uint64_t now(){timespec t{};if(clock_gettime(CLOCK_MONOTONIC,&t))return 0;return std::uint64_t(t.tv_sec)*1000000000+std::uint64_t(t.tv_nsec);}
    void close(){if(shutdown_)shutdown_();shutdown_=nullptr;system_=nullptr;if(library_)dlclose(library_);library_=nullptr;}
#endif
public:
    NativeHeadPose(const NativeHeadPose&)=delete;NativeHeadPose& operator=(const NativeHeadPose&)=delete;
    explicit NativeHeadPose(const char* path){
#if defined(FRAME_HANDS_WITH_OPENVR) && !defined(_WIN32)
        if(!path||path[0]!='/')throw std::invalid_argument("Absolute local OpenVR library required");
        library_=dlopen(path,RTLD_NOW|RTLD_LOCAL);if(!library_)throw std::runtime_error("OpenVR library unavailable");
        auto init=reinterpret_cast<uint32_t(*)(vr::EVRInitError*,vr::EVRApplicationType,const char*)>(dlsym(library_,"VR_InitInternal2"));
        auto iface=reinterpret_cast<void*(*)(const char*,vr::EVRInitError*)>(dlsym(library_,"VR_GetGenericInterface"));
        auto stop=reinterpret_cast<void(*)()>(dlsym(library_,"VR_ShutdownInternal"));
        if(!init||!iface||!stop){close();throw std::runtime_error("Incomplete OpenVR library");}
        shutdown_=stop;vr::EVRInitError error=vr::VRInitError_None;init(&error,vr::VRApplication_Background,nullptr);
        if(error){close();throw std::runtime_error("OpenVR background initialization failed");}
        system_=static_cast<vr::IVRSystem*>(iface(vr::IVRSystem_Version,&error));
        if(!system_||error){close();throw std::runtime_error("OpenVR system interface unavailable");}
        error=system_->SetSDKVersion(vr::k_nSteamVRVersionMajor,vr::k_nSteamVRVersionMinor,vr::k_nSteamVRVersionBuild);
        if(error){close();throw std::runtime_error("OpenVR SDK version rejected");}
#else
        (void)path;throw std::runtime_error("Local capture head support not compiled");
#endif
    }
    ~NativeHeadPose(){
#if defined(FRAME_HANDS_WITH_OPENVR) && !defined(_WIN32)
        close();
#endif
    }
    CaptureHeadPose query(std::int64_t raw,std::uint64_t capture){
        CaptureHeadPose out;out.raw_ns=raw;out.capture_ns=capture;
#if defined(FRAME_HANDS_WITH_OPENVR) && !defined(_WIN32)
        out.generation=generation_;if(stopped_||!system_)return out;
        vr::VREvent_t event{};int events=0;
        while(events<64&&system_->PollNextEvent(&event,sizeof(event))){++events;
            bool head_event=event.trackedDeviceIndex==vr::k_unTrackedDeviceIndex_Hmd&&
                (event.eventType==vr::VREvent_TrackedDeviceActivated||event.eventType==vr::VREvent_TrackedDeviceDeactivated||event.eventType==vr::VREvent_TrackedDeviceUpdated);
            if(head_event||event.eventType==vr::VREvent_ChaperoneUniverseHasChanged||event.eventType==vr::VREvent_ChaperoneDataHasChanged||event.eventType==vr::VREvent_SeatedZeroPoseReset)++generation_;
            if(event.eventType==vr::VREvent_Quit){stopped_=true;++generation_;}
        }
        out.generation=generation_;if(stopped_||events==64)return out;
        out.query_ns=now();float age=0;if(raw<=0||!head_request_age(capture,out.query_ns,age))return out;
        vr::TrackedDevicePose_t pose{};
        system_->GetDeviceToAbsoluteTrackingPose(vr::TrackingUniverseRawAndUncalibrated,-age,&pose,1);
        out.returned_ns=now();if(!pose.bPoseIsValid||!pose.bDeviceIsConnected||pose.eTrackingResult!=vr::TrackingResult_Running_OK)return out;
        double matrix[12];for(int i=0;i<12;++i)matrix[i]=pose.mDeviceToAbsoluteTracking.m[i/4][i%4];
        out.valid=head_from_matrix(matrix,out.head);out.valid=out.usable(raw);
#endif
        return out;
    }
};
}
