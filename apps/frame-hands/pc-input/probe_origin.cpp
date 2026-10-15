// Read-only background probe. Does not recenter, reload or write calibration.
#define main unused_input_probe_main
#include "probe_openvr.cpp"
#undef main
void matrix(const char* name,HmdMatrix34_t m){
    std::printf("%s=",name);for(int r=0;r<3;++r)for(int c=0;c<4;++c)std::printf("%s%.6f",r||c?",":"",m.m[r][c]);std::puts("");
}
int main(int argc,char** argv){
    std::setvbuf(stdout,nullptr,_IONBF,0);
    if(argc!=2)return 2;auto lib=LoadLibraryA(argv[1]);if(!lib)return 3;
    auto init=symbol<uint32_t(*)(EVRInitError*,EVRApplicationType,const char*)>(lib,"VR_InitInternal2");
    auto iface=symbol<void*(*)(const char*,EVRInitError*)>(lib,"VR_GetGenericInterface");
    auto close=symbol<void(*)()>(lib,"VR_ShutdownInternal");if(!init||!iface||!close)return 4;
    EVRInitError error;init(&error,VRApplication_Background,nullptr);if(error){std::printf("init_error=%d\n",int(error));return 5;}
    auto system=static_cast<IVRSystem*>(iface(IVRSystem_Version,&error));
    auto chaperone=static_cast<IVRChaperone*>(iface(IVRChaperone_Version,&error));
    if(!system||!chaperone){close();return 6;}
    // Use pointer-output APIs: this MinGW build cannot safely consume the
    // MSVC C++ ABI's matrix-by-value IVRSystem methods.
    auto setup=static_cast<IVRChaperoneSetup*>(iface(IVRChaperoneSetup_Version,&error));
    HmdMatrix34_t m{};
    if(setup&&setup->GetLiveSeatedZeroPoseToRawTrackingPose(&m))matrix("live_seated_to_raw",m);
    float x=0,z=0;bool area=chaperone->GetPlayAreaSize(&x,&z);
    std::printf("calibration_state=%d play_area_valid=%d play_area_m=%.3f,%.3f\n",int(chaperone->GetCalibrationState()),area,x,z);
    for(auto origin:{TrackingUniverseRawAndUncalibrated,TrackingUniverseStanding,TrackingUniverseSeated}){
        TrackedDevicePose_t poses[k_unMaxTrackedDeviceCount]{};system->GetDeviceToAbsoluteTrackingPose(origin,0,poses,k_unMaxTrackedDeviceCount);
        for(uint32_t i=0;i<k_unMaxTrackedDeviceCount;++i){auto kind=system->GetTrackedDeviceClass(i);
            if(kind!=TrackedDeviceClass_HMD&&kind!=TrackedDeviceClass_Controller)continue;
            auto p=poses[i];std::printf("universe=%d device=%u class=%d connected=%d valid=%d result=%d xyz=%.6f,%.6f,%.6f\n",int(origin),i,int(kind),p.bDeviceIsConnected,p.bPoseIsValid,int(p.eTrackingResult),p.mDeviceToAbsoluteTracking.m[0][3],p.mDeviceToAbsoluteTracking.m[1][3],p.mDeviceToAbsoluteTracking.m[2][3]);
        }
    }
    close();return 0;
}
