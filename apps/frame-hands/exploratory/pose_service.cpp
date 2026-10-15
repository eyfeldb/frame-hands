// Read-only background HMD pose service. No action sets, calibration or driver writes.
#define main unused_probe_main
#include "../pc-input/probe_openvr.cpp"
#undef main
int main(int argc,char** argv){
    std::setvbuf(stdout,nullptr,_IONBF,0);
    if(argc!=2)return 2;auto lib=LoadLibraryA(argv[1]);if(!lib)return 3;
    auto init=symbol<uint32_t(*)(EVRInitError*,EVRApplicationType,const char*)>(lib,"VR_InitInternal2");
    auto iface=symbol<void*(*)(const char*,EVRInitError*)>(lib,"VR_GetGenericInterface");
    auto close=symbol<void(*)()>(lib,"VR_ShutdownInternal");if(!init||!iface||!close)return 4;
    EVRInitError error;init(&error,VRApplication_Background,nullptr);if(error)return 5;
    auto system=static_cast<IVRSystem*>(iface(IVRSystem_Version,&error));if(!system){close();return 6;}
    double age=0;
    while(std::scanf("%lf",&age)==1){
        if(age<0||age>5000)break;
        TrackedDevicePose_t capture{},now{};
        system->GetDeviceToAbsoluteTrackingPose(TrackingUniverseRawAndUncalibrated,-float(age)*.001f,&capture,1);
        system->GetDeviceToAbsoluteTrackingPose(TrackingUniverseRawAndUncalibrated,0,&now,1);
        std::printf("{\"valid\":%s,\"tick_ms\":%llu,\"capture\":[",capture.bPoseIsValid&&capture.bDeviceIsConnected&&now.bPoseIsValid&&now.bDeviceIsConnected?"true":"false",(unsigned long long)GetTickCount64());
        for(int i=0;i<12;++i)std::printf("%s%.9g",i?",":"",capture.mDeviceToAbsoluteTracking.m[i/4][i%4]);
        std::printf("],\"current\":[");
        for(int i=0;i<12;++i)std::printf("%s%.9g",i?",":"",now.mDeviceToAbsoluteTracking.m[i/4][i%4]);
        std::puts("]}");
    }
    close();return 0;
}
