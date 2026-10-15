// Read-only SteamVR HMD connection/pose check. No scene, cameras or actions.
#define main unused_probe_main
#include "../pc-input/probe_openvr.cpp"
#undef main
int main(int argc,char** argv){
    if(argc!=2)return 2;
    auto lib=LoadLibraryA(argv[1]);if(!lib)return 3;
    auto init=symbol<uint32_t(*)(EVRInitError*,EVRApplicationType,const char*)>(lib,"VR_InitInternal2");
    auto iface=symbol<void*(*)(const char*,EVRInitError*)>(lib,"VR_GetGenericInterface");
    auto close=symbol<void(*)()>(lib,"VR_ShutdownInternal");
    if(!init||!iface||!close)return 4;
    EVRInitError error;init(&error,VRApplication_Background,nullptr);
    if(error){std::printf("{\"runtime_error\":%d}\n",int(error));return 0;}
    auto system=static_cast<IVRSystem*>(iface(IVRSystem_Version,&error));
    if(!system){close();return 6;}
    TrackedDevicePose_t pose{};
    system->GetDeviceToAbsoluteTrackingPose(TrackingUniverseRawAndUncalibrated,0,&pose,1);
    char type[128]{};
    system->GetStringTrackedDeviceProperty(0,Prop_ControllerType_String,type,sizeof type);
    std::printf("{\"connected\":%s,\"pose_valid\":%s,\"frame_hmd\":%s,\"tracking_result\":%d}\n",
        pose.bDeviceIsConnected?"true":"false",pose.bPoseIsValid?"true":"false",
        std::string(type)=="frame_hmd"?"true":"false",int(pose.eTrackingResult));
    close();return 0;
}
