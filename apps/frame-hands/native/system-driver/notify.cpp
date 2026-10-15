// Headset notification only. No camera, scene, controller input or app launch.
#include <openvr.h>
#include <dlfcn.h>
#include <cstdio>
#include <string>
#include <unistd.h>
int main(int argc,char** argv){
    if(argc!=2)return 2;
    void* lib=dlopen("/opt/steamvr/bin/linuxarm64/libopenvr_api.so",RTLD_NOW|RTLD_LOCAL);if(!lib)return 3;
    auto init=reinterpret_cast<uint32_t(*)(vr::EVRInitError*,vr::EVRApplicationType,const char*)>(dlsym(lib,"VR_InitInternal2"));
    auto get=reinterpret_cast<void*(*)(const char*,vr::EVRInitError*)>(dlsym(lib,"VR_GetGenericInterface"));
    auto shutdown=reinterpret_cast<void(*)()>(dlsym(lib,"VR_ShutdownInternal"));
    if(!init||!get||!shutdown)return 4;
    vr::EVRInitError error=vr::VRInitError_None;init(&error,vr::VRApplication_Background,nullptr);if(error)return 5;
    auto* overlay=static_cast<vr::IVROverlay*>(get(vr::IVROverlay_Version,&error));
    auto* notifications=static_cast<vr::IVRNotifications*>(get(vr::IVRNotifications_Version,&error));
    if(!overlay||!notifications){shutdown();return 6;}
    vr::VROverlayHandle_t handle=0;std::string key="frame.hands.status."+std::to_string(getpid());
    if(overlay->CreateOverlay(key.c_str(),"Frame Hands",&handle)){shutdown();return 7;}
    vr::VRNotificationId id=0;
    auto result=notifications->CreateNotification(handle,0,vr::EVRNotificationType_Transient,argv[1],vr::EVRNotificationStyle_Application,nullptr,&id);
    std::printf("headset_notification=%d\n",int(result));std::fflush(stdout);
    if(!result){sleep(6);notifications->RemoveNotification(id);}
    overlay->DestroyOverlay(handle);shutdown();dlclose(lib);return int(result);
}
