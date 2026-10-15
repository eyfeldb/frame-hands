// One native SteamVR menu; opening it does not change mode or start cameras.
#include <openvr.h>
#include <dlfcn.h>
#include <cstdio>
int main(int argc,char** argv){
 if(argc!=2)return 2;
 auto lib=dlopen("/opt/steamvr/bin/linuxarm64/libopenvr_api.so",RTLD_NOW|RTLD_LOCAL);if(!lib)return 10;
 auto init=reinterpret_cast<uint32_t(*)(vr::EVRInitError*,vr::EVRApplicationType,const char*)>(dlsym(lib,"VR_InitInternal2"));
 auto get=reinterpret_cast<void*(*)(const char*,vr::EVRInitError*)>(dlsym(lib,"VR_GetGenericInterface"));
 auto shutdown=reinterpret_cast<void(*)()>(dlsym(lib,"VR_ShutdownInternal"));if(!init||!get||!shutdown)return 11;
 vr::EVRInitError error{};init(&error,vr::VRApplication_Overlay,nullptr);if(error)return 12;
 auto overlay=static_cast<vr::IVROverlay*>(get(vr::IVROverlay_Version,&error));if(!overlay||error){shutdown();return 13;}
 auto result=overlay->ShowMessageOverlay(argv[1],"Frame Hands","Hands","Controllers","Stop tracking","Close");
 shutdown();std::printf("selection=%d\n",int(result));return int(result);
}
