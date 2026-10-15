// Read or set the Frame's controller optical-tracking mode (SteamVR setting driver_cv/controllerTracking3) so the
// forward cameras stay in normal exposure while optical hands are in use ('none'), and go back to the Frame's
// default ('alternate') when the wearer returns to controllers. Background app: starts no scene, no cameras.
// usage: cvtrack <openvr_api.so> get | set <mode> [section]
#include <openvr.h>
#include <cstdio>
#include <cstring>
#include <dlfcn.h>
using namespace vr;
int main(int argc,char** argv){
    if(argc<3){std::puts("usage: cvtrack <openvr_api.so> get|set <mode> [section]");return 2;}
    void* lib=dlopen(argv[1],RTLD_NOW);if(!lib){std::printf("dlopen failed: %s\n",dlerror());return 3;}
    auto init=reinterpret_cast<uint32_t(*)(EVRInitError*,EVRApplicationType,const char*)>(dlsym(lib,"VR_InitInternal2"));
    auto iface=reinterpret_cast<void*(*)(const char*,EVRInitError*)>(dlsym(lib,"VR_GetGenericInterface"));
    auto shutdown=reinterpret_cast<void(*)()>(dlsym(lib,"VR_ShutdownInternal"));
    if(!init||!iface||!shutdown)return 4;
    EVRInitError error{};init(&error,VRApplication_Background,nullptr);if(error){std::printf("init_error=%d\n",int(error));return 5;}
    auto settings=static_cast<IVRSettings*>(iface(IVRSettings_Version,&error));if(!settings){std::printf("settings_error=%d\n",int(error));shutdown();return 6;}
    const char* section=argc>=5?argv[4]:"driver_cv";const char* key="controllerTracking3";
    char value[64]{};EVRSettingsError se{};
    settings->GetString(section,key,value,sizeof value,&se);
    std::printf("before: %s/%s = '%s' (error %d)\n",section,key,value,int(se));
    if(!std::strcmp(argv[2],"set")&&argc>=4){
        settings->SetString(section,key,argv[3],&se);std::printf("set '%s' -> error %d\n",argv[3],int(se));
        std::memset(value,0,sizeof value);settings->GetString(section,key,value,sizeof value,&se);
        std::printf("after: %s/%s = '%s' (error %d)\n",section,key,value,int(se));
    }
    shutdown();return 0;
}
