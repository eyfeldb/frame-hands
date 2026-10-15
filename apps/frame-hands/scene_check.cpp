// Read-only utility connection: never claims the XR scene or changes settings.
#include "openvr.h"
#include <cstdio>

int main() {
    vr::EVRInitError error=vr::VRInitError_None;
    vr::VR_Init(&error,vr::VRApplication_Utility);
    if(error!=vr::VRInitError_None){
        std::printf("{\"available\":false,\"error\":%d}\n",int(error));return 2;
    }
    auto *applications=vr::VRApplications();
    if(!applications){vr::VR_Shutdown();return 3;}
    const auto owner=applications->GetCurrentSceneProcessId();
    std::printf("{\"available\":true,\"scene_pid\":%u}\n",owner);
    vr::VR_Shutdown();
    return 0;
}
