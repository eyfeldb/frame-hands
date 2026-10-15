// Read-only check of the registered pointer's runtime identity and inactive state.
#include <openvr.h>
#include <cstdio>
#include <cstring>
int main(){
    vr::EVRInitError e=vr::VRInitError_None;auto s=vr::VR_Init(&e,vr::VRApplication_Utility);
    if(e||!s){std::printf("{\"init_error\":%d}\n",int(e));return 1;}
    bool found=false;
    for(uint32_t i=0;i<vr::k_unMaxTrackedDeviceCount;++i){
        char type[128]{};s->GetStringTrackedDeviceProperty(i,vr::Prop_ControllerType_String,type,sizeof type);
        if(std::strcmp(type,"fh_pointer"))continue;
        found=true;
        auto role=s->GetInt32TrackedDeviceProperty(i,vr::Prop_ControllerRoleHint_Int32);
        std::printf("{\"pointer_loaded\":true,\"index\":%u,\"connected\":%s,\"role_hint\":%d,\"opt_out\":%s}\n",i,s->IsTrackedDeviceConnected(i)?"true":"false",role,role==vr::TrackedControllerRole_OptOut?"true":"false");
    }
    if(!found)std::puts("{\"pointer_loaded\":false}");
    vr::VR_Shutdown();return found?0:3;
}
