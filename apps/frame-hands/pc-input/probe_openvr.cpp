#define NOMINMAX
#include <windows.h>
#include <openvr.h>
#include <cstdio>
#include <string>
#include <chrono>
#include <thread>
using namespace vr;
template<class T>T symbol(HMODULE lib,const char* name){auto raw=GetProcAddress(lib,name);T result=nullptr;static_assert(sizeof raw==sizeof result);std::memcpy(&result,&raw,sizeof result);return result;}
int main(int argc,char** argv){
    if(argc<2)return 2;HMODULE lib=LoadLibraryA(argv[1]);if(!lib)return 3;
    auto init=symbol<uint32_t(*)(EVRInitError*,EVRApplicationType,const char*)>(lib,"VR_InitInternal2");
    auto iface=symbol<void*(*)(const char*,EVRInitError*)>(lib,"VR_GetGenericInterface");
    auto close=symbol<void(*)()>(lib,"VR_ShutdownInternal");if(!init||!iface||!close)return 4;
    EVRInitError error;init(&error,VRApplication_Background,nullptr);if(error){std::printf("init_error=%d\n",int(error));return 5;}
    auto system=static_cast<IVRSystem*>(iface(IVRSystem_Version,&error));auto input=static_cast<IVRInput*>(iface(IVRInput_Version,&error));
    if(!system||!input){close();return 6;}
    for(uint32_t i=0;i<k_unMaxTrackedDeviceCount;++i){
        auto kind=system->GetTrackedDeviceClass(i);if(kind==TrackedDeviceClass_Invalid)continue;
        char type[128]{},model[128]{};system->GetStringTrackedDeviceProperty(i,Prop_ControllerType_String,type,sizeof type);system->GetStringTrackedDeviceProperty(i,Prop_ModelNumber_String,model,sizeof model);
        std::printf("device=%u class=%d connected=%d role=%d type=%s model=%s\n",i,int(kind),system->IsTrackedDeviceConnected(i),int(system->GetControllerRoleForTrackedDeviceIndex(i)),type,model);
    }
    if(argc>=3){
        auto result=input->SetActionManifestPath(argv[2]);std::printf("manifest=%d\n",int(result));
        VRActionSetHandle_t set=0;input->GetActionSetHandle("/actions/main",&set);VRActiveActionSet_t active{};active.ulActionSet=set;
        VRActionHandle_t skeleton[2]{};input->GetActionHandle("/actions/main/in/skeletonleft",&skeleton[0]);input->GetActionHandle("/actions/main/in/skeletonright",&skeleton[1]);
        unsigned counts[2]{};auto end=std::chrono::steady_clock::now()+std::chrono::seconds(5);
        while(std::chrono::steady_clock::now()<end){input->UpdateActionState(&active,sizeof active,1);for(int side=0;side<2;++side){InputSkeletalActionData_t data{};auto e=input->GetSkeletalActionData(skeleton[side],&data,sizeof data);if(!e&&data.bActive)++counts[side];}std::this_thread::sleep_for(std::chrono::milliseconds(10));}
        for(int side=0;side<2;++side){uint32_t count=0;EVRSkeletalTrackingLevel level{};auto e=input->GetBoneCount(skeleton[side],&count);auto l=input->GetSkeletalTrackingLevel(skeleton[side],&level);std::printf("side=%d active_samples=%u bones=%u bone_error=%d tracking_level=%d level_error=%d\n",side,counts[side],count,int(e),int(level),int(l));}
    }
    if(argc>=3){
        for(int side=0;side<2;++side){VRActionHandle_t action{};input->GetActionHandle(side?"/actions/main/in/skeletonright":"/actions/main/in/skeletonleft",&action);
            VRBoneTransform_t bones[31]{};auto e=input->GetSkeletalReferenceTransforms(action,VRSkeletalTransformSpace_Parent,VRSkeletalReferencePose_BindPose,bones,31);std::printf("reference_side=%d error=%d\n",side,int(e));
            VRBoneTransform_t live[31]{};auto le=input->GetSkeletalBoneData(action,VRSkeletalTransformSpace_Parent,VRSkeletalMotionRange_WithoutController,live,31);std::printf("live_wrist side=%d error=%d q=%.6g,%.6g,%.6g,%.6g\n",side,int(le),live[1].orientation.w,live[1].orientation.x,live[1].orientation.y,live[1].orientation.z);
            if(!e)for(int n=0;n<31;++n){auto b=bones[n];std::printf("rest %d %d %.12g %.12g %.12g %.12g %.12g %.12g %.12g\n",side,n,b.position.v[0],b.position.v[1],b.position.v[2],b.orientation.w,b.orientation.x,b.orientation.y,b.orientation.z);}
        }
    }
    close();return 0;
}
