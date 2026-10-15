// Loads the actual Windows DLL into a mock server. No SteamVR or input injection.
#define NOMINMAX
#ifdef _WIN32
#include <windows.h>
#endif
#include <openvr_driver.h>
#include "skeleton.h"
#undef NDEBUG
#include <cassert>
#include <cstdio>
#include <map>
#include <string>
#include <fstream>
#include <vector>
using namespace vr;
struct Input:IVRDriverInput {
    uint64_t next=1;std::map<uint64_t,std::string> names;std::map<uint64_t,uint64_t> containers;
    std::map<uint64_t,bool> states;std::map<uint64_t,std::array<VRBoneTransform_t,31>> bones;unsigned updates[2]{};
    EVRInputError create(uint64_t c,const char* n,uint64_t* h){*h=next++;names[*h]=n;containers[*h]=c;return VRInputError_None;}
    EVRInputError CreateBooleanComponent(PropertyContainerHandle_t c,const char* n,VRInputComponentHandle_t* h)override{return create(c,n,h);}
    EVRInputError UpdateBooleanComponent(VRInputComponentHandle_t h,bool b,double)override{states[h]=b;return VRInputError_None;}
    EVRInputError CreateScalarComponent(PropertyContainerHandle_t c,const char* n,VRInputComponentHandle_t* h,EVRScalarType,EVRScalarUnits)override{return create(c,n,h);}
    EVRInputError UpdateScalarComponent(VRInputComponentHandle_t,float v,double)override{assert(std::isfinite(v)&&v>=-1&&v<=1);return VRInputError_None;}
    EVRInputError CreateHapticComponent(PropertyContainerHandle_t c,const char* n,VRInputComponentHandle_t* h)override{return create(c,n,h);}
    EVRInputError CreateSkeletonComponent(PropertyContainerHandle_t c,const char* n,const char* s,const char* base,EVRSkeletalTrackingLevel level,const VRBoneTransform_t*,uint32_t,VRInputComponentHandle_t* h)override{
        assert(level==VRSkeletalTracking_Full&&!std::strcmp(base,"/pose/raw"));assert(std::strstr(s,c==5?"left":"right"));return create(c,n,h);
    }
    EVRInputError UpdateSkeletonComponent(VRInputComponentHandle_t h,EVRSkeletalMotionRange range,const VRBoneTransform_t* b,uint32_t n)override{
        assert(n==31);std::copy(b,b+31,bones[h].begin());++updates[int(range)];for(uint32_t i=0;i<n;++i){auto q=b[i].orientation;assert(std::abs(q.w*q.w+q.x*q.x+q.y*q.y+q.z*q.z-1)<.0001);}
        return VRInputError_None;
    }
    EVRInputError CreatePoseComponent(PropertyContainerHandle_t,const char*,VRInputComponentHandle_t*)override{return VRInputError_InvalidParam;}
    EVRInputError UpdatePoseComponent(VRInputComponentHandle_t,const HmdMatrix34_t*,double)override{return VRInputError_InvalidParam;}
    EVRInputError CreateEyeTrackingComponent(PropertyContainerHandle_t,const char*,VRInputComponentHandle_t*)override{return VRInputError_InvalidParam;}
    EVRInputError UpdateEyeTrackingComponent(VRInputComponentHandle_t,const VREyeTrackingData_t*,double)override{return VRInputError_InvalidParam;}
    bool any(uint64_t container){for(auto v:states)if(v.second&&containers[v.first]==container)return true;return false;}
    bool pressed(uint64_t container,const char* path){for(auto v:states)if(v.second&&containers[v.first]==container&&names[v.first]==path)return true;return false;}
} input;
struct Properties:IVRProperties {
    std::map<uint64_t,int> roles,classes;std::map<uint64_t,std::string> models;
    ETrackedPropertyError ReadPropertyBatch(PropertyContainerHandle_t c,PropertyRead_t* batch,uint32_t n)override{
        for(uint32_t i=0;i<n;++i){auto& b=batch[i];b.eError=TrackedProp_UnknownProperty;
            if(b.prop==Prop_DeviceClass_Int32&&classes.count(c)&&b.unBufferSize>=4){
                std::memcpy(b.pvBuffer,&classes[c],4);b.unTag=k_unInt32PropertyTag;b.unRequiredBufferSize=4;b.eError=TrackedProp_Success;
            }
        }return n?batch[0].eError:TrackedProp_Success;
    }
    ETrackedPropertyError WritePropertyBatch(PropertyContainerHandle_t c,PropertyWrite_t* b,uint32_t n)override{for(uint32_t i=0;i<n;++i){if(b[i].prop==Prop_RenderModelName_String)models[c]=static_cast<const char*>(b[i].pvBuffer);if(b[i].prop==Prop_ControllerRoleHint_Int32)std::memcpy(&roles[c],b[i].pvBuffer,4);b[i].eError=TrackedProp_Success;}return TrackedProp_Success;}
    const char* GetPropErrorNameFromEnum(ETrackedPropertyError)override{return "test";}
    PropertyContainerHandle_t TrackedDeviceToPropertyContainer(TrackedDeviceIndex_t i)override{return i+1;}
} properties;
struct Host:IVRServerDriverHost {
    std::vector<VREvent_t> events;
    ITrackedDeviceServerDriver* device[2]{};DriverPose_t pose[2]{};unsigned count=0;bool head=true;float oldest=0;bool recorded=false,capture=true;float recorded_head[12]{};
    bool TrackedDeviceAdded(const char*,ETrackedDeviceClass cls,ITrackedDeviceServerDriver* d)override{assert(cls==TrackedDeviceClass_Controller&&count<2);device[count++]=d;return true;}
    void TrackedDevicePoseUpdated(uint32_t i,const DriverPose_t& p,uint32_t n)override{assert(i>=4&&i<=5&&n==sizeof p);if(!p.deviceIsConnected)assert(!input.any(i+1));pose[i-4]=p;}
    void VsyncEvent(double)override{}
    void VendorSpecificEvent(uint32_t,EVREventType,const VREvent_Data_t&,double)override{}
    bool IsExiting()override{return false;}
    bool PollNextEvent(VREvent_t* out,uint32_t)override{if(events.empty())return false;*out=events.front();events.erase(events.begin());return true;}
    void GetRawTrackedDevicePoses(float age,TrackedDevicePose_t* p,uint32_t n)override{oldest=std::min(oldest,age);for(uint32_t i=0;i<n;++i)p[i]={};p[0].bPoseIsValid=p[0].bDeviceIsConnected=head;p[0].mDeviceToAbsoluteTracking.m[0][0]=p[0].mDeviceToAbsoluteTracking.m[1][1]=p[0].mDeviceToAbsoluteTracking.m[2][2]=1;p[0].mDeviceToAbsoluteTracking.m[1][3]=1.6f;if(recorded){std::memcpy(p[0].mDeviceToAbsoluteTracking.m,recorded_head,sizeof recorded_head);p[0].bPoseIsValid=p[0].bDeviceIsConnected=capture;}}
    void RequestRestart(const char*,const char*,const char*,const char*)override{assert(false);}
    uint32_t GetFrameTimings(Compositor_FrameTiming*,uint32_t)override{return 0;}
    void SetDisplayEyeToHead(uint32_t,const HmdMatrix34_t&,const HmdMatrix34_t&)override{}
    void SetDisplayProjectionRaw(uint32_t,const HmdRect2_t&,const HmdRect2_t&)override{}
    void SetRecommendedRenderTargetSize(uint32_t,uint32_t,uint32_t)override{}
} host;
struct Logger:IVRDriverLog{void Log(const char* text)override{std::puts(text);}} logger;
struct Settings:IVRSettings {
    const char* GetSettingsErrorNameFromEnum(EVRSettingsError)override{return "test";}
    void SetBool(const char*,const char*,bool,EVRSettingsError*)override{assert(false);}
    void SetInt32(const char*,const char*,int32_t,EVRSettingsError*)override{assert(false);}
    void SetFloat(const char*,const char*,float,EVRSettingsError*)override{assert(false);}
    void SetString(const char*,const char*,const char*,EVRSettingsError*)override{assert(false);}
    bool GetBool(const char*,const char*,EVRSettingsError*)override{assert(false);return false;}
    int32_t GetInt32(const char*,const char*,EVRSettingsError*)override{assert(false);return 0;}
    float GetFloat(const char*,const char*,EVRSettingsError*)override{assert(false);return 0;}
    void GetString(const char*,const char*,char*,uint32_t,EVRSettingsError*)override{assert(false);}
    void RemoveSection(const char*,EVRSettingsError*)override{assert(false);}
    void RemoveKeyInSection(const char*,const char*,EVRSettingsError*)override{assert(false);}
} settings;
struct Resources:IVRResources {
    uint32_t LoadSharedResource(const char*,char*,uint32_t)override{assert(false);return 0;}
    uint32_t GetResourceFullPath(const char*,const char*,char*,uint32_t)override{assert(false);return 0;}
} resources;
struct Manager:IVRDriverManager {
    uint32_t GetDriverCount()const override{return 1;}
    uint32_t GetDriverName(DriverId_t,char*,uint32_t)override{assert(false);return 0;}
    DriverHandle_t GetDriverHandle(const char*)override{return 1;}
    bool IsEnabled(DriverId_t)const override{return true;}
} manager;
struct Context:IVRDriverContext {
    void* GetGenericInterface(const char* name,EVRInitError* error)override{
        if(error)*error=VRInitError_None;
        if(!std::strcmp(name,IVRServerDriverHost_Version))return &host;
        if(!std::strcmp(name,IVRProperties_Version))return &properties;
        if(!std::strcmp(name,IVRDriverInput_Version))return &input;
        if(!std::strcmp(name,IVRDriverLog_Version))return &logger;
        if(!std::strcmp(name,IVRSettings_Version))return &settings;
        if(!std::strcmp(name,IVRResources_Version))return &resources;
        if(!std::strcmp(name,IVRDriverManager_Version))return &manager;
        if(error)*error=VRInitError_Init_InterfaceNotFound;return nullptr;
    }
    DriverHandle_t GetDriverHandle()override{return 1;}
} context;
#ifndef FRAME_HANDS_DRIVER_MOCKS_ONLY
int main(int argc,char** argv){
    assert(argc==2);std::string dll=argv[1];auto path=dll.substr(0,dll.find_last_of("\\/"))+"/input.bin";
    assert(GetFileAttributesA(path.c_str())==INVALID_FILE_ATTRIBUTES); // never overwrite live transport
    HMODULE library=LoadLibraryA(argv[1]);assert(library);
    auto symbol=GetProcAddress(library,"HmdDriverFactory");void*(*factory)(const char*,int*)=nullptr;static_assert(sizeof symbol==sizeof factory);std::memcpy(&factory,&symbol,sizeof factory);assert(factory);
    int error=0;auto provider=static_cast<IServerTrackedDeviceProvider*>(factory(IServerTrackedDeviceProvider_Version,&error));assert(provider);
    assert(provider->Init(&context)==VRInitError_None&&host.count==2);
    for(int side=0;side<2;++side)assert(host.device[side]->Activate(side+4)==VRInitError_None);
    provider->RunFrame();assert(!host.pose[0].deviceIsConnected&&!host.pose[1].deviceIsConnected);
    fhs::Packet packet;
    for(int side=0;side<2;++side){std::array<fhs::Transform,31> local{};for(int i=0;i<31;++i)local[i]={fhs::rest[side][i].p,fhs::rest[side][i].q};auto f=fhs::forward(local);
        packet.hands[side].id=side+1;packet.hands[side].flags=fhs::Present|fhs::Stereo;
        for(int i=0;i<21;++i){auto p=f[fhs::landmark_bone[i]].p+fhs::V{side?.2:-.2,0,-.5};packet.hands[side].points[i][0]=float(p.x);packet.hands[side].points[i][1]=float(p.y);packet.hands[side].points[i][2]=float(p.z);}}
    auto write=[&]{std::ofstream f(path,std::ios::binary|std::ios::trunc);f.write(reinterpret_cast<const char*>(&packet),sizeof packet);f.close();provider->RunFrame();};
    auto publish=[&]{++packet.sequence;packet.sent_ms=GetTickCount64();packet.capture_ms=packet.sent_ms-20;write();};
    publish();assert(host.pose[0].deviceIsConnected&&host.pose[1].deviceIsConnected);
    assert(properties.roles[5]==TrackedControllerRole_LeftHand&&properties.roles[6]==TrackedControllerRole_RightHand);
    assert(input.updates[0]==input.updates[1]&&input.updates[0]>=4);assert(host.oldest<=-.019);
    assert(std::abs(host.pose[0].vecPosition[1]-1.6)<.01);
    const float original=packet.hands[0].points[2][0];
    packet.hands[0].points[2][0]=NAN;publish();
    assert(!host.pose[0].deviceIsConnected&&host.pose[1].deviceIsConnected);
    packet.hands[0].points[2][0]=original;publish();
    assert(host.pose[0].deviceIsConnected&&host.pose[1].deviceIsConnected);
    packet.options=fhs::Controls;publish();
    std::memcpy(packet.hands[0].points[4],packet.hands[0].points[8],sizeof packet.hands[0].points[4]);publish();assert(input.pressed(5,"/input/trigger/click"));
    host.head=false;provider->RunFrame();assert(!host.pose[0].deviceIsConnected&&!host.pose[1].deviceIsConnected);
    host.head=true;publish();assert(!input.pressed(5,"/input/trigger/click"));packet.hands[0].flags=0;publish();assert(!host.pose[0].deviceIsConnected&&host.pose[1].deviceIsConnected);
    Sleep(170);provider->RunFrame();assert(!host.pose[1].deviceIsConnected);
    write();assert(!host.pose[1].deviceIsConnected); // replay does not resurrect
    publish();assert(host.pose[1].deviceIsConnected);
    packet.hands[1].points[0][0]=NAN;publish();Sleep(170);provider->RunFrame();assert(!host.pose[1].deviceIsConnected);
    DeleteFileA(path.c_str());for(auto* d:host.device)d->Deactivate();provider->Cleanup();FreeLibrary(library);
    std::puts("PASS actual Windows DLL: two devices, initial disconnect, both skeleton ranges, roles, capture-time head transform, independent hand loss, HMD loss, expiry, replay, invalid input and cleanup");
}

#endif
