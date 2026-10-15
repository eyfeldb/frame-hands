// Load the actual .so against an in-process fake OpenVR host. No libopenvr_api,
// VR_Init, real runtime, or controller input. Only bind() is redirected to a
// unique test socket, so even a concurrently loaded real driver is untouched.
#include <openvr_driver.h>
#include "protocol.h"
#include <cassert>
#include <array>
#include <chrono>
#include <cstdio>
#include <dlfcn.h>
#include <map>
#include <string>
#include <thread>
#include <sys/socket.h>
#include <sys/syscall.h>
#include <sys/un.h>
#include <unistd.h>
using namespace vr;
static const std::string test_socket="fh_pointer_test_"+std::to_string(getpid());
static bool redirected=false;
extern "C" int bind(int fd,const sockaddr* address,socklen_t length)noexcept {
    assert(address->sa_family==AF_UNIX);
    auto original=reinterpret_cast<const sockaddr_un*>(address);
    const auto expected="fh_pointer_"+std::to_string(getuid());
    assert(length==offsetof(sockaddr_un,sun_path)+1+expected.size());
    assert(!original->sun_path[0]&&!std::memcmp(original->sun_path+1,expected.data(),expected.size()));
    sockaddr_un local{};local.sun_family=AF_UNIX;std::memcpy(local.sun_path+1,test_socket.data(),test_socket.size());
    redirected=true;
    return syscall(SYS_bind,fd,&local,offsetof(sockaddr_un,sun_path)+1+test_socket.size());
}
struct Input:IVRDriverInput {
    std::map<std::string,uint64_t> names;std::map<uint64_t,bool> states;uint64_t next=1;
    EVRInputError CreateBooleanComponent(PropertyContainerHandle_t,const char* n,VRInputComponentHandle_t* h)override{*h=next++;names[n]=*h;states[*h]=false;return VRInputError_None;}
    EVRInputError UpdateBooleanComponent(VRInputComponentHandle_t h,bool v,double)override{assert(states.count(h));states[h]=v;return VRInputError_None;}
    EVRInputError CreateScalarComponent(PropertyContainerHandle_t,const char*,VRInputComponentHandle_t* h,EVRScalarType,EVRScalarUnits)override{*h=next++;return VRInputError_None;}
    EVRInputError UpdateScalarComponent(VRInputComponentHandle_t,float v,double)override{assert(v==0);return VRInputError_None;}
    EVRInputError CreateHapticComponent(PropertyContainerHandle_t,const char*,VRInputComponentHandle_t*)override{assert(false);return VRInputError_InvalidParam;}
    EVRInputError CreateSkeletonComponent(PropertyContainerHandle_t,const char*,const char*,const char*,EVRSkeletalTrackingLevel,const VRBoneTransform_t*,uint32_t,VRInputComponentHandle_t*)override{assert(false);return VRInputError_InvalidParam;}
    EVRInputError UpdateSkeletonComponent(VRInputComponentHandle_t,EVRSkeletalMotionRange,const VRBoneTransform_t*,uint32_t)override{assert(false);return VRInputError_InvalidParam;}
    EVRInputError CreatePoseComponent(PropertyContainerHandle_t,const char*,VRInputComponentHandle_t*)override{assert(false);return VRInputError_InvalidParam;}
    EVRInputError UpdatePoseComponent(VRInputComponentHandle_t,const HmdMatrix34_t*,double)override{assert(false);return VRInputError_InvalidParam;}
    EVRInputError CreateEyeTrackingComponent(PropertyContainerHandle_t,const char*,VRInputComponentHandle_t*)override{assert(false);return VRInputError_InvalidParam;}
    EVRInputError UpdateEyeTrackingComponent(VRInputComponentHandle_t,const VREyeTrackingData_t*,double)override{assert(false);return VRInputError_InvalidParam;}
    bool trigger(){return states.at(names.at("/input/trigger/click"));}
    bool any(){for(auto pair:states)if(pair.second)return true;return false;}
} input;
struct Properties:IVRProperties {
    int role=-1;
    ETrackedPropertyError ReadPropertyBatch(PropertyContainerHandle_t,PropertyRead_t*,uint32_t)override{assert(false);return TrackedProp_UnknownProperty;}
    ETrackedPropertyError WritePropertyBatch(PropertyContainerHandle_t,PropertyWrite_t* batch,uint32_t n)override{
        for(uint32_t i=0;i<n;++i){auto& p=batch[i];if(p.prop==Prop_ControllerRoleHint_Int32){assert(p.unBufferSize==sizeof role);std::memcpy(&role,p.pvBuffer,sizeof role);}p.eError=TrackedProp_Success;}return TrackedProp_Success;
    }
    const char* GetPropErrorNameFromEnum(ETrackedPropertyError)override{return "test";}
    PropertyContainerHandle_t TrackedDeviceToPropertyContainer(TrackedDeviceIndex_t)override{return 123;}
} properties;
struct Host:IVRServerDriverHost {
    ITrackedDeviceServerDriver* device=nullptr;DriverPose_t pose{};bool head=true;uint64_t publications=0;
    bool TrackedDeviceAdded(const char*,ETrackedDeviceClass kind,ITrackedDeviceServerDriver* d)override{assert(kind==TrackedDeviceClass_Controller);device=d;return true;}
    void TrackedDevicePoseUpdated(uint32_t index,const DriverPose_t& p,uint32_t size)override{
        assert(index==4&&size==sizeof p);if(!p.deviceIsConnected)assert(!input.any());pose=p;++publications;
    }
    void VsyncEvent(double)override{assert(false);}
    void VendorSpecificEvent(uint32_t,EVREventType,const VREvent_Data_t&,double)override{assert(false);}
    bool IsExiting()override{return false;}
    bool PollNextEvent(VREvent_t*,uint32_t)override{return false;}
    void GetRawTrackedDevicePoses(float,TrackedDevicePose_t* p,uint32_t count)override{
        assert(count==1);*p={};p->bDeviceIsConnected=true;p->bPoseIsValid=head;p->mDeviceToAbsoluteTracking.m[0][0]=p->mDeviceToAbsoluteTracking.m[1][1]=p->mDeviceToAbsoluteTracking.m[2][2]=1;
    }
    void RequestRestart(const char*,const char*,const char*,const char*)override{assert(false);}
    uint32_t GetFrameTimings(Compositor_FrameTiming*,uint32_t)override{return 0;}
    void SetDisplayEyeToHead(uint32_t,const HmdMatrix34_t&,const HmdMatrix34_t&)override{assert(false);}
    void SetDisplayProjectionRaw(uint32_t,const HmdRect2_t&,const HmdRect2_t&)override{assert(false);}
    void SetRecommendedRenderTargetSize(uint32_t,uint32_t,uint32_t)override{assert(false);}
} host;
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
struct Logger:IVRDriverLog{void Log(const char* text)override{std::printf("driver: %s\n",text);}} logger;
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
        if(!std::strcmp(name,IVRSettings_Version))return &settings;
        if(!std::strcmp(name,IVRDriverLog_Version))return &logger;
        if(!std::strcmp(name,IVRResources_Version))return &resources;
        if(!std::strcmp(name,IVRDriverManager_Version))return &manager;
        if(error)*error=VRInitError_Init_InterfaceNotFound;
        return nullptr;
    }
    DriverHandle_t GetDriverHandle()override{return 1;}
} context;
int main(int argc,char** argv){
    assert(argc==2);void* library=dlopen(argv[1],RTLD_NOW|RTLD_LOCAL);if(!library){std::puts(dlerror());return 1;}
    auto factory=reinterpret_cast<void*(*)(const char*,int*)>(dlsym(library,"HmdDriverFactory"));assert(factory);
    int error=0;auto provider=static_cast<IServerTrackedDeviceProvider*>(factory(IServerTrackedDeviceProvider_Version,&error));assert(provider);
    assert(provider->Init(&context)==VRInitError_None&&redirected&&host.device);
    assert(host.device->Activate(4)==VRInitError_None);
    provider->RunFrame();assert(!host.pose.deviceIsConnected&&!input.any()&&properties.role==TrackedControllerRole_OptOut);
    int socket_fd=socket(AF_UNIX,SOCK_DGRAM|SOCK_CLOEXEC,0);assert(socket_fd>=0);
    sockaddr_un address{};address.sun_family=AF_UNIX;std::memcpy(address.sun_path+1,test_socket.data(),test_socket.size());
    auto send=[&](const void* data,size_t bytes){assert(sendto(socket_fd,data,bytes,0,reinterpret_cast<sockaddr*>(&address),offsetof(sockaddr_un,sun_path)+1+test_socket.size())==ssize_t(bytes));};
    auto frame=[&]{std::this_thread::sleep_for(std::chrono::milliseconds(20));provider->RunFrame();};
    fh::Packet packet;uint64_t seq=0;
    auto publish=[&](uint32_t flags){packet.flags=flags;packet.sequence=++seq;packet.sent_ns=fh::now_ns();send(&packet,sizeof packet);frame();};
    publish(fh::Visible|fh::Click);assert(host.pose.deviceIsConnected&&!input.trigger()); // closed hand at startup
    publish(fh::Visible);assert(properties.role==TrackedControllerRole_RightHand);
    publish(fh::Visible|fh::Click);assert(input.trigger()&&host.pose.poseIsValid);
    host.head=false;provider->RunFrame();assert(!input.any()&&!host.pose.deviceIsConnected&&properties.role==TrackedControllerRole_OptOut);
    host.head=true;publish(fh::Visible|fh::Click);assert(!input.trigger()); // must see a release after loss
    publish(fh::Visible);publish(fh::Visible|fh::Click);assert(input.trigger());
    auto stale=packet;stale.sequence=++seq;stale.flags=0;stale.sent_ns-=fh::expiry_ns+1;send(&stale,sizeof stale);frame();assert(input.trigger());
    auto invalid=packet;invalid.sequence=++seq;invalid.flags=0;invalid.position[0]=NAN;send(&invalid,sizeof invalid);frame();assert(input.trigger());
    publish(fh::Visible|fh::Click); // refresh before malformed-length cases
    send(&packet,sizeof packet-1);frame();assert(input.trigger());
    std::array<unsigned char,sizeof(fh::Packet)+8> oversized{};std::memcpy(oversized.data(),&packet,sizeof packet);send(oversized.data(),oversized.size());frame();assert(input.trigger());
    std::this_thread::sleep_for(std::chrono::milliseconds(170));provider->RunFrame();
    assert(!input.any()&&!host.pose.deviceIsConnected&&properties.role==TrackedControllerRole_OptOut);
    publish(fh::Visible|fh::Click);assert(!input.trigger());
    publish(fh::Visible);packet.position[0]=.31;packet.position[1]=1.2;packet.position[2]=-.7;publish(fh::Visible|fh::Click);
    assert(input.trigger()&&host.pose.vecPosition[0]==.31&&host.pose.vecPosition[1]==1.2&&host.pose.vecPosition[2]==-.7);
    publish(0);assert(!input.any()&&!host.pose.deviceIsConnected);
    auto before=host.publications;host.device->Deactivate();provider->RunFrame();assert(host.publications==before);
    auto start=fh::now_ns();provider->Cleanup();assert(fh::now_ns()-start<500000000);
    close(socket_fd);dlclose(library);
    std::puts("PASS actual driver init/activate, isolated credentialed datagrams, startup closed-hand suppression, pose publication, release-before-disconnect, HMD loss/rearm, malformed/stale rejection, dead-bridge expiry, role return and clean shutdown; no real runtime initialized");
}
