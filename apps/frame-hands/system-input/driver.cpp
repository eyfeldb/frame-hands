// Frame Hands system pointer. Architecture/resources adapted from MIT Frametop;
// see vendor/LICENSE-Frametop. Atomic state protocol and expiry are local additions.
#include <openvr_driver.h>
#include "protocol.h"
#include <atomic>
#include <mutex>
#include <thread>
#include <string>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

using namespace vr;
namespace {
class Device final: public ITrackedDeviceServerDriver {
public:
    fh::Mailbox mailbox;std::mutex lock;
    EVRInitError Activate(uint32_t index)override {
        index_=index;container_=VRProperties()->TrackedDeviceToPropertyContainer(index);
        auto p=VRProperties();
        p->SetStringProperty(container_,Prop_ModelNumber_String,"Frame Hands Pointer v1");
        p->SetStringProperty(container_,Prop_ManufacturerName_String,"Frame Hands experiment");
        p->SetStringProperty(container_,Prop_ControllerType_String,"fh_pointer");
        p->SetStringProperty(container_,Prop_InputProfilePath_String,"{fh_pointer}/input/fh_pointer_profile.json");
        p->SetStringProperty(container_,Prop_RenderModelName_String,"{fh_pointer}/rendermodels/ft_pointer_invisible");
        p->SetInt32Property(container_,Prop_ControllerRoleHint_Int32,TrackedControllerRole_OptOut);
        p->SetInt32Property(container_,Prop_DeviceClass_Int32,TrackedDeviceClass_Controller);
        p->SetBoolProperty(container_,Prop_NeverTracked_Bool,false);
        const char* names[]={"trigger","b","x","system","joystick","a"};
        for(int i=0;i<6;++i)if(VRDriverInput()->CreateBooleanComponent(container_,(std::string("/input/")+names[i]+"/click").c_str(),&buttons_[i])!=VRInputError_None)return VRInitError_Driver_Failed;
        for(int i=0;i<2;++i)if(VRDriverInput()->CreateScalarComponent(container_,i?"/input/joystick/y":"/input/joystick/x",&scroll_[i],VRScalarType_Absolute,VRScalarUnits_NormalizedTwoSided)!=VRInputError_None)return VRInitError_Driver_Failed;
        VRDriverLog()->Log("fh_pointer v1 activated disconnected; requires fresh bridge packets");
        return VRInitError_None;
    }
    void Deactivate()override{index_=k_unTrackedDeviceIndexInvalid;}
    void EnterStandby()override{}
    void* GetComponent(const char*)override{return nullptr;}
    void DebugRequest(const char*,char* result,uint32_t size)override{if(size)result[0]=0;}
    DriverPose_t GetPose()override{std::lock_guard<std::mutex> guard(lock);return pose_;}
    void frame() {
        if(index_==k_unTrackedDeviceIndexInvalid)return;
        TrackedDevicePose_t head{};VRServerDriverHost()->GetRawTrackedDevicePoses(0,&head,1);
        fh::Packet p;
        {std::lock_guard<std::mutex> guard(lock);p=mailbox.read(fh::now_ns(),head.bPoseIsValid&&head.bDeviceIsConnected);}
        const bool visible=p.flags&fh::Visible;
        // Release buttons before disconnecting, including stale bridge and invalid HMD.
        for(int i=0;i<6;++i)VRDriverInput()->UpdateBooleanComponent(buttons_[i],visible&&((i==0&&(p.flags&fh::Click))||(i==5&&(p.flags&fh::Claim))),0);
        for(auto component:scroll_)VRDriverInput()->UpdateScalarComponent(component,0,0);
        const auto role=visible?TrackedControllerRole_RightHand:TrackedControllerRole_OptOut;
        if(role!=role_){VRProperties()->SetInt32Property(container_,Prop_ControllerRoleHint_Int32,role);role_=role;}
        DriverPose_t pose{};pose.qWorldFromDriverRotation.w=1;pose.qDriverFromHeadRotation.w=1;
        pose.qRotation={p.quaternion[0],p.quaternion[1],p.quaternion[2],p.quaternion[3]};
        for(int i=0;i<3;++i)pose.vecPosition[i]=p.position[i];
        pose.deviceIsConnected=visible;pose.poseIsValid=visible;
        pose.result=visible?TrackingResult_Running_OK:TrackingResult_Uninitialized;
        {std::lock_guard<std::mutex> guard(lock);pose_=pose;}
        VRServerDriverHost()->TrackedDevicePoseUpdated(index_,pose,sizeof pose);
    }
private:
    uint32_t index_=k_unTrackedDeviceIndexInvalid;PropertyContainerHandle_t container_=0;
    VRInputComponentHandle_t buttons_[6]{},scroll_[2]{};
    ETrackedControllerRole role_=TrackedControllerRole_OptOut;DriverPose_t pose_{};
};
class Provider final:public IServerTrackedDeviceProvider {
public:
    EVRInitError Init(IVRDriverContext* context)override {
        VR_INIT_SERVER_DRIVER_CONTEXT(context);
        sock_=socket(AF_UNIX,SOCK_DGRAM|SOCK_CLOEXEC,0);
        if(sock_<0)return VRInitError_Driver_Failed;
        sockaddr_un addr{};addr.sun_family=AF_UNIX;
        const std::string name="fh_pointer_"+std::to_string(getuid());
        std::memcpy(addr.sun_path+1,name.data(),name.size());
        int yes=1;timeval timeout{0,50000};
        if(setsockopt(sock_,SOL_SOCKET,SO_PASSCRED,&yes,sizeof yes)||setsockopt(sock_,SOL_SOCKET,SO_RCVTIMEO,&timeout,sizeof timeout)||
           bind(sock_,reinterpret_cast<sockaddr*>(&addr),offsetof(sockaddr_un,sun_path)+1+name.size())){close(sock_);sock_=-1;return VRInitError_Driver_Failed;}
        if(!VRServerDriverHost()->TrackedDeviceAdded("frame-hands-pointer-v1",TrackedDeviceClass_Controller,&device_)){close(sock_);sock_=-1;return VRInitError_Driver_Failed;}
        running_=true;worker_=std::thread([this]{listen();});return VRInitError_None;
    }
    void Cleanup()override{running_=false;if(worker_.joinable())worker_.join();if(sock_>=0)close(sock_);sock_=-1;VR_CLEANUP_SERVER_DRIVER_CONTEXT();}
    const char* const* GetInterfaceVersions()override{return k_InterfaceVersions;}
    void RunFrame()override{device_.frame();}
    bool ShouldBlockStandbyMode()override{return false;}
    void EnterStandby()override{}
    void LeaveStandby()override{}
private:
    void listen() {
        while(running_) {
            fh::Packet packet{};iovec io{&packet,sizeof packet};
            alignas(cmsghdr) char control[CMSG_SPACE(sizeof(ucred))]{};
            msghdr message{};message.msg_iov=&io;message.msg_iovlen=1;message.msg_control=control;message.msg_controllen=sizeof control;
            ssize_t n=recvmsg(sock_,&message,0);
            if(n!=sizeof packet || (message.msg_flags&(MSG_TRUNC|MSG_CTRUNC)))continue;
            bool own=false;
            for(auto c=CMSG_FIRSTHDR(&message);c;c=CMSG_NXTHDR(&message,c))if(c->cmsg_level==SOL_SOCKET&&c->cmsg_type==SCM_CREDENTIALS&&c->cmsg_len>=CMSG_LEN(sizeof(ucred))){ucred credential{};std::memcpy(&credential,CMSG_DATA(c),sizeof credential);own=credential.uid==getuid();}
            if(own){std::lock_guard<std::mutex> guard(device_.lock);device_.mailbox.accept(packet,fh::now_ns());}
        }
    }
    Device device_;int sock_=-1;std::atomic<bool> running_{false};std::thread worker_;
};
Provider provider;
}
extern "C" __attribute__((visibility("default"))) void* HmdDriverFactory(const char* name,int* error) {
    if(name&&std::strcmp(name,IServerTrackedDeviceProvider_Version)==0)return &provider;
    if(error)*error=VRInitError_Init_InterfaceNotFound;
    return nullptr;
}
