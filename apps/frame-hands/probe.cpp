// Bounded public API probe. No camera images, controller input, or settings saved.
#include "openvr.h"
#include <openxr/openxr.h>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>

int main() {
    uint32_t count=0;
    XrResult r=xrEnumerateInstanceExtensionProperties(nullptr,0,&count,nullptr);
    std::printf("openxr_enumerate=%d count=%u\n",r,count);
    if(XR_SUCCEEDED(r)) {
        std::vector<XrExtensionProperties> ext(count,{XR_TYPE_EXTENSION_PROPERTIES});
        r=xrEnumerateInstanceExtensionProperties(nullptr,count,&count,ext.data());
        if(XR_SUCCEEDED(r)) for(auto &e:ext)
            if(std::strstr(e.extensionName,"hand")||std::strstr(e.extensionName,"camera")||std::strstr(e.extensionName,"headless"))
                std::printf("extension=%s version=%u\n",e.extensionName,e.extensionVersion);
        const char* enabled="XR_EXT_hand_tracking";
        bool has=false;for(auto &e:ext)has|=!std::strcmp(e.extensionName,enabled);
        XrInstanceCreateInfo ci{XR_TYPE_INSTANCE_CREATE_INFO};
        std::strcpy(ci.applicationInfo.applicationName,"FrameHandsCapabilityProbe");
        ci.applicationInfo.apiVersion=XR_MAKE_VERSION(1,0,0);
        ci.enabledExtensionCount=has?1:0;ci.enabledExtensionNames=has?&enabled:nullptr;
        XrInstance instance=XR_NULL_HANDLE;r=xrCreateInstance(&ci,&instance);
        std::printf("openxr_create=%d\n",r);
        if(XR_SUCCEEDED(r)) {
            XrInstanceProperties ip{XR_TYPE_INSTANCE_PROPERTIES};xrGetInstanceProperties(instance,&ip);
            std::printf("runtime=%s version=%u.%u.%u\n",ip.runtimeName,
                XR_VERSION_MAJOR(ip.runtimeVersion),XR_VERSION_MINOR(ip.runtimeVersion),XR_VERSION_PATCH(ip.runtimeVersion));
            XrSystemGetInfo si{XR_TYPE_SYSTEM_GET_INFO};si.formFactor=XR_FORM_FACTOR_HEAD_MOUNTED_DISPLAY;
            XrSystemId sid=0;r=xrGetSystem(instance,&si,&sid);
            std::printf("openxr_system=%d\n",r);
            if(XR_SUCCEEDED(r)) {
                XrSystemHandTrackingPropertiesEXT hp{XR_TYPE_SYSTEM_HAND_TRACKING_PROPERTIES_EXT};
                XrSystemProperties sp{XR_TYPE_SYSTEM_PROPERTIES};sp.next=has?&hp:nullptr;
                r=xrGetSystemProperties(instance,sid,&sp);
                std::printf("system_properties=%d supportsHandTracking=%u (does not prove optical tracking)\n",r,hp.supportsHandTracking);
            }
            xrDestroyInstance(instance);
        }
    }
    vr::EVRInitError error=vr::VRInitError_None;
    auto sys=vr::VR_Init(&error,vr::VRApplication_Background);
    std::printf("openvr_init=%d\n",int(error));
    if(error!=vr::VRInitError_None)return 1;
    auto cam=vr::VRTrackedCamera();
    if(!cam){std::puts("camera_interface=null");vr::VR_Shutdown();return 1;}
    bool present=false;auto ce=cam->HasCamera(0,&present);
    std::printf("has_camera_error=%d has_camera=%d\n",int(ce),present);
    vr::TrackedCameraHandle_t preliminary=0;
    ce=cam->AcquireVideoStreamingService(0,&preliminary);
    std::printf("preliminary_acquire_error=%d\n",int(ce));
    for(int type=0;type<3;++type) {
        uint32_t tw=0,th=0,tb=0;
        auto te=cam->GetCameraFrameSize(0,static_cast<vr::EVRTrackedCameraFrameType>(type),&tw,&th,&tb);
        std::printf("camera_type=%d error=%d width=%u height=%u bytes=%u\n",type,int(te),tw,th,tb);
    }
    uint32_t w=0,h=0,bytes=0;
    ce=cam->GetCameraFrameSize(0,vr::VRTrackedCameraFrameType_Distorted,&w,&h,&bytes);
    std::printf("camera_size_error=%d width=%u height=%u bytes=%u\n",int(ce),w,h,bytes);
    if(present && ce==vr::VRTrackedCameraError_None && bytes>0 && bytes<128*1024*1024) {
        vr::TrackedCameraHandle_t handle=0;ce=cam->AcquireVideoStreamingService(0,&handle);
        std::printf("camera_acquire_error=%d\n",int(ce));
        if(ce==vr::VRTrackedCameraError_None) {
            std::vector<unsigned char> buffer(bytes);uint32_t last=~0u,frames=0;
            auto until=std::chrono::steady_clock::now()+std::chrono::seconds(3);
            while(std::chrono::steady_clock::now()<until) {
                vr::CameraVideoStreamFrameHeader_t header{};
                ce=cam->GetVideoStreamFrameBuffer(handle,vr::VRTrackedCameraFrameType_Distorted,buffer.data(),bytes,&header,sizeof(header));
                if(ce==vr::VRTrackedCameraError_None && header.nFrameSequence!=last){++frames;last=header.nFrameSequence;}
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
            }
            std::printf("camera_frames_3s=%u last_error=%d\n",frames,int(ce));
            cam->ReleaseVideoStreamingService(handle);
        }
    }
    if(preliminary)cam->ReleaseVideoStreamingService(preliminary);
    (void)sys;vr::VR_Shutdown();return 0;
}
