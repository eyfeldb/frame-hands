#define XR_NO_PROTOTYPES
#include <dlfcn.h>
// Bounded public API probe. No camera images, controller input, or settings saved.
#include <openxr/openxr.h>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>

int main() {
    void* lib=dlopen("libopenxr_loader.so",RTLD_NOW|RTLD_LOCAL);
    if(!lib){std::fprintf(stderr,"loader: %s\n",dlerror());return 2;}
    auto get=reinterpret_cast<PFN_xrGetInstanceProcAddr>(dlsym(lib,"xrGetInstanceProcAddr"));
    if(!get)return 3;
    PFN_xrEnumerateInstanceExtensionProperties xrEnumerateInstanceExtensionProperties=nullptr;
    PFN_xrCreateInstance xrCreateInstance=nullptr;
    if(XR_FAILED(get(XR_NULL_HANDLE,"xrEnumerateInstanceExtensionProperties",reinterpret_cast<PFN_xrVoidFunction*>(&xrEnumerateInstanceExtensionProperties))) || XR_FAILED(get(XR_NULL_HANDLE,"xrCreateInstance",reinterpret_cast<PFN_xrVoidFunction*>(&xrCreateInstance))))return 4;

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
            PFN_xrGetInstanceProperties xrGetInstanceProperties=nullptr;
            PFN_xrGetSystem xrGetSystem=nullptr;
            PFN_xrGetSystemProperties xrGetSystemProperties=nullptr;
            PFN_xrDestroyInstance xrDestroyInstance=nullptr;
#define LOAD(name) if(XR_FAILED(get(instance,#name,reinterpret_cast<PFN_xrVoidFunction*>(&name))))return 5
            LOAD(xrGetInstanceProperties);LOAD(xrGetSystem);LOAD(xrGetSystemProperties);LOAD(xrDestroyInstance);
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
    return XR_FAILED(r)?1:0;
}
