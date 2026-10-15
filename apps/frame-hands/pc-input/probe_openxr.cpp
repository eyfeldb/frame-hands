#define NOMINMAX
#define XR_USE_PLATFORM_WIN32
#define XR_USE_GRAPHICS_API_D3D11
#define XR_NO_PROTOTYPES
#include <windows.h>
#include <d3d11.h>
#include <dxgi1_2.h>
#include <openxr/openxr.h>
#include <openxr/openxr_platform.h>
#include <cstdio>
#include <cstring>
#include <vector>
#include <chrono>
#include <thread>
#define CHECK(x) do {auto result=(x);if(XR_FAILED(result)){std::printf("failed=%s result=%d\n",#x,int(result));return 1;}}while(0)
#define LOAD(instance,name) PFN_##name name=nullptr;CHECK(get(instance,#name,reinterpret_cast<PFN_xrVoidFunction*>(&name)))
int main(int argc,char** argv){
    if(argc!=2)return 2;auto library=LoadLibraryA(argv[1]);if(!library)return 3;
    auto symbol=GetProcAddress(library,"xrGetInstanceProcAddr");PFN_xrGetInstanceProcAddr get=nullptr;static_assert(sizeof get==sizeof symbol);std::memcpy(&get,&symbol,sizeof get);if(!get)return 4;
    LOAD(XR_NULL_HANDLE,xrEnumerateInstanceExtensionProperties);LOAD(XR_NULL_HANDLE,xrCreateInstance);
    uint32_t count=0;CHECK(xrEnumerateInstanceExtensionProperties(nullptr,0,&count,nullptr));std::vector<XrExtensionProperties> extensions(count,{XR_TYPE_EXTENSION_PROPERTIES});CHECK(xrEnumerateInstanceExtensionProperties(nullptr,count,&count,extensions.data()));
    auto has=[&](const char* n){for(auto& e:extensions)if(!std::strcmp(n,e.extensionName))return true;return false;};
    bool headless=has("XR_MND_headless"),source=has("XR_EXT_hand_tracking_data_source");
    std::printf("hand_tracking=%d data_source=%d headless=%d\n",has("XR_EXT_hand_tracking"),source,headless);if(!has("XR_EXT_hand_tracking"))return 5;
    std::vector<const char*> enabled{"XR_EXT_hand_tracking"};if(source)enabled.push_back("XR_EXT_hand_tracking_data_source");enabled.push_back(headless?"XR_MND_headless":"XR_KHR_D3D11_enable");
    XrInstanceCreateInfo ci{XR_TYPE_INSTANCE_CREATE_INFO};std::strcpy(ci.applicationInfo.applicationName,"FrameHandsPCProbe");ci.applicationInfo.apiVersion=XR_MAKE_VERSION(1,0,0);ci.enabledExtensionCount=uint32_t(enabled.size());ci.enabledExtensionNames=enabled.data();XrInstance instance;CHECK(xrCreateInstance(&ci,&instance));
    LOAD(instance,xrGetInstanceProperties);LOAD(instance,xrGetSystem);LOAD(instance,xrGetSystemProperties);LOAD(instance,xrCreateSession);LOAD(instance,xrDestroySession);LOAD(instance,xrDestroyInstance);LOAD(instance,xrCreateReferenceSpace);LOAD(instance,xrDestroySpace);LOAD(instance,xrPollEvent);LOAD(instance,xrBeginSession);LOAD(instance,xrEndSession);LOAD(instance,xrWaitFrame);LOAD(instance,xrBeginFrame);LOAD(instance,xrEndFrame);LOAD(instance,xrCreateHandTrackerEXT);LOAD(instance,xrLocateHandJointsEXT);LOAD(instance,xrDestroyHandTrackerEXT);
    XrInstanceProperties ip{XR_TYPE_INSTANCE_PROPERTIES};CHECK(xrGetInstanceProperties(instance,&ip));std::printf("runtime=%s version=%llu\n",ip.runtimeName,static_cast<unsigned long long>(ip.runtimeVersion));
    XrSystemGetInfo gi{XR_TYPE_SYSTEM_GET_INFO};gi.formFactor=XR_FORM_FACTOR_HEAD_MOUNTED_DISPLAY;XrSystemId system;CHECK(xrGetSystem(instance,&gi,&system));
    XrSystemHandTrackingPropertiesEXT handprops{XR_TYPE_SYSTEM_HAND_TRACKING_PROPERTIES_EXT};XrSystemProperties props{XR_TYPE_SYSTEM_PROPERTIES,&handprops};CHECK(xrGetSystemProperties(instance,system,&props));std::printf("system=%s supports_hands=%d\n",props.systemName,handprops.supportsHandTracking);
    ID3D11Device* device=nullptr;ID3D11DeviceContext* context=nullptr;XrGraphicsBindingD3D11KHR binding{XR_TYPE_GRAPHICS_BINDING_D3D11_KHR};
    if(!headless){
        LOAD(instance,xrGetD3D11GraphicsRequirementsKHR);XrGraphicsRequirementsD3D11KHR req{XR_TYPE_GRAPHICS_REQUIREMENTS_D3D11_KHR};CHECK(xrGetD3D11GraphicsRequirementsKHR(instance,system,&req));
        IDXGIFactory1* factory=nullptr;if(FAILED(CreateDXGIFactory1(__uuidof(IDXGIFactory1),reinterpret_cast<void**>(&factory))))return 6;
        IDXGIAdapter1* adapter=nullptr;for(UINT n=0;factory->EnumAdapters1(n,&adapter)!=DXGI_ERROR_NOT_FOUND;++n){DXGI_ADAPTER_DESC1 d{};adapter->GetDesc1(&d);if(d.AdapterLuid.HighPart==req.adapterLuid.HighPart&&d.AdapterLuid.LowPart==req.adapterLuid.LowPart)break;adapter->Release();adapter=nullptr;}
        factory->Release();if(!adapter)return 7;D3D_FEATURE_LEVEL chosen{};D3D_FEATURE_LEVEL levels[]={D3D_FEATURE_LEVEL_11_1,D3D_FEATURE_LEVEL_11_0};
        auto hr=D3D11CreateDevice(adapter,D3D_DRIVER_TYPE_UNKNOWN,nullptr,0,levels,2,D3D11_SDK_VERSION,&device,&chosen,&context);adapter->Release();if(FAILED(hr)||chosen<req.minFeatureLevel)return 8;binding.device=device;
    }
    XrSessionCreateInfo sci{XR_TYPE_SESSION_CREATE_INFO};sci.systemId=system;sci.next=headless?nullptr:&binding;XrSession session;CHECK(xrCreateSession(instance,&sci,&session));
    XrReferenceSpaceCreateInfo ri{XR_TYPE_REFERENCE_SPACE_CREATE_INFO};ri.referenceSpaceType=XR_REFERENCE_SPACE_TYPE_LOCAL;ri.poseInReferenceSpace.orientation.w=1;XrSpace space;CHECK(xrCreateReferenceSpace(session,&ri,&space));
    LOAD(instance,xrStringToPath);LOAD(instance,xrCreateActionSet);LOAD(instance,xrCreateAction);LOAD(instance,xrSuggestInteractionProfileBindings);LOAD(instance,xrAttachSessionActionSets);LOAD(instance,xrSyncActions);LOAD(instance,xrDestroyActionSet);
    XrActionSetCreateInfo ac{XR_TYPE_ACTION_SET_CREATE_INFO};std::strcpy(ac.actionSetName,"hands");std::strcpy(ac.localizedActionSetName,"Hands");XrActionSet actions;CHECK(xrCreateActionSet(instance,&ac,&actions));
    XrPath paths[2]{};CHECK(xrStringToPath(instance,"/user/hand/left",&paths[0]));CHECK(xrStringToPath(instance,"/user/hand/right",&paths[1]));
    XrActionCreateInfo ai{XR_TYPE_ACTION_CREATE_INFO};ai.actionType=XR_ACTION_TYPE_POSE_INPUT;std::strcpy(ai.actionName,"grip");std::strcpy(ai.localizedActionName,"Grip");ai.countSubactionPaths=2;ai.subactionPaths=paths;XrAction action;CHECK(xrCreateAction(actions,&ai,&action));
    XrActionSuggestedBinding suggested[2]{};for(int n=0;n<2;++n){suggested[n].action=action;CHECK(xrStringToPath(instance,n?"/user/hand/right/input/grip/pose":"/user/hand/left/input/grip/pose",&suggested[n].binding));}
    for(const char* profile:{"/interaction_profiles/oculus/touch_controller","/interaction_profiles/valve/index_controller","/interaction_profiles/khr/simple_controller"}){XrInteractionProfileSuggestedBinding b{XR_TYPE_INTERACTION_PROFILE_SUGGESTED_BINDING};CHECK(xrStringToPath(instance,profile,&b.interactionProfile));b.countSuggestedBindings=2;b.suggestedBindings=suggested;CHECK(xrSuggestInteractionProfileBindings(instance,&b));}
    XrSessionActionSetsAttachInfo attach{XR_TYPE_SESSION_ACTION_SETS_ATTACH_INFO};attach.countActionSets=1;attach.actionSets=&actions;CHECK(xrAttachSessionActionSets(session,&attach));
    XrHandTrackerEXT trackers[6]{};
    for(int n=0;n<(source?6:2);++n){XrHandTrackerCreateInfoEXT hc{XR_TYPE_HAND_TRACKER_CREATE_INFO_EXT};hc.hand=n%2?XR_HAND_RIGHT_EXT:XR_HAND_LEFT_EXT;hc.handJointSet=XR_HAND_JOINT_SET_DEFAULT_EXT;
        auto requested=n<4?XR_HAND_TRACKING_DATA_SOURCE_UNOBSTRUCTED_EXT:XR_HAND_TRACKING_DATA_SOURCE_CONTROLLER_EXT;XrHandTrackingDataSourceInfoEXT ds{XR_TYPE_HAND_TRACKING_DATA_SOURCE_INFO_EXT};ds.requestedDataSourceCount=1;ds.requestedDataSources=&requested;if(n>=2)hc.next=&ds;auto r=xrCreateHandTrackerEXT(session,&hc,&trackers[n]);std::printf("create_tracker=%d result=%d\n",n,int(r));}
    bool running=false;unsigned samples=0,active[6]{},tracked[6]{};int reported[6]{};
    auto end=std::chrono::steady_clock::now()+std::chrono::seconds(8);
    while(std::chrono::steady_clock::now()<end){
        XrEventDataBuffer event{XR_TYPE_EVENT_DATA_BUFFER};while(xrPollEvent(instance,&event)==XR_SUCCESS){
            if(event.type==XR_TYPE_EVENT_DATA_SESSION_STATE_CHANGED){auto* e=reinterpret_cast<XrEventDataSessionStateChanged*>(&event);std::printf("session_state=%d\n",int(e->state));if(e->state==XR_SESSION_STATE_READY){XrSessionBeginInfo bi{XR_TYPE_SESSION_BEGIN_INFO};bi.primaryViewConfigurationType=XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO;CHECK(xrBeginSession(session,&bi));running=true;}if(e->state==XR_SESSION_STATE_STOPPING){CHECK(xrEndSession(session));running=false;}}
            event={XR_TYPE_EVENT_DATA_BUFFER};}
        if(running){XrActiveActionSet activeSet{actions,XR_NULL_PATH};XrActionsSyncInfo sync{XR_TYPE_ACTIONS_SYNC_INFO};sync.countActiveActionSets=1;sync.activeActionSets=&activeSet;CHECK(xrSyncActions(session,&sync));XrFrameState state{XR_TYPE_FRAME_STATE};XrFrameWaitInfo wi{XR_TYPE_FRAME_WAIT_INFO};CHECK(xrWaitFrame(session,&wi,&state));XrFrameBeginInfo bi{XR_TYPE_FRAME_BEGIN_INFO};CHECK(xrBeginFrame(session,&bi));++samples;
            for(int n=0;n<6;++n)if(trackers[n]){XrHandJointLocationEXT joints[26]{};XrHandTrackingDataSourceStateEXT ds{XR_TYPE_HAND_TRACKING_DATA_SOURCE_STATE_EXT};XrHandJointLocationsEXT loc{XR_TYPE_HAND_JOINT_LOCATIONS_EXT};loc.next=source?&ds:nullptr;loc.jointCount=26;loc.jointLocations=joints;XrHandJointsLocateInfoEXT li{XR_TYPE_HAND_JOINTS_LOCATE_INFO_EXT};li.baseSpace=space;li.time=state.predictedDisplayTime;auto r=xrLocateHandJointsEXT(trackers[n],&li,&loc);if(XR_SUCCEEDED(r)&&loc.isActive){++active[n];if(joints[XR_HAND_JOINT_INDEX_TIP_EXT].locationFlags&XR_SPACE_LOCATION_POSITION_TRACKED_BIT)++tracked[n];reported[n]=ds.isActive?int(ds.dataSource):0;}}
            XrFrameEndInfo ei{XR_TYPE_FRAME_END_INFO};ei.displayTime=state.predictedDisplayTime;ei.environmentBlendMode=XR_ENVIRONMENT_BLEND_MODE_OPAQUE;CHECK(xrEndFrame(session,&ei));
        }
        if(headless||!running)std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    for(int n=0;n<6;++n){std::printf("tracker=%d samples=%u active=%u tracked_index=%u source=%d\n",n,samples,active[n],tracked[n],reported[n]);if(trackers[n])xrDestroyHandTrackerEXT(trackers[n]);}
    xrDestroySpace(space);xrDestroySession(session);xrDestroyActionSet(actions);xrDestroyInstance(instance);if(context)context->Release();if(device)device->Release();FreeLibrary(library);return 0;
}
