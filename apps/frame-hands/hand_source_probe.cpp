#define XR_USE_TIMESPEC
#include <time.h>
#include <openxr/openxr.h>
#include <openxr/openxr_platform.h>
#include <cstdio>
#include <cstring>
#include <chrono>
#include <thread>

#define CHECK(expr) do {XrResult r=(expr);if(XR_FAILED(r)){std::printf("failed=%s result=%d\n",#expr,r);return 1;}}while(0)
int main() {
 const char* exts[]={"XR_EXT_hand_tracking","XR_EXT_hand_tracking_data_source","XR_MND_headless","XR_KHR_convert_timespec_time"};
 XrInstanceCreateInfo ci{XR_TYPE_INSTANCE_CREATE_INFO};ci.enabledExtensionCount=4;ci.enabledExtensionNames=exts;
 std::strcpy(ci.applicationInfo.applicationName,"FrameHandsSourceProbe");ci.applicationInfo.apiVersion=XR_MAKE_VERSION(1,0,0);
 XrInstance i;CHECK(xrCreateInstance(&ci,&i));
 XrSystemGetInfo gi{XR_TYPE_SYSTEM_GET_INFO};gi.formFactor=XR_FORM_FACTOR_HEAD_MOUNTED_DISPLAY;
 XrSystemId sid;CHECK(xrGetSystem(i,&gi,&sid));
 XrSessionCreateInfo sc{XR_TYPE_SESSION_CREATE_INFO};sc.systemId=sid;
 XrSession s;CHECK(xrCreateSession(i,&sc,&s));
 PFN_xrCreateHandTrackerEXT create;PFN_xrLocateHandJointsEXT locate;PFN_xrDestroyHandTrackerEXT destroy;
 PFN_xrConvertTimespecTimeToTimeKHR convert;
 CHECK(xrGetInstanceProcAddr(i,"xrCreateHandTrackerEXT",reinterpret_cast<PFN_xrVoidFunction*>(&create)));
 CHECK(xrGetInstanceProcAddr(i,"xrLocateHandJointsEXT",reinterpret_cast<PFN_xrVoidFunction*>(&locate)));
 CHECK(xrGetInstanceProcAddr(i,"xrDestroyHandTrackerEXT",reinterpret_cast<PFN_xrVoidFunction*>(&destroy)));
 CHECK(xrGetInstanceProcAddr(i,"xrConvertTimespecTimeToTimeKHR",reinterpret_cast<PFN_xrVoidFunction*>(&convert)));
 XrReferenceSpaceCreateInfo ri{XR_TYPE_REFERENCE_SPACE_CREATE_INFO};ri.referenceSpaceType=XR_REFERENCE_SPACE_TYPE_LOCAL;ri.poseInReferenceSpace.orientation.w=1;
 XrSpace space;CHECK(xrCreateReferenceSpace(s,&ri,&space));
 XrHandTrackerEXT trackers[4]{};
 for(int n=0;n<4;++n) {
   auto source=n<2?XR_HAND_TRACKING_DATA_SOURCE_UNOBSTRUCTED_EXT:XR_HAND_TRACKING_DATA_SOURCE_CONTROLLER_EXT;
   XrHandTrackingDataSourceInfoEXT ds{XR_TYPE_HAND_TRACKING_DATA_SOURCE_INFO_EXT};ds.requestedDataSourceCount=1;ds.requestedDataSources=&source;
   XrHandTrackerCreateInfoEXT hc{XR_TYPE_HAND_TRACKER_CREATE_INFO_EXT};hc.next=&ds;hc.hand=n%2?XR_HAND_RIGHT_EXT:XR_HAND_LEFT_EXT;hc.handJointSet=XR_HAND_JOINT_SET_DEFAULT_EXT;
   XrResult r=create(s,&hc,&trackers[n]);std::printf("create source=%d hand=%d result=%d\n",source,hc.hand,r);
 }
 bool running=false;unsigned active[4]{},tracked[4]{},samples=0;int sources[4]{};XrResult results[4]{};
 auto until=std::chrono::steady_clock::now()+std::chrono::seconds(5);
 while(std::chrono::steady_clock::now()<until) {
   XrEventDataBuffer ev{XR_TYPE_EVENT_DATA_BUFFER};
   while(xrPollEvent(i,&ev)==XR_SUCCESS) {
     if(ev.type==XR_TYPE_EVENT_DATA_SESSION_STATE_CHANGED){auto* state=reinterpret_cast<XrEventDataSessionStateChanged*>(&ev);
       std::printf("state=%d\n",state->state);
       if(state->state==XR_SESSION_STATE_READY){XrSessionBeginInfo bi{XR_TYPE_SESSION_BEGIN_INFO};bi.primaryViewConfigurationType=XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO;CHECK(xrBeginSession(s,&bi));running=true;}
       if(state->state==XR_SESSION_STATE_STOPPING){xrEndSession(s);running=false;}
     }ev={XR_TYPE_EVENT_DATA_BUFFER};
   }
   if(running){timespec ts;clock_gettime(CLOCK_MONOTONIC,&ts);XrTime time;CHECK(convert(i,&ts,&time));++samples;
     for(int n=0;n<4;++n)if(trackers[n]){
       XrHandJointsLocateInfoEXT li{XR_TYPE_HAND_JOINTS_LOCATE_INFO_EXT};li.baseSpace=space;li.time=time;
       XrHandJointLocationEXT joints[XR_HAND_JOINT_COUNT_EXT]{};
       XrHandTrackingDataSourceStateEXT source{XR_TYPE_HAND_TRACKING_DATA_SOURCE_STATE_EXT};
       XrHandJointLocationsEXT locations{XR_TYPE_HAND_JOINT_LOCATIONS_EXT};locations.next=&source;locations.jointCount=XR_HAND_JOINT_COUNT_EXT;locations.jointLocations=joints;
       results[n]=locate(trackers[n],&li,&locations);
       if(XR_SUCCEEDED(results[n]) && locations.isActive){++active[n];sources[n]=source.isActive?source.dataSource:0;
         if(joints[XR_HAND_JOINT_INDEX_TIP_EXT].locationFlags&XR_SPACE_LOCATION_POSITION_TRACKED_BIT)++tracked[n];}
     }
   }
   std::this_thread::sleep_for(std::chrono::milliseconds(10));
 }
 for(int n=0;n<4;++n){std::printf("tracker=%d samples=%u active=%u tracked_index_tip=%u reported_source=%d result=%d\n",n,samples,active[n],tracked[n],sources[n],results[n]);if(trackers[n])destroy(trackers[n]);}
 xrDestroySpace(space);xrDestroySession(s);xrDestroyInstance(i);return 0;
}
