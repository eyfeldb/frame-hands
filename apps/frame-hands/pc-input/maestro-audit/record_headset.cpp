#include <fstream>
#include <cstring>
#include <cstdint>
#include <fcntl.h>
#include <sys/file.h>
#include <vulkan/vulkan.h>
#include <sstream>
#include <string>
#include <openvr.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <chrono>
#include <thread>
#include <csignal>
#include <algorithm>
#include <stdexcept>
#include <unistd.h>
static void require(bool ok,const char* stage) {
    if(!ok) throw std::runtime_error(stage);
}
static void checked(VkResult result,const char* stage) {
    if(result!=VK_SUCCESS)
        throw std::runtime_error(std::string(stage)+":"+std::to_string(int(result)));
}
#include "gpu_readback.h"
static std::vector<std::string> extension_names(const std::vector<char>& bytes) {
    if(bytes.empty()) return {};
    require(bytes.back()==0,"extensions_not_terminated");
    std::istringstream input(bytes.data());std::string name;
    std::vector<std::string> names;
    while(input>>name) names.push_back(name);
    return names;
}
static std::vector<const char*> pointers(const std::vector<std::string>& names) {
    std::vector<const char*> result;
    for(const auto& name:names) result.push_back(name.c_str());
    return result;
}

static volatile std::sig_atomic_t running=1;
static void stop(int){running=0;}
using Clock=std::chrono::steady_clock;
static void state(const char* value){std::ofstream("state.json")<<"{\"state\":\""<<value<<"\"}\n";printf("state=%s\n",value);fflush(stdout);}
static bool hands_ready(){
 unsigned char data[592]{};std::ifstream in("/run/user/1000/frame-hands-native/input.bin",std::ios::binary);
 if(!in.read(reinterpret_cast<char*>(data),sizeof(data))||memcmp(data,"FHMODEL2",8))return false;
 auto u64=[&](int at){uint64_t v;memcpy(&v,data+at,8);return v;};
 auto u32=[&](int at){uint32_t v;memcpy(&v,data+at,4);return v;};
 const auto now=uint64_t(std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now().time_since_epoch()).count());
 if(u64(24)>now||now-u64(24)>80)return false;
 for(int side=0;side<2;++side){int at=40+side*276;if(!u32(at+4)||u64(at+8)>now||u64(at+16)<=now)return false;}
 return true;
}
static bool scene_ready(vr::IVRSystem* sys,vr::IVROverlay* ov){
 vr::TrackedDevicePose_t head{};sys->GetDeviceToAbsoluteTrackingPose(vr::TrackingUniverseRawAndUncalibrated,0,&head,1);
 char key[1024]{};auto apps=vr::VRApplications();apps->GetApplicationKeyByProcessId(apps->GetCurrentSceneProcessId(),key,sizeof(key));
 return head.bPoseIsValid&&head.bDeviceIsConnected&&!ov->IsDashboardVisible()&&strstr(key,"vrlink")&&hands_ready();
}
int main(int argc,char**argv){
 if(argc!=2)return 2;bool armed=std::string(argv[1])=="--arm";bool selftest=std::string(argv[1])=="--self-test";int seconds=armed?20:selftest?5:std::clamp(std::atoi(argv[1]),1,20);
 for(int s:{SIGINT,SIGTERM,SIGHUP,SIGALRM})std::signal(s,stop);alarm(armed?660:85);std::signal(SIGPIPE,stop);int lock=open("/run/user/1000/frame-hands-recorder.lock",O_CREAT|O_RDWR,0600);if(lock<0||flock(lock,LOCK_EX|LOCK_NB))return 6;
 vr::EVRInitError err;vr::VR_Init(&err,vr::VRApplication_Overlay);if(err){fprintf(stderr,"init=%d\n",int(err));return 3;}
 auto sys=vr::VRSystem();auto ov=vr::VROverlay();auto comp=vr::VRCompositor();bool original=comp->IsMirrorWindowVisible(),shown=false;FILE* encoder=nullptr;int result=1;
 VkInstance instance{};VkDevice device{};vr::VROverlayView_t view{};auto settings=vr::VRSettings();bool pause= settings->GetBool("power","pauseCompositorOnStandby");  vr::VROverlayHandle_t testOverlay=0;bool badFade=settings->GetBool("steamvr","forceFadeOnBadTracking");if(selftest)settings->SetBool("steamvr","forceFadeOnBadTracking",false);auto originalFade=comp->GetCurrentFadeColor(false);bool acquired=false;auto viewer=vr::VROverlayView();
try{
 GpuReadback gpu;
          const uint32_t instance_count=comp->GetVulkanInstanceExtensionsRequired(nullptr,0);
        require(instance_count<=65536,"instance_extensions_size");
        std::vector<char> bytes(instance_count,0);
        if(instance_count) require(comp->GetVulkanInstanceExtensionsRequired(bytes.data(),instance_count)<=instance_count,
            "instance_extensions_changed");
        auto names=extension_names(bytes);auto extensions=pointers(names);
        VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
        app.pApplicationName="Frame Cinema existing headset view observation";app.apiVersion=VK_API_VERSION_1_1;
        VkInstanceCreateInfo create{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
        create.pApplicationInfo=&app;create.enabledExtensionCount=uint32_t(extensions.size());
        create.ppEnabledExtensionNames=extensions.data();
        checked(vkCreateInstance(&create,nullptr,&instance),"create_instance");
        uint64_t output=0;sys->GetOutputDevice(&output,vr::TextureType_Vulkan,instance);
        auto physical=reinterpret_cast<VkPhysicalDevice>(output);
        require(physical!=VK_NULL_HANDLE,"runtime_physical_device_unavailable");
        uint32_t family_count=0;vkGetPhysicalDeviceQueueFamilyProperties(physical,&family_count,nullptr);
        require(family_count>0&&family_count<=256,"queue_family_count");
        std::vector<VkQueueFamilyProperties> families(family_count);
        vkGetPhysicalDeviceQueueFamilyProperties(physical,&family_count,families.data());
        uint32_t family=0;
        while(family<family_count&&(!(families[family].queueFlags&VK_QUEUE_GRAPHICS_BIT)||!families[family].queueCount)) ++family;
        require(family<family_count,"graphics_queue_unavailable");
        const uint32_t device_count=comp->GetVulkanDeviceExtensionsRequired(physical,nullptr,0);
        require(device_count<=65536,"device_extensions_size");bytes.assign(device_count,0);
        if(device_count) require(comp->GetVulkanDeviceExtensionsRequired(physical,bytes.data(),device_count)<=device_count,
            "device_extensions_changed");
        names=extension_names(bytes);extensions=pointers(names);
        float priority=1;
        VkDeviceQueueCreateInfo queue_create{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        queue_create.queueFamilyIndex=family;queue_create.queueCount=1;queue_create.pQueuePriorities=&priority;
        VkDeviceCreateInfo device_create{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
        device_create.queueCreateInfoCount=1;device_create.pQueueCreateInfos=&queue_create;
        device_create.enabledExtensionCount=uint32_t(extensions.size());device_create.ppEnabledExtensionNames=extensions.data();
        checked(vkCreateDevice(physical,&device_create,nullptr,&device),"create_device");
        VkQueue queue{};vkGetDeviceQueue(device,family,0,&queue);
        vr::VRVulkanDevice_t vulkan{instance,device,physical,queue,family};
        vr::VRNativeDevice_t native{&vulkan,vr::DeviceType_Vulkan};

if(armed){
 state("armed_waiting_for_streamed_scene_and_both_hands");
 auto deadline=Clock::now()+std::chrono::seconds(600);int consecutive=0;
 while(running&&Clock::now()<deadline){consecutive=scene_ready(sys,ov)?consecutive+1:0;if(consecutive>=10)break;std::this_thread::sleep_for(std::chrono::milliseconds(100));}
 require(running,"Cancelled while armed"); require(consecutive>=10,"No ready streamed scene and hands within ten minutes");
 }
 settings->SetBool("power","pauseCompositorOnStandby",false);
 if(!original){shown=true;comp->ShowMirrorWindow();}
  vr::VROverlayHandle_t handle=0;uint32_t w=0,h=0;auto ready=Clock::now()+std::chrono::seconds(15);
  while(running&&Clock::now()<ready){auto found=ov->FindOverlay(vr::k_pchHeadsetViewOverlayKey,&handle);int sized=-1;if(!found){if(!acquired){require(viewer->IsViewingPermitted(handle),"Headset View not permitted");auto a=viewer->AcquireOverlayView(handle,&native,&view,sizeof(view));fprintf(stdout,"acquire=%d\n",int(a));require(!a,"view acquisition failed");acquired=true;}sized=ov->GetOverlayImageData(handle,nullptr,0,&w,&h);if(w&&h)break;}fprintf(stdout,"mirror visible=%d find=%d size=%d width=%u height=%u\n",int(comp->IsMirrorWindowVisible()),int(found),sized,w,h);fflush(stdout);std::this_thread::sleep_for(std::chrono::milliseconds(1000));}
  if(!handle||!w||!h||uint64_t(w)*h>16000000)throw std::runtime_error("No bounded headset mirror image");
  std::vector<unsigned char> pixels;gpu.read(view,pixels,w,h);
  if(armed){
   state("waiting_for_nonblank_headset_video");bool content=false;auto deadline=Clock::now()+std::chrono::seconds(15);
   while(running&&Clock::now()<deadline){
    require(!viewer->AcquireOverlayView(handle,&native,&view,sizeof(view)),"refresh live view");
    gpu.read(view,pixels,w,h);
    int low=765,high=0;unsigned lit=0;
    for(unsigned y=80;y<std::min(h,1000u);y+=20)for(unsigned x=80;x<std::min(w,1840u);x+=20){size_t k=(size_t(y)*w+x)*4;int light=pixels[k]+pixels[k+1]+pixels[k+2];low=std::min(low,light);high=std::max(high,light);lit+=light>90;}
    content=high-low>90&&lit>80;if(content)break;
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
   }
   require(running&&content,"No nonblank headset video; no recording-start tone played");
  }
  char command[1024];snprintf(command,sizeof command,"ffmpeg -hide_banner -loglevel error -n -f rawvideo -pixel_format rgba -video_size %ux%u -framerate 15 -i pipe:0 -an -c:v libx264 -threads 2 -preset ultrafast -crf 22 -pix_fmt yuv420p -movflags +faststart headset.mp4 2>encoder.log",w,h);encoder=popen(command,"w");if(!encoder)throw std::runtime_error("encoder start");
  fprintf(stdout,"Recording %d seconds, %ux%u, 15 fps, native Headset View\n",seconds,w,h);fflush(stdout);
  if(selftest){require(!ov->CreateOverlay("framehands.recorder-self-test","Recorder self-test",&testOverlay),"test overlay");vr::HmdMatrix34_t pose{{{1,0,0,0},{0,1,0,0},{0,0,1,-0.6f}}};ov->SetOverlayTransformTrackedDeviceRelative(testOverlay,0,&pose);ov->SetOverlayWidthInMeters(testOverlay,1);ov->SetOverlayFlag(testOverlay,vr::VROverlayFlags_VisibleInDashboard,true);ov->SetOverlaySortOrder(testOverlay,100);require(!ov->ShowOverlay(testOverlay),"show self-test");} if(armed){state("recording");system("timeout 3 paplay ../start.wav");} unsigned frames=0,changed=0;uint64_t previous=0;auto start=Clock::now();auto next=start;
  while(running&&Clock::now()-start<std::chrono::seconds(seconds)){
   if(selftest)comp->FadeToColor(0,(frames%30)/29.f,0.2f,1-(frames%30)/29.f,1,false);if(testOverlay){std::vector<unsigned char> card(256*128*4,255);for(int y=0;y<128;++y)for(int x=0;x<256;++x){auto k=(y*256+x)*4;card[k]=(x+frames*9)%256;card[k+1]=y*2;card[k+2]=255-card[k];}require(!ov->SetOverlayRaw(testOverlay,card.data(),256,128,4),"self-test pixels");}require(!viewer->AcquireOverlayView(handle,&native,&view,sizeof(view)),"refresh view");gpu.read(view,pixels,w,h);
   uint64_t hash=1469598103934665603ull;for(size_t k=0;k<pixels.size();k+=97)hash=(hash^pixels[k])*1099511628211ull;changed+=frames&&hash!=previous;previous=hash;
   if(fwrite(pixels.data(),1,pixels.size(),encoder)!=pixels.size())throw std::runtime_error("encoder pipe");++frames;next+=std::chrono::microseconds(66667);std::this_thread::sleep_until(next);
  }
  int code=pclose(encoder);encoder=nullptr;double elapsed=std::chrono::duration<double>(Clock::now()-start).count();fprintf(stdout,"frames=%u changed_frames=%u elapsed=%.3f encoder_exit=%d\n",frames,changed,elapsed,code);result=code||!changed?4:0;state(result?"capture_failed":"complete");if(armed)system("timeout 3 paplay ../done.wav");
 }catch(const std::exception&e){fprintf(stderr,"capture: %s\n",e.what());state("error");}
 if(selftest)comp->FadeToColor(0,originalFade.r,originalFade.g,originalFade.b,originalFade.a,false);if(testOverlay)ov->DestroyOverlay(testOverlay);if(encoder)pclose(encoder);if(device)vkDeviceWaitIdle(device);if(acquired)viewer->ReleaseOverlayView(&view);if(shown)comp->HideMirrorWindow();fprintf(stdout,"mirror_original_visible=%d mirror_restored=%d\n",int(original),int(comp->IsMirrorWindowVisible()==original));if(selftest)settings->SetBool("steamvr","forceFadeOnBadTracking",badFade);settings->SetBool("power","pauseCompositorOnStandby",pause);fprintf(stdout,"standby_pause_restored=%d fade_policy_restored=%d\n",int(settings->GetBool("power","pauseCompositorOnStandby")==pause),int(settings->GetBool("steamvr","forceFadeOnBadTracking")==badFade));vr::VR_Shutdown();if(device)vkDestroyDevice(device,nullptr);if(instance)vkDestroyInstance(instance,nullptr);close(lock);return result;
}
