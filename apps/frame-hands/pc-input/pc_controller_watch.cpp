// PC-side physical controller takeover for the fh_skeletal PC driver (Windows port of
// native/system-driver/controller_watch.cpp). While the PC driver is in Hands mode, a fresh button press on a
// physical Frame controller (as mirrored onto the PC by vrlink) selects Controllers by publishing mode.bin in the
// driver bundle, the same file select_input_mode.py writes. The headset-side watcher does the same on the Frame,
// but during PC streaming its button reads are not reliable, and the headset's choice only reaches the PC while
// the wireless relay runs; this watcher closes both gaps. It waits for SteamVR, survives SteamVR restarts, and
// never changes anything else (no role, binding or pose changes).
// usage: pc-controller-watch <openvr_api.dll> <mode.bin path> [log path]
#define NOMINMAX
#include <windows.h>
#include <openvr.h>
#include "controller_takeover.h"
#include "controller_pickup.h"
#include "input_mode.h"
#include <array>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <thread>
#include <algorithm>
static uint64_t stamp(){return std::chrono::steady_clock::now().time_since_epoch()/std::chrono::nanoseconds(1);}
static FILE* logfile=nullptr;
static void logline(const char* text){SYSTEMTIME t;GetLocalTime(&t);std::printf("%02d:%02d:%02d %s\n",t.wHour,t.wMinute,t.wSecond,text);if(logfile){std::fprintf(logfile,"%02d:%02d:%02d %s\n",t.wHour,t.wMinute,t.wSecond,text);std::fflush(logfile);}}
static bool read_mode(const std::string& path,fhs::ModeCommand& c){
    std::ifstream f(path,std::ios::binary);fhs::ModeSelection s;
    return bool(f.read(reinterpret_cast<char*>(&c),sizeof c))&&f.peek()==EOF&&s.accept(c);
}
template<class T>T symbol(HMODULE lib,const char* name){auto raw=GetProcAddress(lib,name);T result=nullptr;static_assert(sizeof raw==sizeof result);std::memcpy(&result,&raw,sizeof result);return result;}
int main(int argc,char** argv){
    if(argc<3){std::puts("usage: pc-controller-watch <openvr_api.dll> <mode.bin> [log]");return 2;}
    std::string path=argv[2],temp=path+".pc-watch.tmp";
    if(argc>=4)logfile=std::fopen(argv[3],"a");
    HANDLE mutex=CreateMutexA(nullptr,TRUE,"Local\\framehands-pc-controller-watch");
    if(!mutex||GetLastError()==ERROR_ALREADY_EXISTS){logline("another pc-controller-watch is running");return 3;}
    HMODULE lib=LoadLibraryA(argv[1]);if(!lib){logline("cannot load openvr_api.dll");return 4;}
    auto init=symbol<uint32_t(*)(vr::EVRInitError*,vr::EVRApplicationType,const char*)>(lib,"VR_InitInternal2");
    auto iface=symbol<void*(*)(const char*,vr::EVRInitError*)>(lib,"VR_GetGenericInterface");
    auto shutdown=symbol<void(*)()>(lib,"VR_ShutdownInternal");
    if(!init||!iface||!shutdown){logline("openvr entry points missing");return 5;}
    std::setvbuf(stdout,nullptr,_IONBF,0);
    logline("pc-controller-watch started");
    for(;;){   // outer loop: one SteamVR session per iteration
        vr::EVRInitError error{};init(&error,vr::VRApplication_Background,nullptr);
        if(error){std::this_thread::sleep_for(std::chrono::seconds(3));continue;}
        auto* system=static_cast<vr::IVRSystem*>(iface(vr::IVRSystem_Version,&error));
        auto* input=static_cast<vr::IVRInput*>(iface(vr::IVRInput_Version,&error));
        auto* models=static_cast<vr::IVRRenderModels*>(iface(vr::IVRRenderModels_Version,&error));
        auto* drivers=static_cast<vr::IVRDriverManager*>(iface(vr::IVRDriverManager_Version,&error));
        if(!system||!input||!models||!drivers||error){shutdown();std::this_thread::sleep_for(std::chrono::seconds(3));continue;}
        logline("attached to SteamVR");
        std::array<fhnav::ButtonGate,vr::k_unMaxTrackedDeviceCount> gates{};
        std::array<fhnav::PickupGate,vr::k_unMaxTrackedDeviceCount> pickups{};
        std::array<bool,vr::k_unMaxTrackedDeviceCount> available{};
        struct Physical {bool valid=false;std::string model,serial,type;vr::VRInputValueHandle_t path=0;};
        std::array<Physical,vr::k_unMaxTrackedDeviceCount> devices{};
        uint64_t next_discovery=0,epoch=0,settle=0,pending_until=0;bool quit=false;
        while(!quit){
            fhs::ModeCommand mode{};bool selected=read_mode(path,mode);auto now=stamp();
            if(selected&&mode.sequence!=epoch){epoch=mode.sequence;settle=now+300000000;for(auto& g:gates)g.reset();for(auto& g:pickups)g.reset();char b[96];std::snprintf(b,sizeof b,"mode file: %s",mode.mode==fhs::InputMode::Hands?"hands":mode.mode==fhs::InputMode::Controllers?"controllers":"auto");logline(b);}
            std::array<bool,vr::k_unMaxTrackedDeviceCount> physical{};
            bool press=false;
            vr::TrackedDevicePose_t poses[vr::k_unMaxTrackedDeviceCount]{};
            system->GetDeviceToAbsoluteTrackingPose(vr::TrackingUniverseStanding,0.f,poses,vr::k_unMaxTrackedDeviceCount);
            if(now>=next_discovery){
                next_discovery=now+1000000000;
                for(uint32_t i=0;i<devices.size();++i){
                    auto& d=devices[i];d.valid=false;
                    if(!system->IsTrackedDeviceConnected(i)||system->GetTrackedDeviceClass(i)!=vr::TrackedDeviceClass_Controller)continue;
                    char type[128]{},serial[128]{},model[256]{};
                    system->GetStringTrackedDeviceProperty(i,vr::Prop_ControllerType_String,type,sizeof type);
                    // never our own optical devices; any other physical controller counts
                    if(!std::strcmp(type,"fh_skeletal")||!std::strcmp(type,"fh_pointer"))continue;
                    system->GetStringTrackedDeviceProperty(i,vr::Prop_SerialNumber_String,serial,sizeof serial);
                    system->GetStringTrackedDeviceProperty(i,vr::Prop_RenderModelName_String,model,sizeof model);
                    auto parent=system->GetUint64TrackedDeviceProperty(i,vr::Prop_ParentDriver_Uint64);
                    std::string owner;
                    for(uint32_t j=0;j<drivers->GetDriverCount();++j){char name[128]{};drivers->GetDriverName(j,name,sizeof name);if(drivers->GetDriverHandle(name)==parent){owner=name;break;}}
                    if(owner.empty()||!serial[0])continue;
                    if(d.serial!=serial||d.model!=model){gates[i].reset();d.serial=serial;d.model=model;d.type=type;char b[400];std::snprintf(b,sizeof b,"controller %u type=%s owner=%s serial=%s model=%s",i,type,owner.c_str(),serial,model);logline(b);}
                    std::string ipath="/devices/"+owner+"/"+serial;
                    d.valid=input->GetInputSourceHandle(ipath.c_str(),&d.path)==vr::VRInputError_None;
                }
            }
            for(uint32_t i=0;i<devices.size();++i){
                auto& d=devices[i];physical[i]=d.valid&&system->IsTrackedDeviceConnected(i);
                uint64_t buttons=0;bool ok=false;
                if(physical[i]){
                    const char* names[]={"button_a","button_b","button_x","button_y","button_menu","button_view","button_system","bumper","trigger","button_grip","thumbstick","dpad"};
                    if(!d.model.empty())for(unsigned n=0;n<sizeof(names)/sizeof(names[0]);++n){
                        vr::RenderModel_ControllerMode_State_t m{};vr::RenderModel_ComponentState_t c{};
                        if(models->GetComponentStateForDevicePath(d.model.c_str(),names[n],d.path,&m,&c)){ok=true;if(c.uProperties&vr::VRComponentProperty_IsPressed)buttons|=1ull<<n;}
                    }
                    // legacy controller state as a second source (vrlink-mirrored controllers expose it)
                    vr::VRControllerState_t st{};
                    if(system->GetControllerState(i,&st,sizeof st)){ok=true;if(st.ulButtonPressed)buttons|=1ull<<20;}
                }
                if(ok!=available[i]){available[i]=ok;char b[96];std::snprintf(b,sizeof b,"controller_components device=%u available=%d",i,int(ok));logline(b);}
                if(now>=settle){bool p=gates[i].sample(ok,buttons);if(p){char b[96];std::snprintf(b,sizeof b,"takeover_candidate device=%u buttons=%llx",i,(unsigned long long)buttons);logline(b);}press=p||press;}
                // pick-up by motion: the only signal a background app still sees while a streamed game holds input focus
                if(d.valid){const auto& p=poses[i];const auto& m=p.mDeviceToAbsoluteTracking.m;
                    const bool tracked=p.bPoseIsValid&&p.bDeviceIsConnected&&p.eTrackingResult==vr::TrackingResult_Running_OK;
                    bool lifted=pickups[i].sample(tracked,m[0][3],m[1][3],m[2][3],now);
                    if(lifted&&now>=settle){char b[96];std::snprintf(b,sizeof b,"takeover_candidate source=pickup device=%u",i);logline(b);press=true;}
                    static std::array<uint64_t,vr::k_unMaxTrackedDeviceCount> last_debug{};
                    if(now-last_debug[i]>2000000000ull){last_debug[i]=now;char b[200];std::snprintf(b,sizeof b,"pose device=%u tracked=%d result=%d pos=%.2f,%.2f,%.2f resting=%d",i,int(tracked),int(p.eTrackingResult),m[0][3],m[1][3],m[2][3],int(pickups[i].resting));logline(b);}}
            }
            vr::VREvent_t event{};
            while(system->PollNextEvent(&event,sizeof event)){
                if(event.eventType==vr::VREvent_Quit){system->AcknowledgeQuit_Exiting();quit=true;}
                if(event.eventType==vr::VREvent_ButtonPress&&event.trackedDeviceIndex<physical.size()&&physical[event.trackedDeviceIndex]&&now>=settle&&gates[event.trackedDeviceIndex].armed){press=true;logline("takeover_candidate source=event");}
            }
            // A detected press/pick-up stays pending for a second: the relay may rewrite the mode file (new sequence)
            // at the same moment, and a single missed attempt must not swallow the takeover.
            if(press)pending_until=now+1000000000ull;
            if(pending_until&&now<pending_until&&selected&&mode.mode==fhs::InputMode::Hands){
                fhs::ModeCommand current{};
                if(read_mode(path,current)&&current.mode==fhs::InputMode::Hands){
                    current.mode=fhs::InputMode::Controllers;current.sequence=std::max(stamp(),current.sequence+1);
                    std::ofstream out(temp,std::ios::binary|std::ios::trunc);out.write(reinterpret_cast<char*>(&current),sizeof current);out.close();
                    if(out&&MoveFileExA(temp.c_str(),path.c_str(),MOVEFILE_REPLACE_EXISTING)){logline("Controllers selected: physical controller");pending_until=0;}
                    else logline("Controller mode publication failed; retrying");
                }else pending_until=0;
            }else if(pending_until&&now>=pending_until)pending_until=0;
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
        shutdown();logline("SteamVR exited; waiting for the next session");
        std::this_thread::sleep_for(std::chrono::seconds(3));
    }
}
