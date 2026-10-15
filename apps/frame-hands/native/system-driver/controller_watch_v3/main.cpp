// Frame Hands physical-controller takeover, v3 (2026-10-12).
// Problem solved here: SteamVR hides controller buttons from background apps while a scene app (vrlink streaming a
// PC game, or a native game) holds input focus, so the component/event probes below see nothing exactly when the
// wearer is in a game and wants Controllers back. The sanctioned way around it is an OVERLAY application with an
// action set in the overlay global-priority range: its bound inputs are delivered to us and withheld from the scene
// app. The set is only active while Hands is selected (controllers are idle then anyway) and the dashboard is
// closed, so games and the dashboard keep full control otherwise. Requires steamvr.globalActionSetPriority=true.
// usage: controller-watch <libopenvr_api.so> <mode.bin> [action_manifest.json]
#include <openvr.h>
#include "controller_takeover.h"
#include "controller_pickup.h"
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <thread>
#include <dlfcn.h>
#include <unistd.h>
#include <sys/file.h>
#include <fcntl.h>
#include <csignal>
static volatile sig_atomic_t g_stop=0;
static void on_signal(int){g_stop=1;}
#include <algorithm>
#include "input_mode.h"
static uint64_t stamp(){return std::chrono::steady_clock::now().time_since_epoch()/std::chrono::nanoseconds(1);}
static bool read_mode(const std::string& path,fhs::ModeCommand& c){
    std::ifstream f(path,std::ios::binary);fhs::ModeSelection s;
    return bool(f.read(reinterpret_cast<char*>(&c),sizeof c))&&f.peek()==EOF&&s.accept(c);
}
int main(int argc,char** argv){
    if(argc<3||argc>4)return 2;
    std::string path=argv[2],temp=path+".controller-watch.tmp",lockpath=path+".controller-watch.lock";
    int lock=open(lockpath.c_str(),O_CREAT|O_RDWR|O_CLOEXEC,0600);
    if(lock<0||flock(lock,LOCK_EX|LOCK_NB))return 3;
    void* lib=dlopen(argv[1],RTLD_NOW|RTLD_LOCAL);if(!lib)return 4;
    auto init=reinterpret_cast<uint32_t(*)(vr::EVRInitError*,vr::EVRApplicationType,const char*)>(dlsym(lib,"VR_InitInternal2"));
    auto iface=reinterpret_cast<void*(*)(const char*,vr::EVRInitError*)>(dlsym(lib,"VR_GetGenericInterface"));
    auto shutdown=reinterpret_cast<void(*)()>(dlsym(lib,"VR_ShutdownInternal"));
    if(!init||!iface||!shutdown)return 5;
    std::setvbuf(stdout,nullptr,_IONBF,0);
    std::signal(SIGTERM,on_signal);std::signal(SIGINT,on_signal);
    // Overlay application type: required for the global-priority action set. No overlay surface is created.
    vr::EVRInitError error{};init(&error,vr::VRApplication_Overlay,nullptr);
    if(error){std::fprintf(stderr,"VR init failed: %d\n",int(error));return 6;}
    auto* system=static_cast<vr::IVRSystem*>(iface(vr::IVRSystem_Version,&error));if(!system||error)return 7;
    std::array<fhnav::ButtonGate,vr::k_unMaxTrackedDeviceCount> gates{};
    std::array<fhnav::PickupGate,vr::k_unMaxTrackedDeviceCount> pickups{};
    std::array<bool,vr::k_unMaxTrackedDeviceCount> available{};
    auto* input=static_cast<vr::IVRInput*>(iface(vr::IVRInput_Version,&error));
    auto* models=static_cast<vr::IVRRenderModels*>(iface(vr::IVRRenderModels_Version,&error));
    auto* drivers=static_cast<vr::IVRDriverManager*>(iface(vr::IVRDriverManager_Version,&error));
    auto* overlay=static_cast<vr::IVROverlay*>(iface(vr::IVROverlay_Version,&error));
    if(!input||!models||!drivers||!overlay||error)return 8;
    // Action set with every Frame controller button bound to one action (see takeover_frame_controller.json).
    vr::VRActionSetHandle_t takeover_set=vr::k_ulInvalidActionSetHandle;vr::VRActionHandle_t takeover_press=vr::k_ulInvalidActionHandle;
    bool actions=false;
    if(argc==4){
        auto e1=input->SetActionManifestPath(argv[3]);
        auto e2=input->GetActionSetHandle("/actions/takeover",&takeover_set);
        auto e3=input->GetActionHandle("/actions/takeover/in/press",&takeover_press);
        actions=e1==vr::VRInputError_None&&e2==vr::VRInputError_None&&e3==vr::VRInputError_None;
        std::printf("action manifest %s: %d %d %d -> %s\n",argv[3],int(e1),int(e2),int(e3),actions?"armed":"unavailable");
    }
    struct Physical {bool valid=false;std::string model,serial;vr::VRInputValueHandle_t path=0;};
    std::array<Physical,vr::k_unMaxTrackedDeviceCount> devices{};
    uint64_t next_discovery=0,pending_until=0,last_override_log=0;bool override_active=false;
    fhnav::TakeoverWindow window;bool quit=false;bool action_was_down=false;
    while(!quit&&!g_stop){
        fhs::ModeCommand mode{};bool selected=read_mode(path,mode);auto now=stamp();
        // The dashboard being open is NOT a reason to ignore the controller any more (2026-10-13): the wearer uses a
        // button precisely to get the controller back when stuck in the dashboard with Hands active. Only the Hands
        // click itself is protected, by the settle window after the mode change plus the button-release gate. The
        // system button is SteamVR's own dashboard toggle and never counts.
        const bool dashboard=overlay->IsDashboardVisible();
        if(window.update(selected?mode.sequence:0,false,now)){for(auto& g:gates)g.reset();for(auto& g:pickups)g.reset();action_was_down=false;}
        const bool hands=selected&&mode.mode==fhs::InputMode::Hands;
        std::array<bool,vr::k_unMaxTrackedDeviceCount> physical{};
        bool press=false;const char* source="";
        // 1) Global-priority action set: the only button path that survives a focused scene app.
        if(actions){
            const bool want=hands;
            vr::VRActiveActionSet_t set{};set.ulActionSet=takeover_set;set.ulRestrictedToDevice=vr::k_ulInvalidInputValueHandle;
            set.nPriority=vr::k_nActionSetOverlayGlobalPriorityMin+0x10;
            auto e=want?input->UpdateActionState(&set,sizeof set,1):input->UpdateActionState(nullptr,sizeof set,0);
            if(want!=override_active||now-last_override_log>60000000000ull){override_active=want;last_override_log=now;std::printf("button override %s (UpdateActionState %d)\n",want?"ACTIVE":"released",int(e));}
            if(want&&e==vr::VRInputError_None){
                vr::InputDigitalActionData_t d{};
                if(input->GetDigitalActionData(takeover_press,&d,sizeof d,vr::k_ulInvalidInputValueHandle)==vr::VRInputError_None){
                    // Edge-triggered on a press observed after the settle window; the activation click that selected
                    // Hands is swallowed because the set was not active yet when it happened.
                    static bool active_was=false;
                    if(d.bActive!=active_was){active_was=d.bActive;std::printf("takeover action %s\n",d.bActive?"bound and active":"INACTIVE (binding not reaching us)");}
                    if(d.bActive&&d.bState&&!action_was_down)std::printf("action press seen (ready=%d)\n",int(window.ready(now)));
                    if(d.bActive&&d.bState&&!action_was_down&&window.ready(now)){press=true;source="action";}
                    action_was_down=d.bActive&&d.bState;
                }
            }else action_was_down=false;
        }
        // 2) Legacy probes (work when nothing holds focus), 3) pick-up by motion (works always; needs a prior rest).
        vr::TrackedDevicePose_t poses[vr::k_unMaxTrackedDeviceCount]{};
        system->GetDeviceToAbsoluteTrackingPose(vr::TrackingUniverseStanding,0.f,poses,vr::k_unMaxTrackedDeviceCount);
        if(now>=next_discovery){
            next_discovery=now+1000000000;
            for(uint32_t i=0;i<devices.size();++i){
                auto& d=devices[i];d.valid=false;
                if(!system->IsTrackedDeviceConnected(i)||system->GetTrackedDeviceClass(i)!=vr::TrackedDeviceClass_Controller)continue;
                char type[128]{},serial[128]{},model[256]{};
                system->GetStringTrackedDeviceProperty(i,vr::Prop_ControllerType_String,type,sizeof type);
                if(std::strcmp(type,"frame_controller"))continue;
                system->GetStringTrackedDeviceProperty(i,vr::Prop_SerialNumber_String,serial,sizeof serial);
                system->GetStringTrackedDeviceProperty(i,vr::Prop_RenderModelName_String,model,sizeof model);
                auto parent=system->GetUint64TrackedDeviceProperty(i,vr::Prop_ParentDriver_Uint64);
                std::string owner;
                for(uint32_t j=0;j<drivers->GetDriverCount();++j){
                    char name[128]{};drivers->GetDriverName(j,name,sizeof name);
                    if(drivers->GetDriverHandle(name)==parent){owner=name;break;}
                }
                if(owner.empty()||!serial[0]||!model[0])continue;
                if(d.serial!=serial||d.model!=model){gates[i].reset();d.serial=serial;d.model=model;std::printf("controller %u serial=%s model=%s\n",i,serial,model);}
                std::string ipath="/devices/"+owner+"/"+serial;
                d.valid=input->GetInputSourceHandle(ipath.c_str(),&d.path)==vr::VRInputError_None;
            }
        }
        for(uint32_t i=0;i<devices.size();++i){
            auto& d=devices[i];physical[i]=d.valid&&system->IsTrackedDeviceConnected(i);
            uint64_t buttons=0;bool ok=false;
            if(physical[i]){
                const char* names[]={"button_a","button_b","button_x","button_y","button_menu","button_view","button_system","bumper","trigger","button_grip","thumbstick","dpad"};
                for(unsigned n=0;n<sizeof(names)/sizeof(names[0]);++n){
                    vr::RenderModel_ControllerMode_State_t m{};vr::RenderModel_ComponentState_t c{};
                    if(models->GetComponentStateForDevicePath(d.model.c_str(),names[n],d.path,&m,&c)){
                        ok=true;if(c.uProperties&vr::VRComponentProperty_IsPressed)buttons|=1ull<<n;
                    }
                }
            }
            if(ok!=available[i]){available[i]=ok;std::printf("controller_components device=%u available=%d\n",i,ok);}
            buttons&=~(1ull<<6);   // "button_system": SteamVR's dashboard toggle, never a takeover
            if(window.ready(now)&&gates[i].sample(ok,buttons,now)){if(!press)source="components";press=true;}
            if(d.valid){const auto& p=poses[i];const auto& m=p.mDeviceToAbsoluteTracking.m;
                bool lifted=pickups[i].sample(p.bPoseIsValid&&p.bDeviceIsConnected&&p.eTrackingResult==vr::TrackingResult_Running_OK,m[0][3],m[1][3],m[2][3],now);
                if(lifted&&window.ready(now)){if(!press)source="pickup";press=true;}}
        }
        vr::VREvent_t event{};
        while(system->PollNextEvent(&event,sizeof event)){
            if(event.eventType==vr::VREvent_Quit){system->AcknowledgeQuit_Exiting();quit=true;}
            if(event.eventType==vr::VREvent_ButtonPress&&event.data.controller.button!=vr::k_EButton_System&&event.trackedDeviceIndex<physical.size()&&physical[event.trackedDeviceIndex]
                &&gates[event.trackedDeviceIndex].fresh_event(stamp(),event.eventAgeSeconds,window.settle)){if(!press)source="event";press=true;}
        }
        // A button while the dashboard is open only DISMISSES the dashboard and keeps Hands (the wearer wants to get
        // into the game with hands); a button with the dashboard closed hands control back to the controllers.
        if(press&&hands&&dashboard){
            std::printf("dashboard dismissed by controller (source=%s), Hands kept\n",source);
            if(std::system("/opt/steamvr/bin/linuxarm64/vrcmd --hidedashboard >/dev/null 2>&1 &")!=0)std::fprintf(stderr,"vrcmd --hidedashboard failed to launch\n");
            for(auto& g:gates)g.reset();for(auto& g:pickups)g.reset();action_was_down=false;window.update(0,false,now);window.update(selected?mode.sequence:0,false,now);
        }else if(press&&hands){std::printf("takeover_candidate source=%s\n",source);pending_until=now+1000000000ull;}
        // Publish Controllers; a detected press stays pending for a second so a concurrent mode-file rewrite (relay,
        // menu) cannot swallow it. A user request that already left Hands cancels it.
        if(pending_until&&now<pending_until&&hands){
            fhs::ModeCommand current{};
            if(read_mode(path,current)&&current.mode==fhs::InputMode::Hands){
                current.mode=fhs::InputMode::Controllers;current.sequence=std::max(stamp(),current.sequence+1);
                std::ofstream out(temp,std::ios::binary|std::ios::trunc);out.write(reinterpret_cast<char*>(&current),sizeof current);out.close();
                if(out&&std::rename(temp.c_str(),path.c_str())==0){std::puts("Controllers selected: physical controller");pending_until=0;}
                else std::fprintf(stderr,"Controller mode publication failed; retrying\n");
            }else pending_until=0;
        }else if(pending_until&&now>=pending_until)pending_until=0;
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    // Leave nothing behind: release the button override explicitly before disconnecting (service stop = SIGTERM).
    if(actions){input->UpdateActionState(nullptr,sizeof(vr::VRActiveActionSet_t),0);std::puts("button override released on exit");}
    shutdown();dlclose(lib);close(lock);return 0;
}
