#include <openvr.h>
#include "controller_takeover.h"
#include "controller_pickup.h"
#include <array>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <thread>
#include <dlfcn.h>
#include <unistd.h>
#include <sys/file.h>
#include <fcntl.h>
#include <algorithm>
#include "pc-input/input_mode.h"
static uint64_t stamp(){return std::chrono::steady_clock::now().time_since_epoch()/std::chrono::nanoseconds(1);}
static bool read_mode(const std::string& path,fhs::ModeCommand& c){
    std::ifstream f(path,std::ios::binary);fhs::ModeSelection s;
    return bool(f.read(reinterpret_cast<char*>(&c),sizeof c))&&f.peek()==EOF&&s.accept(c);
}
int main(int argc,char** argv){
    if(argc!=3)return 2;
    std::string path=argv[2],temp=path+".controller-watch.tmp",lockpath=path+".controller-watch.lock";
    int lock=open(lockpath.c_str(),O_CREAT|O_RDWR|O_CLOEXEC,0600);
    if(lock<0||flock(lock,LOCK_EX|LOCK_NB))return 3;
    void* lib=dlopen(argv[1],RTLD_NOW|RTLD_LOCAL);if(!lib)return 4;
    auto init=reinterpret_cast<uint32_t(*)(vr::EVRInitError*,vr::EVRApplicationType,const char*)>(dlsym(lib,"VR_InitInternal2"));
    auto iface=reinterpret_cast<void*(*)(const char*,vr::EVRInitError*)>(dlsym(lib,"VR_GetGenericInterface"));
    auto shutdown=reinterpret_cast<void(*)()>(dlsym(lib,"VR_ShutdownInternal"));
    if(!init||!iface||!shutdown)return 5;
    vr::EVRInitError error{};init(&error,vr::VRApplication_Background,nullptr);if(error)return 6;
    auto* system=static_cast<vr::IVRSystem*>(iface(vr::IVRSystem_Version,&error));if(!system||error)return 7;
    std::array<fhnav::ButtonGate,vr::k_unMaxTrackedDeviceCount> gates{};
    std::array<fhnav::PickupGate,vr::k_unMaxTrackedDeviceCount> pickups{};
    std::array<bool,vr::k_unMaxTrackedDeviceCount> available{};
    auto* input=static_cast<vr::IVRInput*>(iface(vr::IVRInput_Version,&error));
    auto* models=static_cast<vr::IVRRenderModels*>(iface(vr::IVRRenderModels_Version,&error));
    auto* drivers=static_cast<vr::IVRDriverManager*>(iface(vr::IVRDriverManager_Version,&error));
    if(!input||!models||!drivers||error)return 8;
    struct Physical {bool valid=false;std::string model,serial;vr::VRInputValueHandle_t path=0;};
    std::array<Physical,vr::k_unMaxTrackedDeviceCount> devices{};
    uint64_t next_discovery=0;
    uint64_t epoch=0,settle=0,pending_until=0;bool quit=false;
    std::setvbuf(stdout,nullptr,_IONBF,0);
    while(!quit){
        fhs::ModeCommand mode{};bool selected=read_mode(path,mode);auto now=stamp();
        if(selected&&mode.sequence!=epoch){epoch=mode.sequence;settle=now+300000000;for(auto& g:gates)g.reset();for(auto& g:pickups)g.reset();}
        std::array<bool,vr::k_unMaxTrackedDeviceCount> physical{};
        bool press=false;
        // Poses are readable even while a streamed PC game holds input focus and hides button presses from us.
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
                if(d.serial!=serial||d.model!=model){gates[i].reset();d.serial=serial;d.model=model;}
                std::string path="/devices/"+owner+"/"+serial;
                d.valid=input->GetInputSourceHandle(path.c_str(),&d.path)==vr::VRInputError_None;
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
            if(now>=settle){bool p=gates[i].sample(ok,buttons);if(p)std::printf("takeover_candidate source=components device=%u buttons=%llu\n",i,(unsigned long long)buttons);press=p||press;}
            if(d.valid){const auto& p=poses[i];const auto& m=p.mDeviceToAbsoluteTracking.m;
                bool lifted=pickups[i].sample(p.bPoseIsValid&&p.bDeviceIsConnected&&p.eTrackingResult==vr::TrackingResult_Running_OK,m[0][3],m[1][3],m[2][3],now);
                if(lifted&&now>=settle){std::printf("takeover_candidate source=pickup device=%u\n",i);press=true;}}
        }
        vr::VREvent_t event{};
        while(system->PollNextEvent(&event,sizeof event)){
            if(event.eventType==vr::VREvent_Quit){system->AcknowledgeQuit_Exiting();quit=true;}
            if(event.eventType==vr::VREvent_ButtonPress&&event.trackedDeviceIndex<physical.size()&&physical[event.trackedDeviceIndex]&&now>=settle&&gates[event.trackedDeviceIndex].armed)press=true;
        }
        // A detected press/pick-up stays pending for a second so a concurrent mode-file rewrite cannot swallow it;
        // a user request that already left Hands cancels it.
        if(press)pending_until=now+1000000000ull;
        if(pending_until&&now<pending_until&&selected&&mode.mode==fhs::InputMode::Hands){
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
    shutdown();dlclose(lib);close(lock);return 0;
}
