// A tiny dashboard visibility lease; no hand samples, clicks, or camera access.
#define FRAME_HANDS_DRIVER_MODEL
#include "driver_platform.h"
#include "dashboard_navigation.h"
#include <openvr.h>
#include <filesystem>
#include <fstream>
#include <thread>
#include <chrono>
#include <cstdio>
#ifndef _WIN32
#include <dlfcn.h>
#endif
int main(int argc,char** argv){
    if(argc!=3)return 2;
#ifdef _WIN32
    auto lib=LoadLibraryW(std::filesystem::u8path(argv[1]).c_str());
    auto symbol=[&](const char* name){return reinterpret_cast<void*>(GetProcAddress(lib,name));};
#else
    auto lib=dlopen(argv[1],RTLD_NOW|RTLD_LOCAL);
    auto symbol=[&](const char* name){return dlsym(lib,name);};
#endif
    if(!lib)return 3;
    auto init=reinterpret_cast<uint32_t(*)(vr::EVRInitError*,vr::EVRApplicationType,const char*)>(symbol("VR_InitInternal2"));
    auto iface=reinterpret_cast<void*(*)(const char*,vr::EVRInitError*)>(symbol("VR_GetGenericInterface"));
    auto shutdown=reinterpret_cast<void(*)()>(symbol("VR_ShutdownInternal"));
    if(!init||!iface||!shutdown)return 4;
    vr::EVRInitError error{};init(&error,vr::VRApplication_Background,nullptr);if(error)return 5;
    auto system=static_cast<vr::IVRSystem*>(iface(vr::IVRSystem_Version,&error));
    auto overlay=static_cast<vr::IVROverlay*>(iface(vr::IVROverlay_Version,&error));
    if(error||!system||!overlay)return 6;
    auto path=std::filesystem::u8path(argv[2]),temp=path;temp+=".ui-context.tmp";
    bool quit=false;
    while(!quit){
        vr::VREvent_t event{};while(system->PollNextEvent(&event,sizeof event))if(event.eventType==vr::VREvent_Quit){system->AcknowledgeQuit_Exiting();quit=true;}
        fhs::DashboardContext c;c.stamp_ms=fhs_platform::now_ms();c.enabled=!quit&&overlay->IsDashboardVisible();
        std::ofstream out(temp,std::ios::binary|std::ios::trunc);out.write(reinterpret_cast<const char*>(&c),sizeof c);out.close();
        if(!out)break;
#ifdef _WIN32
        if(!MoveFileExW(temp.c_str(),path.c_str(),MOVEFILE_REPLACE_EXISTING))break;
#else
        if(std::rename(temp.c_str(),path.c_str()))break;
#endif
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    std::error_code ec;std::filesystem::remove(path,ec);std::filesystem::remove(temp,ec);
    shutdown();return 0;
}
