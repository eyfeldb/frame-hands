// Native Piano transport variant: seconds=0 runs until control pipe EOF.
#include <openvr.h>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdint>
#if !defined(_WIN32)
#include <ctime>
#include <dlfcn.h>
#include <poll.h>
#include <unistd.h>
#endif
#include <string>
#include <system_error>

// Frame camera capture_mono_ns and this clock share CLOCK_MONOTONIC.
// Never supply the sensor/raw timestamp or a remote PC clock here.
#if !defined(_WIN32)
static uint64_t monotonic_ns() {
    timespec ts{};
    if(clock_gettime(CLOCK_MONOTONIC,&ts)) return 0;
    return uint64_t(ts.tv_sec)*1000000000ull+uint64_t(ts.tv_nsec);
}
#endif
static bool capture_age(uint64_t capture,uint64_t now,float& seconds) {
    if(!capture || capture>now || now-capture>250000000ull) return false;
    seconds=float(double(now-capture)*1e-9);
    return true;
}
static bool parse_stamp(const std::string& line,uint64_t& stamp) {
    if(line.empty()) return false;
    auto value=std::from_chars(line.data(),line.data()+line.size(),stamp);
    return value.ec==std::errc{} && value.ptr==line.data()+line.size();
}
static bool finite_pose(const vr::TrackedDevicePose_t& pose) {
    if(!pose.bPoseIsValid || !pose.bDeviceIsConnected) return false;
    for(const auto& row:pose.mDeviceToAbsoluteTracking.m)
        for(float value:row) if(!std::isfinite(value)) return false;
    return true;
}
static int self_test() {
    float age=0;uint64_t stamp=0;
    if(!capture_age(1000000000,1033000000,age) || std::fabs(age-.033f)>1e-7f) return 1;
    if(capture_age(0,1,age) || capture_age(2,1,age) || capture_age(1,250000002,age)) return 2;
    if(!capture_age(1,250000001,age) || !parse_stamp("18446744073709551615",stamp)) return 3;
    if(parse_stamp("-1",stamp) || parse_stamp("1x",stamp) || parse_stamp("18446744073709551616",stamp)) return 4;
    vr::TrackedDevicePose_t pose{};
    if(finite_pose(pose)) return 5;
    pose.bPoseIsValid=pose.bDeviceIsConnected=true;
    pose.mDeviceToAbsoluteTracking.m[0][0]=NAN;
    if(finite_pose(pose)) return 6;
    std::puts("native pose timestamp/validity checks passed");return 0;
}
int main(int argc,char** argv) {
    if(argc==2 && !std::strcmp(argv[1],"--self-test")) return self_test();
#if defined(_WIN32)
    return 2; // Host build only runs pure clock/validation tests.
#else
    if(argc!=3 || argv[1][0]!='/') return 2;
    uint64_t seconds=0;
    if(!parse_stamp(argv[2],seconds) || seconds>300) return 2;
    void* lib=dlopen(argv[1],RTLD_NOW|RTLD_LOCAL);
    if(!lib) {std::fprintf(stderr,"OpenVR library failed: %s\n",dlerror());return 3;}
    auto init=reinterpret_cast<uint32_t(*)(vr::EVRInitError*,vr::EVRApplicationType,const char*)>(dlsym(lib,"VR_InitInternal2"));
    auto iface=reinterpret_cast<void*(*)(const char*,vr::EVRInitError*)>(dlsym(lib,"VR_GetGenericInterface"));
    auto shutdown=reinterpret_cast<void(*)()>(dlsym(lib,"VR_ShutdownInternal"));
    if(!init || !iface || !shutdown) {dlclose(lib);return 4;}
    vr::EVRInitError error=vr::VRInitError_None;
    init(&error,vr::VRApplication_Background,nullptr);
    if(error) {dlclose(lib);return 5;}
    auto* system=static_cast<vr::IVRSystem*>(iface(vr::IVRSystem_Version,&error));
    if(!system || error) {shutdown();dlclose(lib);return 6;}
    std::setvbuf(stdout,nullptr,_IONBF,0);
    const auto started=monotonic_ns();
    if(!started) {shutdown();dlclose(lib);return 7;}
    const auto deadline=started+seconds*1000000000ull;
    std::string pending;bool quit=false;
    while(!quit) {
        const auto tick=monotonic_ns();if(!tick || (seconds && tick>=deadline)) break;
        pollfd input{STDIN_FILENO,POLLIN,0};const int available=poll(&input,1,100);
        if(available<0) break;
        if(!available) continue;
        char bytes[128];const auto count=read(STDIN_FILENO,bytes,sizeof bytes);
        if(count<=0) break;
        pending.append(bytes,size_t(count));
        while(true) {
            auto end=pending.find('\n');if(end==std::string::npos) break;
            auto line=pending.substr(0,end);pending.erase(0,end+1);
            if(!line.empty() && line.back()=='\r') line.pop_back();
            if(line=="Q") {quit=true;break;}
            uint64_t capture=0;float age=0;const auto queried=monotonic_ns();
            if(!parse_stamp(line,capture) || !capture_age(capture,queried,age)) {
                std::puts("{\"valid\":false,\"reason\":\"invalid_or_stale_capture_mono_ns\"}");continue;
            }
            vr::TrackedDevicePose_t historical{},current{};
            system->GetDeviceToAbsoluteTrackingPose(vr::TrackingUniverseRawAndUncalibrated,-age,&historical,1);
            system->GetDeviceToAbsoluteTrackingPose(vr::TrackingUniverseRawAndUncalibrated,0,&current,1);
            const bool valid=finite_pose(historical)&&finite_pose(current);
            if(!valid) {std::puts("{\"valid\":false,\"reason\":\"invalid_head_pose\"}");continue;}
            std::printf("{\"valid\":true,\"capture_mono_ns\":%llu,\"queried_mono_ns\":%llu,\"capture\":[",(unsigned long long)capture,(unsigned long long)queried);
            for(int i=0;i<12;++i) std::printf("%s%.9g",i?",":"",historical.mDeviceToAbsoluteTracking.m[i/4][i%4]);
            std::printf("],\"current\":[");
            for(int i=0;i<12;++i) std::printf("%s%.9g",i?",":"",current.mDeviceToAbsoluteTracking.m[i/4][i%4]);
            std::puts("]}");
        }
        if(pending.size()>64) break;
    }
    shutdown();dlclose(lib);return 0;
#endif
}
