// Capture what the PC game actually sees from our hands next to what we fed SteamVR, for offline diagnosis.
// Every ~4 ms: QPC time, HMD pose (now and +40 ms predicted), both optical hand devices (connected/valid, pose now
// and +40 ms predicted), the skeletal summary finger curls through IVRInput, and the FHMODEL2 mailbox packet
// (sequence, stamps, wrist + index tip per hand). CSV to the given path. Read-only: no input is generated.
// usage: capture-session <openvr_api.dll> <probe-actions.json> <mailbox input.bin> <out.csv> <seconds>
#define NOMINMAX
#include <windows.h>
#include <openvr.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <chrono>
using namespace vr;
template<class T>T symbol(HMODULE lib,const char* name){auto raw=GetProcAddress(lib,name);T result=nullptr;static_assert(sizeof raw==sizeof result);std::memcpy(&result,&raw,sizeof result);return result;}
static double now_ms(){LARGE_INTEGER f,c;QueryPerformanceFrequency(&f);QueryPerformanceCounter(&c);return 1000.0*double(c.QuadPart)/double(f.QuadPart);}
#pragma pack(push,1)
struct Header{char magic[8];uint32_t version,size;uint64_t sequence,published,source;};
struct Hand{uint32_t id,flags;uint64_t capture,until;float points[63];};
#pragma pack(pop)
static bool read_mailbox(const char* path,Header& h,Hand hands[2]){
    auto f=CreateFileA(path,GENERIC_READ,FILE_SHARE_READ|FILE_SHARE_WRITE|FILE_SHARE_DELETE,nullptr,OPEN_EXISTING,0,nullptr);
    if(f==INVALID_HANDLE_VALUE)return false;
    char buf[592];DWORD n=0;bool ok=ReadFile(f,buf,sizeof buf,&n,nullptr)&&n==sizeof buf;CloseHandle(f);
    if(!ok)return false;std::memcpy(&h,buf,sizeof h);std::memcpy(hands,buf+sizeof h,sizeof(Hand)*2);return std::memcmp(h.magic,"FHMODEL2",8)==0;
}
static void pose_fields(FILE* out,const TrackedDevicePose_t& p){
    const auto& m=p.mDeviceToAbsoluteTracking.m;
    // position, forward axis (-z column) so orientation jitter is visible without quaternions
    std::fprintf(out,",%d,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f",int(p.bDeviceIsConnected),int(p.bPoseIsValid),m[0][3],m[1][3],m[2][3],-m[0][2],-m[1][2],-m[2][2]);
}
int main(int argc,char** argv){
    if(argc!=6){std::puts("usage: capture-session <openvr_api.dll> <actions.json> <input.bin> <out.csv> <seconds>");return 2;}
    HMODULE lib=LoadLibraryA(argv[1]);if(!lib)return 3;
    auto init=symbol<uint32_t(*)(EVRInitError*,EVRApplicationType,const char*)>(lib,"VR_InitInternal2");auto iface=symbol<void*(*)(const char*,EVRInitError*)>(lib,"VR_GetGenericInterface");auto shutdown=symbol<void(*)()>(lib,"VR_ShutdownInternal");
    EVRInitError error{};init(&error,VRApplication_Background,nullptr);if(error){std::printf("init_error=%d\n",int(error));return 5;}
    auto system=static_cast<IVRSystem*>(iface(IVRSystem_Version,&error));auto input=static_cast<IVRInput*>(iface(IVRInput_Version,&error));if(!system||!input)return 6;
    if(auto e=input->SetActionManifestPath(argv[2])){std::printf("manifest_error=%d\n",int(e));}
    VRActionSetHandle_t set{};input->GetActionSetHandle("/actions/main",&set);VRActiveActionSet_t active{};active.ulActionSet=set;
    VRActionHandle_t skeleton[2]{};input->GetActionHandle("/actions/main/in/skeletonleft",&skeleton[0]);input->GetActionHandle("/actions/main/in/skeletonright",&skeleton[1]);
    FILE* out=std::fopen(argv[4],"w");if(!out)return 7;
    std::fprintf(out,"t_ms,hmd_conn,hmd_valid,hmd_x,hmd_y,hmd_z,hmd_fx,hmd_fy,hmd_fz,hmdp_conn,hmdp_valid,hmdp_x,hmdp_y,hmdp_z,hmdp_fx,hmdp_fy,hmdp_fz");
    for(const char* s:{"l","r"})std::fprintf(out,",%s_idx,%s_conn,%s_valid,%s_x,%s_y,%s_z,%s_fx,%s_fy,%s_fz,%sp_conn,%sp_valid,%sp_x,%sp_y,%sp_z,%sp_fx,%sp_fy,%sp_fz,%s_curl_thumb,%s_curl_index,%s_curl_middle,%s_curl_ring,%s_curl_pinky,%s_skel_err",s,s,s,s,s,s,s,s,s,s,s,s,s,s,s,s,s,s,s,s,s,s,s,s);
    std::fprintf(out,",mb_ok,mb_seq,mb_published,mb_source");
    for(const char* s:{"l","r"})std::fprintf(out,",mb_%s_id,mb_%s_flags,mb_%s_capture,mb_%s_until,mb_%s_wx,mb_%s_wy,mb_%s_wz,mb_%s_ix,mb_%s_iy,mb_%s_iz",s,s,s,s,s,s,s,s,s,s);
    std::fprintf(out,"\n");
    const double end=now_ms()+1000.0*std::max(5,std::min(600,std::atoi(argv[5])));
    int optical[2]={-1,-1};double inventory_at=0;unsigned rows=0;
    while(now_ms()<end){
        const double t=now_ms();
        if(t-inventory_at>1000){inventory_at=t;optical[0]=optical[1]=-1;
            for(uint32_t i=1;i<k_unMaxTrackedDeviceCount;++i){char serial[128]{};ETrackedPropertyError pe{};system->GetStringTrackedDeviceProperty(i,Prop_SerialNumber_String,serial,sizeof serial,&pe);
                if(!std::strcmp(serial,"frame-hands-optical-left"))optical[0]=int(i);if(!std::strcmp(serial,"frame-hands-optical-right"))optical[1]=int(i);}}
        TrackedDevicePose_t now_poses[k_unMaxTrackedDeviceCount]{},pred_poses[k_unMaxTrackedDeviceCount]{};
        // Raw tracking space: the same space the relay writes the mailbox in, so feed and output compare directly.
        system->GetDeviceToAbsoluteTrackingPose(TrackingUniverseRawAndUncalibrated,0.f,now_poses,k_unMaxTrackedDeviceCount);
        system->GetDeviceToAbsoluteTrackingPose(TrackingUniverseRawAndUncalibrated,0.04f,pred_poses,k_unMaxTrackedDeviceCount);
        input->UpdateActionState(&active,sizeof active,1);
        std::fprintf(out,"%.3f",t);pose_fields(out,now_poses[0]);pose_fields(out,pred_poses[0]);
        for(int side=0;side<2;++side){
            const int idx=optical[side];TrackedDevicePose_t empty{};
            std::fprintf(out,",%d",idx);pose_fields(out,idx>=0?now_poses[idx]:empty);pose_fields(out,idx>=0?pred_poses[idx]:empty);
            VRSkeletalSummaryData_t summary{};EVRInputError se=input->GetSkeletalSummaryData(skeleton[side],VRSummaryType_FromDevice,&summary);
            for(int f=0;f<5;++f)std::fprintf(out,",%.3f",se==VRInputError_None?summary.flFingerCurl[f]:-1.f);
            std::fprintf(out,",%d",int(se));
        }
        Header h{};Hand hands[2]{};const bool ok=read_mailbox(argv[3],h,hands);
        std::fprintf(out,",%d,%llu,%llu,%llu",int(ok),(unsigned long long)h.sequence,(unsigned long long)h.published,(unsigned long long)h.source);
        for(int side=0;side<2;++side){const auto& hd=hands[side];
            std::fprintf(out,",%u,%u,%llu,%llu,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f",hd.id,hd.flags,(unsigned long long)hd.capture,(unsigned long long)hd.until,hd.points[0],hd.points[1],hd.points[2],hd.points[24],hd.points[25],hd.points[26]);}
        std::fprintf(out,"\n");++rows;
        std::this_thread::sleep_for(std::chrono::milliseconds(4));
    }
    std::fclose(out);shutdown();std::printf("rows=%u optical_left=%d optical_right=%d\n",rows,optical[0],optical[1]);return 0;
}
