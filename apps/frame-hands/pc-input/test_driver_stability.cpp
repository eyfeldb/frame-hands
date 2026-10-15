// Compile the actual candidate driver with the mock server and recorded clock/poses.
// No SteamVR, camera, registered bundle or input.bin is opened.
#define FRAME_HANDS_DRIVER_MOCKS_ONLY
#define FRAME_HANDS_DRIVER_STABILITY
#include "test_driver.cpp"
#include "driver.cpp"
#include "trace.h"

fhs::Packet example(){
    fhs::Packet p;p.sequence=1;p.sent_ms=1000;p.capture_ms=980;
    for(int side=0;side<2;++side){
        std::array<fhs::Transform,31> local{};
        for(int i=0;i<31;++i)local[i]={fhs::rest[side][i].p,fhs::rest[side][i].q};
        auto bones=fhs::forward(local);auto& h=p.hands[side];h.id=side+1;h.flags=fhs::Present|fhs::Stereo;
        for(int i=0;i<21;++i){auto v=bones[fhs::landmark_bone[i]].p+fhs::V{side?.2:-.2,0,-.5};h.points[i][0]=float(v.x);h.points[i][1]=float(v.y);h.points[i][2]=float(v.z);}
    }return p;
}
void tick(const fhs::Packet& packet,uint64_t now,bool fresh=true,bool head=true,bool busy=false){
    for(auto* device:host.device)static_cast<Device*>(device)->frame(packet,fresh,head,busy,now);
    assert(!input.any(5)&&!input.any(6)); // skeleton-only regardless of packet control options
}
void reset_devices(){for(int side=0;side<2;++side){host.device[side]->Deactivate();assert(host.device[side]->Activate(side+4)==VRInitError_None);}}
fhs::V world(const float* m,fhs::V v){return {m[0]*v.x+m[1]*v.y+m[2]*v.z+m[3],m[4]*v.x+m[5]*v.y+m[6]*v.z+m[7],m[8]*v.x+m[9]*v.y+m[10]*v.z+m[11]};}
std::array<fhs::V,21> published(int side){
    std::array<fhs::Transform,31> transforms{};bool found=false;
    for(const auto& item:input.bones)if(input.containers[item.first]==uint64_t(side+5)){
        for(int i=0;i<31;++i){const auto& b=item.second[i];transforms[i]={{b.position.v[0],b.position.v[1],b.position.v[2]},{b.orientation.w,b.orientation.x,b.orientation.y,b.orientation.z}};}found=true;
    }assert(found);auto bones=fhs::forward(transforms);auto p=host.pose[side];
    fhs::Q q{p.qRotation.w,p.qRotation.x,p.qRotation.y,p.qRotation.z};fhs::V position{p.vecPosition[0],p.vecPosition[1],p.vecPosition[2]};
    std::array<fhs::V,21> result{};for(int i=0;i<21;++i)result[i]=position+q.rotate(bones[fhs::landmark_bone[i]].p);return result;
}
int main(int argc,char** argv){
    std::setvbuf(stdout,nullptr,_IONBF,0);
    assert(provider.Init(&context)==VRInitError_None);reset_devices();
    auto p=example();p.options=fhs::Controls;tick(p,1000);assert(host.pose[0].poseIsValid&&host.pose[1].poseIsValid);
    p.hands[0]={};p.sequence++;p.capture_ms=p.sent_ms=1030;tick(p,1030);
    assert(host.pose[0].poseIsValid&&host.pose[1].poseIsValid); // independently hold missing left
    tick(p,1060);assert(host.pose[0].poseIsValid);tick(p,1061);assert(!host.pose[0].poseIsValid&&host.pose[1].poseIsValid);
    tick(p,1111);assert(!host.pose[1].poseIsValid); // duplicate packet never extends 80 ms
    reset_devices();p=example();tick(p,1000);tick(p,1001,true,false);assert(!host.pose[0].poseIsValid);
    tick(p,1002);assert(host.pose[0].poseIsValid);tick(p,1003,true,true,true);assert(!host.pose[0].poseIsValid);
    tick(p,1004);tick(p,1005,false);assert(!host.pose[0].poseIsValid);
    host.recorded=true;host.capture=false;tick(p,1006);assert(!host.pose[0].poseIsValid);host.recorded=false;
    reset_devices();p=example();tick(p,1000);
    // Large one-frame displacement is quarantined; it cannot create a click.
    auto before=published(0);p.capture_ms=p.sent_ms=1013;p.sequence++;
    for(auto& point:p.hands[0].points)point[0]+=.6f;
    tick(p,1033);auto after=published(0);assert((before[0]-after[0]).len()<1e-6);
    p.capture_ms=p.sent_ms=1046;p.sequence++;tick(p,1066);after=published(0);assert((before[0]-after[0]).len()>.1);
    // A new identity starts cleanly; no interpolation across physical hands.
    p.hands[0].id=55;p.capture_ms=p.sent_ms=1079;tick(p,1099);
    assert(static_cast<Device*>(host.device[0])->stability.id==55);
    std::puts("PASS candidate driver: bounded independent hold, duplicate expiry, headset/capture/stale/handoff reset, spike quarantine, identity reset, no synthesized controls");
    for(int file=1;file<argc;++file){
        reset_devices();host.recorded=true;FILE* in=std::fopen(argv[file],"rb");assert(in);
        uint64_t frames=0,visible=0,visibility_mismatch=0;double max_delta=0;fhs::Trace r;
        while(size_t n=std::fread(&r,1,sizeof r,in)){
            assert(n==sizeof r&&!std::memcmp(r.magic,"FHTRACE1",8)&&r.version==1&&r.size==sizeof r&&(r.flags&4));
            host.capture=(r.capture_flags&3)==3;std::memcpy(host.recorded_head,r.captured_head,sizeof r.captured_head);
            tick(r.mailbox,r.time,r.flags&2,(r.head_flags&3)==3);
            for(int side=0;side<2;++side){bool expected=r.view.hands[side].flags&4,actual=host.pose[side].poseIsValid;visibility_mismatch+=actual!=expected;
                if(actual&&expected){++visible;auto points=published(side);for(int i=0;i<21;++i){auto v=r.view.hands[side].candidate[i];auto want=world(r.head,{v[0],v[1],v[2]});max_delta=std::max(max_delta,(points[i]-want).len());}}
            }++frames;
        }assert(!std::ferror(in));std::fclose(in);
        std::printf("trace=%s frames=%llu visible_hands=%llu visibility_mismatch=%llu max_joint_delta_m=%.9f\n",argv[file],(unsigned long long)frames,(unsigned long long)visible,(unsigned long long)visibility_mismatch,max_delta);
        assert(frames&&visibility_mismatch==0&&max_delta<.00001);
    }
    for(auto* d:host.device)d->Deactivate();provider.Cleanup();
    return 0;
}
