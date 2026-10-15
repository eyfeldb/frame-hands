// Exercise the real driver code with deterministic observation/host time.
#define main stability_regression_main
#include "test_driver_stability.cpp"
#undef main
int main(int argc,char** argv){
    assert(argc==2);
    assert(provider.Init(&context)==VRInitError_None);reset_devices();
    auto p=example();tick(p,1000);auto before=host.pose[0];
    tick(p,1061);assert(!host.pose[0].poseIsValid);
#ifdef FRAME_HANDS_DRIVER_ROLE_GRACE
    assert(host.pose[0].deviceIsConnected&&properties.roles[5]==TrackedControllerRole_LeftHand);
    assert(host.pose[0].result==TrackingResult_Running_OutOfRange);
    for(int i=0;i<3;++i)assert(host.pose[0].vecPosition[i]==before.vecPosition[i]);
    tick(p,1730,false);assert(host.pose[0].deviceIsConnected&&!host.pose[0].poseIsValid);
    tick(p,1731,false);assert(!host.pose[0].deviceIsConnected&&properties.roles[5]==TrackedControllerRole_OptOut);
    reset_devices();p=example();tick(p,1000);tick(p,1001,true,true,true);
    assert(!host.pose[0].deviceIsConnected);tick(p,1100,false);assert(!host.pose[0].deviceIsConnected);
    reset_devices();tick(p,1000);tick(p,1001,true,false);assert(!host.pose[0].deviceIsConnected);
    reset_devices();p=example();tick(p,1000);p.hands[0]={};p.capture_ms=1090;tick(p,1100);
    assert(!host.pose[0].poseIsValid&&host.pose[1].poseIsValid&&host.pose[0].deviceIsConnected);
    p=example();p.capture_ms=1110;tick(p,1120);assert(host.pose[0].poseIsValid);
#else
    (void)before;assert(!host.pose[0].deviceIsConnected);
#endif
    reset_devices();host.recorded=true;
    FILE* in=std::fopen(argv[1],"rb");assert(in);fhs::Trace r;
    uint64_t frames=0,valid[2]{},changes[2]{},connected[2]{},hash=14695981039346656037ULL;
    bool last[2]{};
    while(size_t n=std::fread(&r,1,sizeof r,in)){
        assert(n==sizeof r&&!std::memcmp(r.magic,"FHTRACE1",8)&&r.version==1&&r.size==sizeof r);
        host.capture=(r.capture_flags&3)==3;std::memcpy(host.recorded_head,r.captured_head,sizeof r.captured_head);
        // Counterfactual replay: controller takeover events were not recorded.
        tick(r.mailbox,r.time,r.flags&2,(r.head_flags&3)==3);
        for(int s=0;s<2;++s){auto pose=host.pose[s];valid[s]+=pose.poseIsValid;connected[s]+=pose.deviceIsConnected;
            changes[s]+=last[s]!=pose.deviceIsConnected;last[s]=pose.deviceIsConnected;
            hash=(hash^uint64_t(pose.poseIsValid))*1099511628211ULL;
            if(pose.poseIsValid){auto points=published(s);for(auto v:points)for(double value:{v.x,v.y,v.z}){
                auto bytes=reinterpret_cast<const unsigned char*>(&value);for(size_t i=0;i<sizeof value;++i)hash=(hash^bytes[i])*1099511628211ULL;
            }}
        }++frames;
    }assert(!std::ferror(in)&&frames);std::fclose(in);
    std::printf("frames=%llu valid=%llu,%llu connected=%llu,%llu role_transitions=%llu,%llu valid_pose_hash=%llu\n",
        (unsigned long long)frames,(unsigned long long)valid[0],(unsigned long long)valid[1],
        (unsigned long long)connected[0],(unsigned long long)connected[1],
        (unsigned long long)changes[0],(unsigned long long)changes[1],(unsigned long long)hash);
    for(auto* d:host.device)d->Deactivate();provider.Cleanup();return 0;
}
