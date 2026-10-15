// Explain driver-visible loss, without inventing camera/model ground truth.
#define main stability_regression_main
#include "test_driver_stability.cpp"
#undef main
#include <map>
int main(int argc,char** argv){
    assert(argc==2);assert(provider.Init(&context)==VRInitError_None);reset_devices();host.recorded=true;
    FILE* in=std::fopen(argv[1],"rb");assert(in);fhs::Trace r;
    bool started[2]{};uint64_t absent_since[2]{},max_gap[2]{},completed[2]{};
    std::map<std::string,uint64_t> reasons[2];
    while(size_t n=std::fread(&r,1,sizeof r,in)){
        assert(n==sizeof r&&r.size==sizeof r&&!std::memcmp(r.magic,"FHTRACE1",8));
        host.capture=(r.capture_flags&3)==3;std::memcpy(host.recorded_head,r.captured_head,sizeof r.captured_head);
        tick(r.mailbox,r.time,r.flags&2,(r.head_flags&3)==3);
        for(int s=0;s<2;++s){
            if(host.pose[s].poseIsValid){
                started[s]=true;
                if(absent_since[s]){++completed[s];max_gap[s]=std::max(max_gap[s],r.time-absent_since[s]);absent_since[s]=0;}
                continue;
            }
            if(!started[s])continue;
            if(!absent_since[s])absent_since[s]=r.time;
            std::string reason;
            if((r.head_flags&3)!=3)reason="current_hmd_unavailable";
            else if(!(r.flags&2))reason="stale_transport";
            else if(!host.capture)reason="capture_hmd_unavailable";
            else if(!(r.mailbox.hands[s].flags&fhs::Present)){
                if(r.incoming.sequence==r.mailbox.sequence&&(r.incoming.hands[s].flags&fhs::Present))reason=std::string("bridge_")+fhs::hand_issue_name(fhs::hand_issue(r.incoming.hands[s]));
                else reason="source_hand_absent";
            }else if(r.time-r.mailbox.capture_ms>fhs::Stability::hold_ms)reason="observation_older_than_80ms";
            else reason="filter_or_geometry_rejected";
            ++reasons[s][reason];
        }
    }assert(!std::ferror(in));std::fclose(in);
    for(int s=0;s<2;++s){std::printf("side=%d completed_loss_episodes=%llu longest_completed_ms=%llu open_loss_at_end=%d\n",s,(unsigned long long)completed[s],(unsigned long long)max_gap[s],bool(absent_since[s]));for(const auto& item:reasons[s])std::printf("  %s=%llu sampler_records\n",item.first.c_str(),(unsigned long long)item.second);}
    for(auto* d:host.device)d->Deactivate();provider.Cleanup();
}
