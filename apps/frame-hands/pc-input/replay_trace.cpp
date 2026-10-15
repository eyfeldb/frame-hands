// Replay the stability candidate against recorded measurements and HMD poses.
// No OpenVR, XR, camera, driver, or live input file is opened.
#include "trace.h"
#include "stability.h"
#include <cstdio>
#include <cmath>
using fhs::V;
V world(const float* m,V p){return {m[0]*p.x+m[1]*p.y+m[2]*p.z+m[3],m[4]*p.x+m[5]*p.y+m[6]*p.z+m[7],m[8]*p.x+m[9]*p.y+m[10]*p.z+m[11]};}
V local(const float* m,V p){p=p-V{m[3],m[7],m[11]};return {m[0]*p.x+m[4]*p.y+m[8]*p.z,m[1]*p.x+m[5]*p.y+m[9]*p.z,m[2]*p.x+m[6]*p.y+m[10]*p.z};}
int main(int argc,char** argv){
    if(argc!=3){std::puts("usage: replay-trace INPUT.bin NEW-OUTPUT.bin");return 2;}
    FILE* in=std::fopen(argv[1],"rb");if(!in)return 3;
    FILE* out=std::fopen(argv[2],"wbx");if(!out){std::fclose(in);return 4;}
    fhs::Stability stability[2];uint64_t frames=0,changed=0,last=0;double max_delta=0;int error=0;
    fhs::Trace record;
    while(size_t size=std::fread(&record,1,sizeof record,in)){
        if(size!=sizeof record||std::memcmp(record.magic,"FHTRACE1",8)||record.version!=1||record.size!=sizeof record||record.time<last||std::memcmp(record.view.magic,"FHVIEW2",8)){error=5;break;}
        last=record.time;const bool valid=record.flags&2,preview=record.flags&4;
        for(int side=0;side<2;++side){
            auto& view=record.view.hands[side];auto previous=view;view.flags&=~12u;view.candidate_id=0;
            std::memset(view.candidate,0,sizeof view.candidate);
            auto& filter=stability[side];
            if((record.head_flags&3)!=3)filter.reset();
            else if(preview){
                const auto& hand=record.mailbox.hands[side];
                if(valid&&(record.capture_flags&3)==3&&(hand.flags&fhs::Present)){
                    std::array<V,21> points{};
                    for(int i=0;i<21;++i){auto p=hand.points[i];points[i]=world(record.captured_head,{p[0],p[1],p[2]});}
                    filter.update(points,hand.id,record.mailbox.capture_ms,record.time);
                }
                if(!valid||(record.capture_flags&3)!=3)filter.reset();
                if(filter.visible(record.time)){
                    view.flags|=4;view.candidate_id=filter.id;
                    if(!(view.flags&1)||filter.stamp!=record.mailbox.capture_ms)view.flags|=8;
                    for(int i=0;i<21;++i){auto p=local(record.head,filter.value[i]);view.candidate[i][0]=float(p.x);view.candidate[i][1]=float(p.y);view.candidate[i][2]=float(p.z);}
                }
            }
            bool differs=(previous.flags&12)!=(view.flags&12)||previous.candidate_id!=view.candidate_id;
            if((previous.flags&view.flags&4)!=0)for(int i=0;i<21;++i)for(int j=0;j<3;++j){double delta=std::abs(double(previous.candidate[i][j])-view.candidate[i][j]);max_delta=std::max(max_delta,delta);differs|=delta>1e-5;}
            changed+=differs;
        }
        if(std::fwrite(&record,1,sizeof record,out)!=sizeof record){error=6;break;}++frames;
    }
    if(std::ferror(in))error=7;
    std::fclose(in);if(std::fclose(out))error=8;
    std::printf("frames=%llu changed_candidate_hand_frames=%llu max_coordinate_delta_m=%.9f error=%d\n",(unsigned long long)frames,(unsigned long long)changed,max_delta,error);
    return error?error:(frames?0:9);
}
