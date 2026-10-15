#define FRAME_HANDS_DRIVER_PRESENTATION
#define FRAME_HANDS_DRIVER_ROLE_GRACE
#define FRAME_HANDS_DRIVER_INPUT_MODE
#define main stability_regression_main
#include "test_driver_stability.cpp"
#undef main
#include <vector>
double degrees(fhs::Q a,fhs::Q b){return 2*std::acos(std::clamp(std::abs(a.w*b.w+a.x*b.x+a.y*b.y+a.z*b.z),0.,1.))*180/3.141592653589793;}
double percentile(std::vector<double> x,double p){assert(!x.empty());std::sort(x.begin(),x.end());return x[size_t((x.size()-1)*p)];}
int main(int argc,char** argv){
    assert(argc>1);assert(provider.Init(&context)==VRInitError_None);
    for(int file=1;file<argc;++file){
        reset_devices();host.recorded=true;FILE* in=std::fopen(argv[file],"rb");assert(in);fhs::Trace r;
        fhs::Skeleton previous[2]{},raw_previous[2]{};uint32_t ids[2]{};uint64_t last_time[2]{};
        std::vector<double> steps,raw_steps,angles,raw_angles,offsets;uint64_t frames=0,visible=0,gaps=0;
        while(size_t n=std::fread(&r,1,sizeof r,in)){
            assert(n==sizeof r&&r.size==sizeof r&&!std::memcmp(r.magic,"FHTRACE1",8));++frames;
            host.capture=(r.capture_flags&3)==3;std::memcpy(host.recorded_head,r.captured_head,sizeof r.captured_head);
            tick(r.mailbox,r.time,r.flags&2,(r.head_flags&3)==3);
            for(int s=0;s<2;++s){
                auto* device=static_cast<Device*>(host.device[s]);auto& estimate=device->stability;
                bool expected=(r.flags&2)&&(r.head_flags&3)==3&&(r.capture_flags&3)==3&&estimate.visible(r.time);
                assert(host.pose[s].poseIsValid==expected); // presentation never extends availability
                if(!expected){++gaps;ids[s]=0;assert(device->presentation.id==0);continue;}++visible;
                auto raw=fhs::retarget(estimate.value,s),out=device->presentation.value;
                auto output_points=published(s);auto model=fhs::forward(out.bones);
                for(int b=0;b<31;++b)assert(model[b].p.finite());
                for(int i=0;i<21;++i)assert((output_points[i]-(out.position+out.rotation.rotate(model[fhs::landmark_bone[i]].p))).len()<1e-6);
                offsets.push_back((out.position-raw.position).len()*1000);
                if(ids[s]==estimate.id&&r.time-last_time[s]<=40){
                    steps.push_back((out.position-previous[s].position).len()*1000);
                    raw_steps.push_back((raw.position-raw_previous[s].position).len()*1000);
                    auto prev_model=fhs::forward(previous[s].bones),raw_model=fhs::forward(raw.bones),old_raw=fhs::forward(raw_previous[s].bones);
                    double a=0,b=0;for(int bone=2;bone<26;++bone){a=std::max(a,degrees(out.rotation*model[bone].q,previous[s].rotation*prev_model[bone].q));b=std::max(b,degrees(raw.rotation*raw_model[bone].q,raw_previous[s].rotation*old_raw[bone].q));}
                    angles.push_back(a);raw_angles.push_back(b);
                }
                ids[s]=estimate.id;last_time[s]=r.time;previous[s]=out;raw_previous[s]=raw;
            }
        }assert(!std::ferror(in));std::fclose(in);
        std::printf("trace=%s frames=%llu visible=%llu absent=%llu availability_mismatch=0 wrist_step_p95_mm=%.4f->%.4f bone_world_step_p95_deg=%.4f->%.4f additional_wrist_offset_p50_p95_mm=%.4f/%.4f\n",argv[file],(unsigned long long)frames,(unsigned long long)visible,(unsigned long long)gaps,percentile(raw_steps,.95),percentile(steps,.95),percentile(raw_angles,.95),percentile(angles,.95),percentile(offsets,.5),percentile(offsets,.95));
    }
    for(auto* d:host.device)d->Deactivate();provider.Cleanup();
}
