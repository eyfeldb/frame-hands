// Offline diagnostic: distinguish skeleton rotation jumps from landmark motion.
#define main stability_regression_main
#include "test_driver_stability.cpp"
#undef main
#include "retarget_continuity.h"
#include "retarget_chain.h"
int main(int argc,char** argv){
    assert(argc==2);assert(provider.Init(&context)==VRInitError_None);reset_devices();host.recorded=true;
    FILE* in=std::fopen(argv[1],"rb");assert(in);fhs::Trace r;
    fhs::Skeleton previous[2];std::array<fhs::V,21> points_before[2];uint64_t stamp[2]{};uint32_t id[2]{};
    fhs::RetargetContinuity continuity[2];
    uint64_t pairs=0,large=0,small_motion_large_rotation=0,model_large=0,model_small=0,world_large=0,world_small=0;double maximum=0,max_position_error=0;
    while(size_t n=std::fread(&r,1,sizeof r,in)){
        assert(n==sizeof r&&r.size==sizeof r&&!std::memcmp(r.magic,"FHTRACE1",8));
        host.capture=(r.capture_flags&3)==3;std::memcpy(host.recorded_head,r.captured_head,sizeof r.captured_head);
        tick(r.mailbox,r.time,r.flags&2,(r.head_flags&3)==3);
        for(int s=0;s<2;++s){auto& estimate=static_cast<Device*>(host.device[s])->stability;
            if(!host.pose[s].poseIsValid||stamp[s]==estimate.stamp)continue;
            auto original=fhs::retarget(estimate.value,s);auto candidate=continuity[s].update(estimate.value,s,estimate.id,estimate.stamp);
            auto expected_model=fhs::forward(original.bones),candidate_model=fhs::forward(candidate.bones);
            for(int b=0;b<31;++b)max_position_error=std::max(max_position_error,(expected_model[b].p-candidate_model[b].p).len());
#ifdef USE_RETARGET_CHAIN
            candidate=fhs::retarget_chain(estimate.value,s);
            candidate_model=fhs::forward(candidate.bones);
            for(int b=0;b<31;++b)max_position_error=std::max(max_position_error,(expected_model[b].p-candidate_model[b].p).len());
            auto sk=candidate;
#elif defined(USE_RETARGET_CONTINUITY)
            auto sk=candidate;
#else
            auto sk=original;
#endif
            if(id[s]==estimate.id&&estimate.stamp>stamp[s]&&estimate.stamp-stamp[s]<=50){
                ++pairs;double displacement=fhs::Stability::distance(estimate.value,points_before[s]);
                auto now_model=fhs::forward(sk.bones),before_model=fhs::forward(previous[s].bones);
                for(int bone=2;bone<26;++bone){auto a=sk.bones[bone].q,b=previous[s].bones[bone].q;
                    double angle=2*std::acos(std::clamp(std::abs(a.w*b.w+a.x*b.x+a.y*b.y+a.z*b.z),0.,1.))*180/3.141592653589793;
                    maximum=std::max(maximum,angle);
                    auto c=now_model[bone].q,d=before_model[bone].q;
                    double model_angle=2*std::acos(std::clamp(std::abs(c.w*d.w+c.x*d.x+c.y*d.y+c.z*d.z),0.,1.))*180/3.141592653589793;
                    if(model_angle>90){++model_large;model_small+=displacement<.02;}
                    c=sk.rotation*c;d=previous[s].rotation*d;
                    double world_angle=2*std::acos(std::clamp(std::abs(c.w*d.w+c.x*d.x+c.y*d.y+c.z*d.z),0.,1.))*180/3.141592653589793;
                    if(world_angle>90){++world_large;world_small+=displacement<.02;}
                    if(angle>90){++large;if(displacement<.02){++small_motion_large_rotation;
                        if(small_motion_large_rotation<=12)std::printf("rotation_jump side=%d t_ms=%llu bone=%d degrees=%.2f max_joint_delta_mm=%.3f\n",s,(unsigned long long)r.time,bone,angle,1000*displacement);
                    }}
                }
            }
            previous[s]=sk;points_before[s]=estimate.value;stamp[s]=estimate.stamp;id[s]=estimate.id;
        }
    }assert(!std::ferror(in));std::fclose(in);
    std::printf("adjacent_observation_pairs=%llu bone_rotation_jumps_over_90deg=%llu under_20mm_joint_motion=%llu max_bone_delta_degrees=%.3f\n",(unsigned long long)pairs,(unsigned long long)large,(unsigned long long)small_motion_large_rotation,maximum);
    std::printf("model_rotation_jumps_over_90deg=%llu under_20mm_joint_motion=%llu max_preserved_joint_position_error_m=%.12f\n",(unsigned long long)model_large,(unsigned long long)model_small,max_position_error);
    std::printf("world_rotation_jumps_over_90deg=%llu under_20mm_joint_motion=%llu\n",(unsigned long long)world_large,(unsigned long long)world_small);
    assert(max_position_error<1e-9);
    for(auto* d:host.device)d->Deactivate();provider.Cleanup();return 0;
}
