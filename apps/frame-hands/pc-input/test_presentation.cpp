#include "presentation.h"
#include "retarget_chain.h"
#undef NDEBUG
#include <cassert>
#include <cstdio>
using namespace fhs;
double angle(Q a,Q b){return 2*std::acos(std::clamp(std::abs(a.w*b.w+a.x*b.x+a.y*b.y+a.z*b.z),0.,1.));}
int main(){
    double baseline_jump=0,chain_jump=0;
    for(int side=0;side<2;++side){
        std::array<Transform,31> local{};for(int b=0;b<31;++b)local[b]={rest[side][b].p,rest[side][b].q};
        auto bind=forward(local);std::array<V,21> base{};
        for(int i=0;i<21;++i)base[i]=bind[landmark_bone[i]].p;
        auto rest_result=retarget_chain(base,side);auto rest_model=forward(rest_result.bones);
        Skeleton previous{},previous_original{};
        // Closed flexion cycle with distal directions passing through 180 degrees.
        // Positions are generated geometrically; no candidate output feeds them.
        for(int step=0;step<=1000;++step){
            double curl=(1-std::cos(6.283185307179586*step/1000.))*.85;
            auto points=base;V axis=(base[5]-base[17]).unit();
            for(int f=0;f<4;++f)for(int j=1;j<=3;++j){
                int n=5+4*f+j;double a=curl*j;
                Q q{std::cos(a/2),axis.x*std::sin(a/2),axis.y*std::sin(a/2),axis.z*std::sin(a/2)};
                points[n]=points[n-1]+q.rotate(base[n]-base[n-1]);
            }
            auto candidate=retarget_chain(points,side),original=retarget(points,side);
            auto model=forward(candidate.bones),reference_model=forward(original.bones);
            for(int b=0;b<31;++b)assert((model[b].p-reference_model[b].p).len()<1e-9);
            if(step)for(int b=2;b<26;++b){
                chain_jump=std::max(chain_jump,angle(candidate.bones[b].q,previous.bones[b].q));
                baseline_jump=std::max(baseline_jump,angle(original.bones[b].q,previous_original.bones[b].q));
            }
            // Equivariance under arbitrary rigid wrist movement.
            V axis2=V{.3,.7,.5}.unit();double a=step*.02;
            Q turn{std::cos(a/2),axis2.x*std::sin(a/2),axis2.y*std::sin(a/2),axis2.z*std::sin(a/2)};
            auto rotated=points;for(auto& v:rotated)v=turn.rotate(v)+V{.1,.3,-.5};
            auto rotated_result=retarget_chain(rotated,side);auto rotated_model=forward(rotated_result.bones);
            for(int b=0;b<31;++b){
                assert(angle(rotated_result.rotation*rotated_model[b].q,turn*candidate.rotation*model[b].q)<1e-6);
                assert((rotated_result.position+rotated_result.rotation.rotate(rotated_model[b].p)-(turn.rotate(candidate.position+candidate.rotation.rotate(model[b].p))+V{.1,.3,-.5})).len()<1e-9);
            }
            if(step==1000)for(int b=0;b<31;++b)assert(angle(model[b].q,rest_model[b].q)<1e-6);
            previous=candidate;previous_original=original;
        }
        Presentation p;auto first=p.update(rest_result,1,1000,1000);
        auto goal=first;goal.position.x+=.1;goal.rotation=Q{0,0,1,0}*goal.rotation;
        for(int step=1;step<=10;++step){
            auto out=p.update(goal,1,1000+step*10,1000+step*10);
            assert(out.position.x>first.position.x&&out.position.x<goal.position.x);
            auto model=forward(out.bones);
            for(int b=0;b<31;++b){assert(model[b].p.finite());assert(std::abs(model[b].q.w*model[b].q.w+model[b].q.x*model[b].q.x+model[b].q.y*model[b].q.y+model[b].q.z*model[b].q.z-1)<1e-9);}
            for(int f=0;f<5;++f)assert((model[26+f].p-model[4+5*f].p).len()<1e-9);
        }
        auto before_repeat=p.value;auto repeat=p.update(goal,1,1100,1100);assert((repeat.position-before_repeat.position).len()==0);
        auto reset=p.update(first,2,1101,1101);assert((reset.position-first.position).len()==0);
        reset=p.update(goal,2,1200,1200);assert((reset.position-goal.position).len()==0);
        reset=p.update(first,2,1190,1201);assert((reset.position-first.position).len()==0);
        p.reset();reset=p.update(goal,2,1202,1202);assert((reset.position-goal.position).len()==0);
        // Opposing translations interpolate direction without collapsing length.
        assert(std::abs(blend_segment({1,0,0},{-1,0,0},.5).len()-1)<1e-9);
        assert(angle(blend_rotation({1,0,0,0},{-1,0,0,0},.5),{})<1e-9);
        // 0.5 m/s at 30 Hz optical / 100 Hz driver. Additional delay relative
        // to the latest optical target stays bounded, including between samples.
        p.reset();double offset_sum=0;int measured=0;auto moving=rest_result;
        for(uint64_t now=1000;now<=3000;now+=10){
            uint64_t capture=1000+((now-1000)/33)*33;
            moving.position.x=rest_result.position.x+.0005*(capture-1000);
            auto out=p.update(moving,1,capture,now);
            if(now>=2000){double offset=moving.position.x-out.position.x;assert(offset>=0&&offset<.017);offset_sum+=offset;++measured;}
        }
        std::printf("constant_motion_side=%d additional_delay_mean_ms=%.3f\n",side,offset_sum/measured/.0005);
    }
    std::printf("PASS chain: 1001 closed flexion/rigid poses per side, all 31 positions preserved, no cycle drift. Maximum flexion local step original=%.6f chain=%.6f degrees\n",baseline_jump*180/3.141592653589793,chain_jump*180/3.141592653589793);
    assert(chain_jump<.05);
    std::puts("PASS presentation: bounded response, unit rotations, auxiliary knuckles, identity/time/gap/reset, opposing-vector lengths, quaternion sign");
}
