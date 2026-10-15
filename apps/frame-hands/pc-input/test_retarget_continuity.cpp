#include "retarget_continuity.h"
#undef NDEBUG
#include <cassert>
#include <cstdio>
double similarity(fhs::Q a,fhs::Q b){return std::abs(a.w*b.w+a.x*b.x+a.y*b.y+a.z*b.z);}
int main(){
    using namespace fhs;
    for(int side=0;side<2;++side){
        std::array<Transform,31> rest_pose{};for(int b=0;b<31;++b)rest_pose[b]={rest[side][b].p,rest[side][b].q};
        auto bind=forward(rest_pose);std::array<V,21> base{};
        for(int i=0;i<21;++i)base[i]=bind[landmark_bone[i]].p;
        RetargetContinuity candidate;auto first=candidate.update(base,side,1,1000);auto first_model=forward(first.bones);
        for(int step=1;step<=500;++step){
            double angle=.02*step;V axis=V{.3,.7,.5}.unit();Q turn{std::cos(angle/2),axis.x*std::sin(angle/2),axis.y*std::sin(angle/2),axis.z*std::sin(angle/2)};
            std::array<V,21> points{};for(int i=0;i<21;++i)points[i]=turn.rotate(base[i])+V{.1,.3,-.5};
            auto out=candidate.update(points,side,1,1000+33*step);auto model=forward(out.bones);
            for(int i=0;i<21;++i)assert((out.position+out.rotation.rotate(model[landmark_bone[i]].p)-points[i]).len()<1e-9);
            for(int b=1;b<31;++b)assert(similarity(out.rotation*model[b].q,turn*first.rotation*first_model[b].q)>1-1e-9);
        }
        auto fresh=retarget(base,side);auto reset=candidate.update(base,side,2,18000);
        for(int b=0;b<31;++b)assert(similarity(fresh.bones[b].q,reset.bones[b].q)>1-1e-9);
        auto repeat=candidate.update(base,side,2,18000);for(int b=0;b<31;++b)assert(similarity(reset.bones[b].q,repeat.bones[b].q)>1-1e-9);
        auto after_gap=candidate.update(base,side,2,19000);for(int b=0;b<31;++b)assert(similarity(fresh.bones[b].q,after_gap.bones[b].q)>1-1e-9);
    }
    std::puts("PASS retarget continuity: measured joints preserved, rigid hand rotation preserved over 500 frames per side, identity/gap reset, duplicate idempotence");
}
