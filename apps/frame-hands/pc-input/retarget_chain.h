#pragma once
#include "skeleton.h"
namespace fhs {
// Determine each segment's orientation relative to its already oriented parent.
// Unlike independent rest-to-measured swings, a curled distal bone does not
// approach the rest direction's antipode. Stateless: no temporal roll drift.
inline Skeleton retarget_chain(const std::array<V,21>& points,int side,const V* forearm=nullptr,const Q* device_fix=nullptr){
    auto result=retarget(points,side,forearm,device_fix);
    auto target=forward(result.bones);
    std::array<Transform,31> rest_local{};
    for(int b=0;b<31;++b)rest_local[b]={rest[side][b].p,rest[side][b].q};
    auto bind=forward(rest_local);
    for(int b=2;b<26;++b){
        bool tip=b==5||b==10||b==15||b==20||b==25;
        if(tip){target[b].q=target[b-1].q;continue;}
        int parent=parents[b];
        Q inherited=(target[parent].q*rest_local[b].q).unit();
        V local_direction=bind[b].q.inv().rotate(bind[b+1].p-bind[b].p);
        target[b].q=(swing(inherited.rotate(local_direction),target[b+1].p-target[b].p)*inherited).unit();
    }
    for(int f=0;f<5;++f)target[26+f]=target[4+5*f];
    for(int b=0;b<31;++b){int parent=parents[b];result.bones[b]=parent<0?target[b]:Transform{target[parent].q.inv().rotate(target[b].p-target[parent].p),(target[parent].q.inv()*target[b].q).unit()};}
    return result;
}
}
