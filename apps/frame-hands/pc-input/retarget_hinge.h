#pragma once
#include "retarget_chain.h"
#include <cmath>
namespace fhs {
// Anatomical finger rotations (2026-10-12). The swing retarget reproduces every measured joint position, but a
// "minimal rotation from the rest direction" also carries whatever roll the measurement noise implies, and a
// position-free rig (Maestro re-poses its own glove from bone ROTATIONS only, while SteamVR's dashboard glove is
// skinned to our positions) shows that roll as warped fingers. Here every joint rotates about fixed anatomical
// axes taken from the reference pose: fingers flex about ONE lateral axis per finger (the knuckle also abducts
// about the palm normal), the thumb flexes about its own plane normal (the base also abducts). No twist can appear.
// Positions stay measured, so the dashboard glove is unchanged.
inline Q axis_angle(V axis,double angle){const double s=std::sin(angle/2);return Q{std::cos(angle/2),axis.x*s,axis.y*s,axis.z*s}.unit();}
// Signed angle that rotates l onto d about the unit axis a, using the components perpendicular to a.
inline double hinge_angle(V a,V l,V d){
    V lp=l-a*l.dot(a),dp=d-a*d.dot(a);
    if(lp.len()<1e-6||dp.len()<1e-6)return 0;
    return std::atan2(a.cross(lp).dot(dp),lp.dot(dp));
}
inline Skeleton retarget_hinge(const std::array<V,21>& points,int side,const V* forearm=nullptr,const Q* device_fix=nullptr){
    auto result=retarget(points,side,forearm,device_fix);
    auto target=forward(result.bones);
    std::array<Transform,31> rest_local{};
    for(int b=0;b<31;++b)rest_local[b]={rest[side][b].p,rest[side][b].q};
    auto bind=forward(rest_local);
    std::array<V,21> reference{};
    for(int i=0;i<21;++i)reference[i]=bind[landmark_bone[i]].p;
    // Reference-pose axes (device frame): palm normal, one lateral axis per finger, the thumb plane normal.
    V y=(reference[9]-reference[0]).unit();V x=(reference[5]-reference[17]).unit();const V palm_normal=x.cross(y).unit();
    V t2=(bind[3].p-bind[2].p).unit(),t3=(bind[4].p-bind[3].p).unit();V thumb_normal=t2.cross(t3);
    if(thumb_normal.len()<.05)thumb_normal=t2.cross(palm_normal.cross(t2));   // straight reference thumb: its plane with the palm
    thumb_normal=thumb_normal.unit();
    V lateral[5]{};
    for(int f=1;f<5;++f){const int prox=2+5*f;lateral[f]=(bind[prox+1].p-bind[prox].p).unit().cross(palm_normal).unit();}
    for(int b=2;b<26;++b){
        const bool tip=b==5||b==10||b==15||b==20||b==25;
        if(tip){target[b].q=target[b-1].q;continue;}
        const int parent=parents[b],finger=(b-1)/5;      // 0 thumb, 1 index .. 4 pinky
        const bool thumb=finger==0,metacarpal=!thumb&&b==1+5*finger;
        const bool two_dof=b==2||metacarpal||b==2+5*finger;   // thumb base, finger metacarpals, knuckles
        const Q neutral=(target[parent].q*rest_local[b].q).unit();                      // parent carries the bone's rest frame
        const Q inv=bind[b].q.inv();
        const V l=inv.rotate(bind[b+1].p-bind[b].p).unit();                             // rest direction, bone frame
        V A,B;                                                                          // abduction axis, flexion axis (bone frame)
        if(thumb){B=inv.rotate(thumb_normal).unit();A=l.cross(B).unit();}
        else{A=inv.rotate(palm_normal).unit();B=metacarpal?l.cross(A).unit():inv.rotate(lateral[finger]).unit();}
        const V d=neutral.inv().rotate(target[b+1].p-target[b].p).unit();               // measured direction, neutral frame
        if(!d.finite()||d.len()<.5)continue;
        const double alpha=two_dof?hinge_angle(A,l,d):0;
        const Q ra=axis_angle(A,alpha);
        const double beta=hinge_angle(ra.rotate(B),ra.rotate(l),d);
        target[b].q=(neutral*ra*axis_angle(B,beta)).unit();
    }
    for(int f=0;f<5;++f)target[26+f]=target[4+5*f];
    for(int b=0;b<31;++b){int parent=parents[b];result.bones[b]=parent<0?target[b]:Transform{target[parent].q.inv().rotate(target[b].p-target[parent].p),(target[parent].q.inv()*target[b].q).unit()};}
    return result;
}
}
