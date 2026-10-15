#pragma once
#include "protocol.h"
#include "reference.h"
namespace fhs {
constexpr int parents[31]={-1,0,1,2,3,4,1,6,7,8,9,1,11,12,13,14,1,16,17,18,19,1,21,22,23,24,0,0,0,0,0};
constexpr int landmark_bone[21]={1,2,3,4,5,7,8,9,10,12,13,14,15,17,18,19,20,22,23,24,25};
struct Transform {V p;Q q;};
struct Skeleton {V position;Q rotation;std::array<Transform,31> bones;};
inline std::array<Transform,31> forward(const std::array<Transform,31>& local){
    std::array<Transform,31> result{};
    for(int i=0;i<31;++i){auto t=local[i];int p=parents[i];result[i]=p<0?t:Transform{result[p].p+result[p].q.rotate(t.p),result[p].q*t.q};}
    return result;
}
// forearm (optional, world frame, unit vector from the elbow toward the wrist): the device pose then follows the
// FOREARM instead of the palm, and the wrist bone carries the flexion between them. Games draw the sleeve/forearm
// from the controller pose (as with a real controller in the fist); with a palm-aligned device pose every wrist
// flexion bent the drawn forearm instead of the wrist, which the wearer saw as crooked wrists (2026-10-12).
// device_fix (optional): a constant rotation applied to the device pose in the MEASURED PALM's own axes (x across the
// palm toward the index side, y wrist->middle MCP, z palm normal). The wrist bone carries the inverse, so the drawn
// hand does not move; only where a game hangs its controller-attached geometry (Maestro's sleeve) changes.
inline Skeleton retarget(const std::array<V,21>& points,int side,const V* forearm=nullptr,const Q* device_fix=nullptr){
    std::array<Transform,31> local{},target{};
    for(int i=0;i<31;++i)local[i]={rest[side][i].p,rest[side][i].q};
    auto bind=forward(local);
    std::array<V,21> reference{};
    for(int i=0;i<21;++i)reference[i]=bind[landmark_bone[i]].p;
    const Q palm_rotate=palm(points)*palm(reference).inv();
    Q rotate=palm_rotate;Q wrist_extra{};
    if(device_fix&&!forearm){
        rotate=(palm(points)*(*device_fix)*palm(reference).inv()).unit();
        wrist_extra=(rotate.inv()*palm_rotate).unit();
    }
    if(forearm&&forearm->finite()&&forearm->len()>.5){
        // palm frame: y = wrist->middle MCP, x = pinky->index (orthogonalised), z = palm normal. Replace y by the
        // forearm direction, keep the palm normal, unless the wrist is flexed so far that they nearly coincide.
        V y=forearm->unit();V zp=(points[5]-points[17]).unit().cross((points[9]-points[0]).unit()).unit();
        if(std::abs(y.dot(zp))<.9){V x=y.cross(zp).unit();V z=x.cross(y).unit();
            rotate=basis(x,y,z)*palm(reference).inv();
            wrist_extra=(rotate.inv()*palm_rotate).unit();   // palm relative to the forearm-aligned device frame
        }
    }
    Skeleton out{points[0],rotate,{}};
    // Express in the wrist device frame. All 21 measured positions are retained.
    for(int i=0;i<21;++i)target[landmark_bone[i]].p=rotate.inv().rotate(points[i]-points[0]);
    for(int f=0;f<4;++f){
        int b=6+5*f;
        V ref=bind[b+1].p-bind[1].p,measured=target[b+1].p;
        double scale=measured.len()/ref.len();
        target[b].p=swing(ref,measured).rotate(bind[b].p-bind[1].p)*scale;
    }
    target[0]={};target[1].q=(wrist_extra*bind[1].q).unit();
    for(int i=2;i<26;++i){
        bool tip=i==5||i==10||i==15||i==20||i==25;
        int a=tip?i-1:i,b=tip?i:i+1;
        target[i].q=tip?target[i-1].q:(swing(bind[b].p-bind[a].p,target[b].p-target[a].p)*bind[i].q).unit();
    }
    // Valve auxiliary transforms are the LAST KNUCKLE, not the fingertip.
    for(int f=0;f<5;++f)target[26+f]=target[4+5*f];
    for(int i=0;i<31;++i){int p=parents[i];out.bones[i]=p<0?target[i]:Transform{target[p].q.inv().rotate(target[i].p-target[p].p),(target[p].q.inv()*target[i].q).unit()};}
    return out;
}
struct Smooth {
    uint32_t id=0;std::array<V,21> value{};
    void reset(){id=0;}
    std::array<V,21> update(const std::array<V,21>& p,uint32_t next,double dt){
        if(id!=next){id=next;value=p;return value;}
        constexpr int par[21]={0,0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19};
        double blend=1-std::exp(-std::clamp(dt,0.,.1)/.020);
        auto before=value;value[0]=before[0]*(1-blend)+p[0]*blend;
        for(int i=1;i<21;++i){V a=before[i]-before[par[i]],b=p[i]-p[par[i]];
            double d=std::clamp(a.unit().dot(b.unit()),-1.,1.);double angle=std::acos(d);V direction;
            if(angle<1e-6)direction=b.unit();
            else if(d<-.9999)direction=(a.unit()*(1-blend)+b.unit()*blend).unit();
            else direction=(a.unit()*std::sin((1-blend)*angle)+b.unit()*std::sin(blend*angle))*(1/std::sin(angle));
            value[i]=value[par[i]]+direction*(a.len()*(1-blend)+b.len()*blend);
        }
        return value;
    }
};
struct Gate {
    bool armed=false,down=false;
    bool update(bool valid,double value,double on=.8,double off=.5){
        if(!valid){armed=down=false;return false;}
        if(value<off){armed=true;down=false;}
        else if(armed&&value>=on)down=true;
        return down;
    }
};
struct ControlsState {
    Gate pinch[4],grip;uint32_t id=0;uint64_t menu_start=0;
    uint64_t clutch_start=0;bool clutch=false;V center{},stick_right{},stick_forward{};double axes[2]{};
    std::array<bool,7> buttons{}; // trigger, grip, a, b, menu, system, stick
    void reset(){*this={};}
    void update(const std::array<V,21>& p,uint32_t next,bool enabled,uint64_t now){
        if(id!=next){reset();id=next;}
        double width=std::clamp((p[5]-p[17]).len(),.04,.12);
        bool pinky=false;
        for(int f=0;f<4;++f){double gap=(p[4]-p[8+f*4]).len()/width;
            bool d=pinch[f].update(enabled,std::clamp((.65-gap)/.45,0.,1.));
            if(f<3)buttons[f==0?0:f+1]=d;
            if(f==3)pinky=d;
        }
        double curl=0;for(int f=0;f<4;++f){int b=5+4*f;double length=(p[b+1]-p[b]).len()+(p[b+2]-p[b+1]).len()+(p[b+3]-p[b+2]).len();curl+=std::clamp((1-(p[b+3]-p[b]).len()/length)/.65,0.,1.)/4;}
        buttons[1]=grip.update(enabled,curl);
        V normal=(p[5]-p[0]).cross(p[17]-p[0]).unit();
        bool stick_request=pinky&&std::abs(normal.y)>.75;
        if(stick_request){
            menu_start=0;
            if(!clutch_start)clutch_start=now;
            if(!clutch&&now-clutch_start>=300){clutch=true;center=p[0];V forward=p[9]-p[0];forward.y=0;stick_forward=forward.unit();stick_right=stick_forward.cross({0,1,0}).unit();}
        }else{clutch_start=0;clutch=false;}
        axes[0]=axes[1]=0;
        if(clutch){V d=p[0]-center;axes[0]=std::clamp(d.dot(stick_right)/.06,-1.,1.);axes[1]=std::clamp(d.dot(stick_forward)/.06,-1.,1.);}
        for(auto& axis:axes)if(std::abs(axis)<.15)axis=0;
        if(pinky&&!stick_request){if(!menu_start)menu_start=now;}else menu_start=0;
        buttons[4]=menu_start&&now-menu_start>=800;
        buttons[5]=false; // system dashboard is deliberately not opened by a conducting gesture
        buttons[6]=clutch&&buttons[0];
        if(clutch)buttons[0]=buttons[1]=buttons[2]=buttons[3]=false;
    }
};
}
