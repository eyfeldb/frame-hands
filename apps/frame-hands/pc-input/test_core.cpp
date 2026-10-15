#include "skeleton.h"
#include "sample_input.h"
#undef NDEBUG
#include <cassert>
#include <cstdio>
using namespace fhs;
std::array<V,21> fixture(int side){
    std::array<Transform,31> r{};for(int i=0;i<31;++i)r[i]={rest[side][i].p,rest[side][i].q};auto f=forward(r);
    std::array<V,21> p{};for(int i=0;i<21;++i)p[i]=f[landmark_bone[i]].p+V{side?.2:-.2,0,-.5};return p;
}
int main(){
    for(int side=0;side<2;++side)for(int variant=0;variant<4;++variant){
        auto p=fixture(side);
        if(variant==1)for(auto& v:p)v=Q{.7071067811865,0,.7071067811865,0}.rotate(v)+V{.1,1.5,.2};
        if(variant==2)for(int f=0;f<5;++f){int tip=4+4*f;p[tip]=p[tip-1]+V{0,-.023,.009};}
        if(variant==3)for(auto& v:p)v=v*1.2;
        auto s=retarget(p,side);auto f=forward(s.bones);
        for(int i=0;i<21;++i)assert((s.position+s.rotation.rotate(f[landmark_bone[i]].p)-p[i]).len()<1e-6);
        for(auto b:s.bones){auto q=b.q;assert(std::abs(q.w*q.w+q.x*q.x+q.y*q.y+q.z*q.z-1)<1e-6);assert(b.p.finite());}
        assert(s.bones[0].p.len()==0&&s.bones[0].q.w==1);
        // Independent runtime convention: wrist is Y-180, not exported GLB identity.
        assert((s.bones[1].q.rotate({1,0,0})-V{-1,0,0}).len()<1e-6);
        for(int digit=0;digit<5;++digit){
            assert((f[26+digit].p-f[4+5*digit].p).len()<1e-6);
            assert(std::abs(s.bones[5+5*digit].q.w)>1-1e-6);
        }
    }
    Packet p;p.sequence=1;p.capture_ms=980;p.sent_ms=1000;
    for(int side=0;side<2;++side){p.hands[side].id=side+1;p.hands[side].flags=Present|Stereo;auto points=fixture(side);
        for(int i=0;i<21;++i){p.hands[side].points[i][0]=float(points[i].x);p.hands[side].points[i][1]=float(points[i].y);p.hands[side].points[i][2]=float(points[i].z);}}
    assert(valid(p,1000));Mailbox m;assert(m.accept(p,1000));assert(!m.accept(p,1001));assert(!m.fresh(1131));
    auto bad=p;bad.hands[0].points[2][0]=NAN;assert(!valid(bad,1000));bad=p;bad.hands[1].id=1;assert(!valid(bad,1000));
    bad=p;bad.capture_ms=1001;assert(!valid(bad,1000));bad=p;bad.options=4;assert(!valid(bad,1000));
    // A bad hand must release immediately without suppressing the good hand.
    bad=p;bad.sequence=2;bad.hands[0].points[2][0]=NAN;
    SampleInput sampled;assert(sampled.observe(bad,1000));
    assert(sampled.mailbox.fresh(1000)&&!sampled.mailbox.packet.hands[0].flags&&sampled.mailbox.packet.hands[1].flags==3);
    assert(sampled.issues[0][int(HandIssue::Nonfinite)]==1&&sampled.observations==1);
    assert(!sampled.observe(bad,1001)&&sampled.observations==1);
    assert(!sampled.mailbox.fresh(1131));
    auto segment_bad=p;segment_bad.sequence=3;
    for(int j=0;j<3;++j)segment_bad.hands[0].points[8][j]=segment_bad.hands[0].points[7][j];
    assert(sampled.observe(segment_bad,1000));
    assert(sampled.last_issue[0]==HandIssue::Segment&&sampled.bad_segments[0]==(1u<<8));
    assert(sampled.last_id[0]==1&&!sampled.bad_segments[1]&&sampled.mailbox.packet.hands[1].flags==3);
    auto duplicate=p;duplicate.sequence=4;duplicate.hands[1].id=1;
    assert(sampled.observe(duplicate,1000)&&sampled.last_duplicate);
    assert(!sampled.bad_segments[0]&&!sampled.mailbox.packet.hands[0].flags&&!sampled.mailbox.packet.hands[1].flags);
    auto recovered=p;recovered.sequence=5;
    assert(sampled.observe(recovered,1000)&&!sampled.last_duplicate&&sampled.last_issue[0]==HandIssue::None);
    assert(m.accept(bad,1000)&&m.fresh(1000));assert(!m.packet.hands[0].flags&&m.packet.hands[1].flags==3);
    bad=p;bad.sequence=3;bad.hands[1].points[2][0]=NAN;
    assert(m.accept(bad,1000)&&m.packet.hands[0].flags==3&&!m.packet.hands[1].flags);
    bad=p;bad.sequence=4;bad.hands[1].id=bad.hands[0].id;
    assert(m.accept(bad,1000)&&!m.packet.hands[0].flags&&!m.packet.hands[1].flags);
    bad=p;bad.hands[0].points[5][0]=bad.hands[0].points[17][0];bad.hands[0].points[5][1]=bad.hands[0].points[17][1];bad.hands[0].points[5][2]=bad.hands[0].points[17][2];assert(!valid(bad,1000));
    Gate gate;assert(!gate.update(true,1));assert(!gate.update(true,0));assert(gate.update(true,1));assert(!gate.update(false,1));assert(!gate.update(true,1));
    Smooth sm;auto a=fixture(0),b=a;for(auto& v:b)v.x+=.1;sm.update(a,1,.01);auto c=sm.update(b,1,.01);assert(c[0].x>a[0].x&&c[0].x<b[0].x);sm.reset();assert(sm.update(b,1,.01)[0].x==b[0].x);
    auto q=swing({1,0,0},{-1,0,0});assert((q.rotate({1,0,0})-V{-1,0,0}).len()<1e-6);
    ControlsState control;auto open=fixture(0);auto closed=open;closed[4]=closed[8];
    control.update(closed,1,true,1000);assert(!control.buttons[0]);control.update(open,1,true,1010);control.update(closed,1,true,1020);assert(control.buttons[0]);
    control.update(closed,2,true,1030);assert(!control.buttons[0]);control.update(open,2,false,1040);control.update(closed,2,false,1050);assert(!control.buttons[0]);
    auto flat=open;auto rotation=swing((flat[5]-flat[0]).cross(flat[17]-flat[0]),{0,1,0});for(auto& v:flat)v=rotation.rotate(v);
    control.reset();control.update(flat,1,true,2000);flat[4]=flat[20];control.update(flat,1,true,2010);control.update(flat,1,true,2310);assert(control.clutch&&!control.buttons[4]);
    auto moved=flat;for(auto& v:moved)v=v+control.stick_right*.09;control.update(moved,1,true,2320);assert(control.axes[0]>.99);control.update(open,1,true,2330);assert(control.axes[0]==0&&!control.clutch);
    std::puts("PASS left/right 21-landmark FK roundtrip, head rotation/translation, curled poses, scale, normalized bone rotations, wire validation, expiry/replay, invalid geometry, loss/rearm and smoothing reset");
}
