#define FRAME_HANDS_DRIVER_NAVIGATION
#define FRAME_HANDS_DRIVER_MODEL
#define FRAME_HANDS_DRIVER_INPUT_MODE
#define FRAME_HANDS_DRIVER_ROLE_GRACE
#define main previous_stability_main
#include "test_driver_stability.cpp"
#undef main
int main(){
    auto old=example();fhs::ModelPacket p;p.sequence=1;p.render_ms=p.source_ms=1000;
    for(int s=0;s<2;++s){auto& h=p.hands[s];h.id=s+1;h.flags=fhs::Present;h.capture_ms=980;h.until_ms=1150;std::memcpy(h.points,old.hands[s].points,sizeof h.points);}
    auto gap=[&](float value){auto& h=p.hands[0];for(int k=0;k<3;++k)h.points[4][k]=h.points[8][k];h.points[4][0]+=value;};
    auto advance=[&](uint64_t t){p.sequence++;p.source_ms=p.render_ms=t;for(auto& h:p.hands){h.capture_ms=t;h.until_ms=t+170;}};
    fhs::DashboardPinch gate;gap(.01f);assert(!gate.update(p,1000,true)); // closed at activation
    gap(.055f);advance(1033);assert(!gate.update(p,1033,true));
    gap(.01f);advance(1066);assert(!gate.update(p,1066,true));
    assert(!gate.update(p,1090,true)); // duplicate capture cannot debounce a press
    advance(1100);assert(!gate.update(p,1100,true));advance(1166);assert(!gate.update(p,1166,true)); // 100 ms: not yet (120 ms hold)
    advance(1200);assert(gate.update(p,1200,true));
    assert(!gate.update(p,1201,false)); // dashboard/mode/head loss releases
    assert(!gate.update(p,1202,true)); // held closed after returning cannot click
    gap(.055f);advance(1233);assert(!gate.update(p,1233,true));gap(.01f);advance(1266);assert(!gate.update(p,1266,true));
    gap(.025f);advance(1300);assert(!gate.update(p,1300,true)); // a frame between close and open keeps the hold running
    gap(.01f);advance(1333);assert(!gate.update(p,1333,true));advance(1400);assert(gate.update(p,1400,true));
    p.hands[1].id=9;assert(!gate.update(p,1401,true)); // pointer identity change releases
    assert(!gate.update(p,1501,true)); // old observation releases even before visual expiry
    gap(.055f);advance(1533);assert(!gate.update(p,1533,true));gap(.018f);advance(1566);assert(!gate.update(p,1566,true));advance(1700);assert(!gate.update(p,1700,true)); // fingers merely close (18 mm, above the widest close) never click
    fhs::DashboardContext c;c.stamp_ms=1000;c.enabled=1;assert(fhs::dashboard_context(c,1100));assert(!fhs::dashboard_context(c,1101));
    assert(provider.Init(&context)==VRInitError_None);reset_devices();gap(.01f);advance(1800);
    for(auto* d:host.device)static_cast<Device*>(d)->frame_model(p,true,true,false,1800,true,true);
    for(auto item:input.states)if(item.second){
        const auto& name=input.names[item.first];
#ifdef FRAME_HANDS_DRIVER_GESTURE_CONTROLS
        // The closed synthetic left hand also clicks the gesture trigger (index pinch = 1) on its own device (5).
        assert((input.containers[item.first]==6&&name=="/input/navigation/click")||(input.containers[item.first]==5&&(name=="/input/trigger/click"||name=="/input/grip/click")));
#else
        assert(input.containers[item.first]==6&&name=="/input/navigation/click");
#endif
    }
    for(auto* d:host.device)static_cast<Device*>(d)->frame_model(p,true,true,true,1801,true,true);
    assert(!input.any(5)&&!input.any(6));
    for(auto* d:host.device)d->Deactivate();provider.Cleanup();
    std::puts("PASS dashboard pinch: open rearm, capture debounce, tracking/identity/context loss release, right-only dedicated click, controller takeover");
}
