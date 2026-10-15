#define FRAME_HANDS_DRIVER_DISPLAY
#define FRAME_HANDS_DRIVER_NAVIGATION
#define FRAME_HANDS_DRIVER_MODEL
#define FRAME_HANDS_DRIVER_INPUT_MODE
#define FRAME_HANDS_DRIVER_ROLE_GRACE
#define main previous_stability_main
#include "test_driver_stability.cpp"
#undef main
int main(){
    fhs::DisplayLease lease;lease.stamp_ms=1000;lease.hands=1;
    assert(fhs::display_lease(lease,1100,0));assert(!fhs::display_lease(lease,1101,0));
    assert(!fhs::display_lease(lease,999,0));assert(!fhs::display_lease(lease,1000,1));
    assert(!fhs::display_lease(lease,1000,-1));lease.reserved=1;assert(!fhs::display_lease(lease,1000,0));lease.reserved=0;
    assert(provider.Init(&context)==VRInitError_None);reset_devices();auto old=example();fhs::ModelPacket p;p.sequence=1;p.render_ms=p.source_ms=1000;
    for(int s=0;s<2;++s){auto& h=p.hands[s];h.id=s+1;h.flags=fhs::Present;h.capture_ms=1000;h.until_ms=1170;std::memcpy(h.points,old.hands[s].points,sizeof h.points);}
    auto d=static_cast<Device*>(host.device[0]);assert(properties.models[5]=="generic_controller");
    d->frame_model(p,true,true,false,1000,true,false,true);assert(properties.models[5]=="{fh_skeletal}fh_display_empty");
    assert(properties.models[6]=="generic_controller"); // other side untouched
    d->frame_model(p,true,true,false,1001,true,false,false);assert(properties.models[5]=="generic_controller"); // lease expired
    d->frame_model(p,true,true,false,1002,true,false,true);
    d->frame_model(p,true,true,true,1003,true,false,true);assert(properties.models[5]=="generic_controller"); // physical takeover
    d->frame_model(p,false,true,false,1004,true,false,true);assert(properties.models[5]=="generic_controller"); // dead transport
    for(auto* item:host.device)item->Deactivate();provider.Cleanup();
    std::puts("PASS display lease expiry, per-side rendering, controller takeover and transport fallback");
}
