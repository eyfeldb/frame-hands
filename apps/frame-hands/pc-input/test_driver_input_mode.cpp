#define FRAME_HANDS_DRIVER_INPUT_MODE
#define FRAME_HANDS_DRIVER_ROLE_GRACE
#define main stability_regression_main
#include "test_driver_stability.cpp"
#undef main
int main(){
    fhs::ModeSelection selection;fhs::ModeCommand command;command.sequence=1;command.mode=fhs::InputMode::Hands;
    assert(selection.accept(command));selection.controller_press();assert(selection.mode==fhs::InputMode::Controllers);
    assert(!selection.accept(command)&&selection.mode==fhs::InputMode::Controllers);
    command.sequence++;assert(selection.accept(command)&&selection.mode==fhs::InputMode::Hands);
    command.sequence++;command.mode=fhs::InputMode(3);assert(!selection.accept(command));
    assert(provider.Init(&context)==VRInitError_None);reset_devices();auto p=example();
    auto frame=[&](uint64_t now,bool fresh=true,bool head=true,bool busy=false){for(auto* d:host.device)static_cast<Device*>(d)->frame(p,fresh,head,busy,now,true);};
    frame(1000);assert(host.pose[0].poseIsValid);
    p.hands[0]={};p.hands[1]={};
    for(uint64_t now=1100;now<12000;now+=100){p.sequence++;p.sent_ms=p.capture_ms=now;frame(now);
        for(int s=0;s<2;++s){assert(host.pose[s].deviceIsConnected&&!host.pose[s].poseIsValid);assert(properties.roles[s+5]==(s?TrackedControllerRole_RightHand:TrackedControllerRole_LeftHand));}
    }
    frame(12001,true,true,true);assert(!host.pose[0].deviceIsConnected); // deliberate controller mode
    frame(12002,false);assert(!host.pose[0].deviceIsConnected); // dead transport
    frame(12003,true,false);assert(!host.pose[0].deviceIsConnected); // HMD loss
    p=example();p.capture_ms=p.sent_ms=12010;frame(12020);assert(host.pose[0].poseIsValid);
    assert(!input.any(5)&&!input.any(6));
    for(auto* d:host.device)d->Deactivate();provider.Cleanup();
    std::puts("PASS explicit hands mode: loss does not release role; missing pose invalid; deliberate handoff latches; duplicate command cannot undo handoff; transport/HMD loss; reacquisition; controls disabled");
}
