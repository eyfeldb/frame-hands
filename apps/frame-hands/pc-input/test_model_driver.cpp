#define FRAME_HANDS_DRIVER_MODEL
#define FRAME_HANDS_DRIVER_INPUT_MODE
#define FRAME_HANDS_DRIVER_ROLE_GRACE
#define main previous_stability_main
#include "test_driver_stability.cpp"
#undef main
int main(int argc,char** argv){
    if(argc==2&&!std::strcmp(argv[1],"--clock")){std::printf("%llu\n",(unsigned long long)now_ms());return 0;}
    assert(provider.Init(&context)==VRInitError_None);reset_devices();
    auto old=example();fhs::ModelPacket p;p.sequence=1;p.render_ms=p.source_ms=1000;
    for(int side=0;side<2;++side){auto& h=p.hands[side];h.id=side+1;h.flags=fhs::Present;h.capture_ms=980;h.until_ms=1200;std::memcpy(h.points,old.hands[side].points,sizeof h.points);}
    fhs::ModelMailbox box;assert(box.accept(p,1000));assert(!box.accept(p,1000));
    auto tick_model=[&](const fhs::ModelPacket& value,uint64_t now,bool head=true,bool busy=false){for(auto* d:host.device)static_cast<Device*>(d)->frame_model(value,fhs::model_envelope(value,now),head,busy,now,true);};
    tick_model(p,1000);assert(host.pose[0].poseIsValid&&host.pose[1].poseIsValid);
    // Render publications can advance; original capture deadline cannot.
    p.sequence++;p.render_ms=p.source_ms=1199;tick_model(p,1199);assert(host.pose[0].poseIsValid);
    p.render_ms=p.source_ms=1200;tick_model(p,1200);assert(!host.pose[0].poseIsValid&&host.pose[0].deviceIsConnected);
    tick_model(p,1281);assert(!host.pose[0].deviceIsConnected);
    p.render_ms=p.source_ms=1000;tick_model(p,1000,false);assert(!host.pose[0].deviceIsConnected);
    tick_model(p,1000,true,true);assert(!host.pose[0].deviceIsConnected);
    auto extended=p;extended.hands[0].until_ms=extended.hands[1].until_ms=1330;extended.render_ms=1329;
    tick_model(extended,1329);assert(host.pose[0].poseIsValid);
    extended.render_ms=1330;tick_model(extended,1330);assert(!host.pose[0].poseIsValid&&host.pose[0].deviceIsConnected);
    extended.render_ms=1501;tick_model(extended,1501);assert(!host.pose[0].deviceIsConnected);
    extended.hands[0].until_ms=1331;assert(!fhs::model_hand(extended.hands[0],extended.source_ms));
    auto old_clock=p;old_clock.magic[7]='1';old_clock.version=1;assert(!fhs::model_envelope(old_clock,1000));
    p.sequence++;p.hands[0].points[2][0]=NAN;assert(box.accept(p,1000));assert(!box.packet.hands[0].flags&&box.packet.hands[1].flags);
    tick_model(box.packet,1000);assert(!host.pose[0].poseIsValid&&host.pose[1].poseIsValid);
    fhs::ModeSelection selection;fhs::ModeCommand mode;mode.sequence=1;mode.mode=fhs::InputMode::Hands;assert(selection.accept(mode));selection.controller_press();assert(selection.mode==fhs::InputMode::Controllers);
    size_t packets=0,hand_count=0;double max_error=0;
    if(argc==2){
        std::ifstream file(argv[1],std::ios::binary);assert(file);
        while(file.read(reinterpret_cast<char*>(&p),sizeof p)){
            assert(fhs::model_envelope(p,p.render_ms));tick_model(p,p.render_ms);++packets;
            for(int side=0;side<2;++side)if(p.hands[side].flags){
                assert(fhs::model_hand(p.hands[side],p.source_ms));assert(host.pose[side].poseIsValid);++hand_count;
                auto result=published(side);
                for(int i=0;i<21;++i){auto q=p.hands[side].points[i];auto error=(result[i]-fhs::V{q[0],q[1],q[2]}).len();max_error=std::max(max_error,error);assert(error<.000005);}
            }
        }assert(packets>0&&hand_count>0);
    }
    assert(!input.any(5)&&!input.any(6));
    for(auto* d:host.device)d->Deactivate();provider.Cleanup();
    std::printf("PASS model driver: expiry, independent rejection, role retention, controller handoff, transport/HMD loss, no buttons; packets=%zu hands=%zu max_joint_error_mm=%.8f\n",packets,hand_count,max_error*1000);
}
