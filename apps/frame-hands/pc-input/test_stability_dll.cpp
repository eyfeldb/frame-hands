// Load the release candidate DLL into a mock server; never use the installed bundle.
#define FRAME_HANDS_DRIVER_MOCKS_ONLY
#include "test_driver.cpp"
#ifdef FRAME_HANDS_DRIVER_INPUT_MODE
#include "input_mode.h"
#endif
int main(int argc,char** argv){
    std::setvbuf(stdout,nullptr,_IONBF,0);assert(argc==2);
    std::string dll=argv[1],path=dll.substr(0,dll.find_last_of("\\/"))+"/input.bin";
    assert(GetFileAttributesA(path.c_str())==INVALID_FILE_ATTRIBUTES);
    auto library=LoadLibraryA(dll.c_str());assert(library);
    auto symbol=GetProcAddress(library,"HmdDriverFactory");void*(*factory)(const char*,int*)=nullptr;
    static_assert(sizeof symbol==sizeof factory);std::memcpy(&factory,&symbol,sizeof factory);assert(factory);
    int error=0;auto provider=static_cast<IServerTrackedDeviceProvider*>(factory(IServerTrackedDeviceProvider_Version,&error));assert(provider);
    assert(provider->Init(&context)==VRInitError_None);
    for(int side=0;side<2;++side)assert(host.device[side]->Activate(side+4)==VRInitError_None);
    provider->RunFrame();assert(!host.pose[0].poseIsValid&&!host.pose[1].poseIsValid);
    fhs::Packet packet;packet.options=fhs::Controls;
    for(int side=0;side<2;++side){
        std::array<fhs::Transform,31> local{};for(int i=0;i<31;++i)local[i]={fhs::rest[side][i].p,fhs::rest[side][i].q};auto bones=fhs::forward(local);
        auto& hand=packet.hands[side];hand.id=side+1;hand.flags=fhs::Present|fhs::Stereo;
        for(int i=0;i<21;++i){auto p=bones[fhs::landmark_bone[i]].p+fhs::V{side?.2:-.2,0,-.5};hand.points[i][0]=float(p.x);hand.points[i][1]=float(p.y);hand.points[i][2]=float(p.z);}
    }
    auto write=[&]{std::ofstream f(path,std::ios::binary|std::ios::trunc);f.write(reinterpret_cast<const char*>(&packet),sizeof packet);f.close();provider->RunFrame();assert(!input.any(5)&&!input.any(6));};
    auto publish=[&]{++packet.sequence;packet.sent_ms=GetTickCount64();packet.capture_ms=packet.sent_ms;write();};
    publish();assert(host.pose[0].poseIsValid&&host.pose[1].poseIsValid);
    assert(properties.roles[5]==TrackedControllerRole_LeftHand&&properties.roles[6]==TrackedControllerRole_RightHand);
    assert(input.updates[0]==input.updates[1]);
    auto good=packet.hands[0];packet.hands[0].points[2][0]=NAN;publish();
    assert(host.pose[0].poseIsValid&&host.pose[1].poseIsValid); // corrupt left isolated, bounded previous estimate
    Sleep(95);publish();assert(!host.pose[0].poseIsValid&&host.pose[1].poseIsValid);
#ifdef FRAME_HANDS_DRIVER_ROLE_GRACE
    assert(host.pose[0].deviceIsConnected&&properties.roles[5]==TrackedControllerRole_LeftHand);
#endif
    packet.hands[0]=good;Sleep(2);publish();assert(host.pose[0].poseIsValid);
    std::memcpy(packet.hands[0].points[4],packet.hands[0].points[8],sizeof packet.hands[0].points[4]);Sleep(2);publish();
    assert(!input.pressed(5,"/input/trigger/click"));
    host.head=false;provider->RunFrame();assert(!host.pose[0].poseIsValid&&!host.pose[1].poseIsValid);
    assert(!host.pose[0].deviceIsConnected&&!host.pose[1].deviceIsConnected);
    host.head=true;Sleep(2);publish();assert(host.pose[1].poseIsValid);
    Sleep(95);write();assert(!host.pose[0].poseIsValid&&!host.pose[1].poseIsValid); // repeat cannot renew capture age
    Sleep(70);write();assert(!host.pose[1].poseIsValid); // packet expiry also enforced
#ifdef FRAME_HANDS_DRIVER_ROLE_GRACE
    assert(host.pose[1].deviceIsConnected&&properties.roles[6]==TrackedControllerRole_RightHand);
    Sleep(610);write();assert(!host.pose[1].deviceIsConnected&&properties.roles[6]==TrackedControllerRole_OptOut);
#endif
    packet.hands[0]=good;publish();assert(host.pose[0].poseIsValid&&host.pose[1].poseIsValid);
#ifdef FRAME_HANDS_DRIVER_PRESENTATION
    // Exercise smoothing through the exported DLL, not only a header model.
    double start_x=host.pose[0].vecPosition[0];
    Sleep(2);for(auto& point:packet.hands[0].points)point[0]+=.05f;publish();
    double first_x=host.pose[0].vecPosition[0];assert(first_x>start_x&&first_x<start_x+.045);
    Sleep(12);provider->RunFrame();assert(host.pose[0].vecPosition[0]>first_x);
    Sleep(85);provider->RunFrame();assert(!host.pose[0].poseIsValid); // display filter cannot extend hold
    packet.hands[0]=good;Sleep(2);publish();assert(std::abs(host.pose[0].vecPosition[0]-start_x)<1e-5);
#endif
#ifdef FRAME_HANDS_DRIVER_INPUT_MODE
    std::string mode_path=dll.substr(0,dll.find_last_of("\\/"))+"/mode.bin";
    assert(GetFileAttributesA(mode_path.c_str())==INVALID_FILE_ATTRIBUTES);
    fhs::ModeCommand mode;mode.sequence=1;mode.mode=fhs::InputMode::Hands;
    auto mode_write=[&]{std::ofstream f(mode_path,std::ios::binary|std::ios::trunc);f.write(reinterpret_cast<const char*>(&mode),sizeof mode);f.close();Sleep(110);publish();};
    mode_write();packet.hands[0]={};packet.hands[1]={};
    for(int i=0;i<9;++i){Sleep(100);publish();}
    assert(host.pose[0].deviceIsConnected&&!host.pose[0].poseIsValid&&properties.roles[5]==TrackedControllerRole_LeftHand);
    properties.classes[4]=TrackedDeviceClass_Controller;
    VREvent_t press{};press.eventType=VREvent_ButtonPress;press.trackedDeviceIndex=3;host.events.push_back(press);publish();
    assert(!host.pose[0].deviceIsConnected&&!host.pose[1].deviceIsConnected);
    packet.hands[0]=good;mode_write();assert(!host.pose[0].deviceIsConnected); // same request cannot undo input
    mode.sequence++;mode_write();assert(host.pose[0].poseIsValid&&host.pose[0].deviceIsConnected);
    mode.sequence++;mode.mode=fhs::InputMode::Controllers;mode_write();assert(!host.pose[0].deviceIsConnected);
    assert(DeleteFileA(mode_path.c_str()));
#endif
    assert(DeleteFileA(path.c_str()));for(auto* d:host.device)d->Deactivate();provider->Cleanup();FreeLibrary(library);
    std::puts("PASS release candidate DLL: 31-bone ranges, roles, independent bounded hold, invalid-hand isolation, duplicate/packet expiry, HMD reset, skeleton-only controls, cleanup");
}
