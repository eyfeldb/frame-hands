// Bounded system-input bridge. Dry-run unless --inject is explicit.
#include <openvr.h>
#include "vendor/fh_hands.h"
#include "vendor/fh_gestures.h"
#include "protocol.h"
#include "gaze_math.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <fcntl.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>

namespace {
volatile sig_atomic_t stopping=0;
void stop(int){stopping=1;}
using Vec=fhg::Vec;
Vec vec(const float* v){return{v[0],v[1],v[2]};}
vr::HmdVector3_t vrvec(Vec v){return{{float(v.x),float(v.y),float(v.z)}};}
Vec transform(const vr::HmdMatrix34_t& m,Vec v){return{m.m[0][0]*v.x+m.m[0][1]*v.y+m.m[0][2]*v.z+m.m[0][3],m.m[1][0]*v.x+m.m[1][1]*v.y+m.m[1][2]*v.z+m.m[1][3],m.m[2][0]*v.x+m.m[2][1]*v.y+m.m[2][2]*v.z+m.m[2][3]};}
struct Ray{Vec origin,direction;bool valid()const{return origin.finite()&&direction.finite()&&direction.length()>.99&&direction.length()<1.01;}};
void set_ray(fh::Packet& p,Ray r) {
    p.position[0]=r.origin.x;p.position[1]=r.origin.y;p.position[2]=r.origin.z;
    // Yaw then pitch: the pointer's forward is -Z.
    const double yaw=std::atan2(-r.direction.x,-r.direction.z),pitch=std::asin(std::clamp(r.direction.y,-1.,1.));
    p.quaternion[0]=std::cos(yaw/2)*std::cos(pitch/2);p.quaternion[1]=std::cos(yaw/2)*std::sin(pitch/2);
    p.quaternion[2]=std::sin(yaw/2)*std::cos(pitch/2);p.quaternion[3]=-std::sin(yaw/2)*std::sin(pitch/2);
}
template<class T> bool read_packet(const std::string& path,const char* magic,uint32_t version,T& out,uint64_t now) {
    int fd=open(path.c_str(),O_RDONLY|O_CLOEXEC|O_NOFOLLOW);if(fd<0)return false;
    struct stat st{};bool ok=false;
    if(!fstat(fd,&st)&&S_ISREG(st.st_mode)&&st.st_uid==getuid()&&st.st_size==sizeof(T))for(int retry=0;retry<3;++retry){
        uint64_t before=0,after=0;
        if(pread(fd,&before,8,offsetof(T,seq))!=8 || (before&1))continue;
        if(pread(fd,&out,sizeof out,0)!=sizeof out)break;
        if(pread(fd,&after,8,offsetof(T,seq))!=8)break;
        if(before==after && before==out.seq && !std::memcmp(out.magic,magic,8)&&out.version==version&&out.size==sizeof out&&fh::fresh(out.capture_ns,now,100000000)&&fh::fresh(out.publish_ns,now,100000000)&&out.publish_ns>=out.capture_ns){ok=true;break;}
    }
    close(fd);return ok;
}
struct Sender {
    int fd=-1;sockaddr_un address{};socklen_t length=0;
    Sender(){fd=socket(AF_UNIX,SOCK_DGRAM|SOCK_CLOEXEC|SOCK_NONBLOCK,0);address.sun_family=AF_UNIX;std::string n="fh_pointer_"+std::to_string(getuid());std::memcpy(address.sun_path+1,n.data(),n.size());length=offsetof(sockaddr_un,sun_path)+1+n.size();}
    ~Sender(){if(fd>=0)close(fd);}
    bool send(fh::Packet p){p.sequence=p.sent_ns=fh::now_ns();return sendto(fd,&p,sizeof p,0,reinterpret_cast<sockaddr*>(&address),length)==sizeof p;}
};
}
int main(int argc,char** argv) {
    bool inject=false,touch=false;int seconds=20;const char* manifest=nullptr;std::string calibration_path;
    for(int i=1;i<argc;++i){std::string a=argv[i];if(a=="--inject")inject=true;else if(a=="--touch")touch=true;else if(a=="--seconds"&&i+1<argc)seconds=std::atoi(argv[++i]);else if(a=="--manifest"&&i+1<argc)manifest=argv[++i];else if(a=="--calibration"&&i+1<argc)calibration_path=argv[++i];else return 2;}
    if(!manifest||seconds<1||seconds>120){std::fprintf(stderr,"usage: fh-input --manifest ABS_JSON [--seconds 1..120] [--inject] [--touch]\n");return 2;}
    signal(SIGINT,stop);signal(SIGTERM,stop);
    fhg::Calibration calibration;bool calibration_ok=true;
    if(!calibration_path.empty()){
        calibration_ok=calibration.load(calibration_path,fhg::eye_session(),time(nullptr));
        if(!calibration_ok){std::fprintf(stderr,"Calibration is invalid, expired or from another eye-tracker session; recalibrate.\n");return 2;}
        std::printf("{\"calibration_loaded\":true,\"terms\":%d,\"max_age_seconds\":900}\n",calibration.terms);
    }
    vr::EVRInitError error=vr::VRInitError_None;auto sys=vr::VR_Init(&error,vr::VRApplication_Overlay);
    if(error||!sys){std::printf("{\"init_error\":%d}\n",int(error));return 1;}
    auto input=vr::VRInput();auto overlays=vr::VROverlay();vr::VRActionSetHandle_t set=0;vr::VRActionHandle_t action=0;
    if(input->SetActionManifestPath(manifest)||input->GetActionSetHandle("/actions/main",&set)||input->GetActionHandle("/actions/main/in/gaze",&action)){std::fprintf(stderr,"eye action setup failed\n");vr::VR_Shutdown();return 1;}
    vr::VRActiveActionSet_t active{};active.ulActionSet=set;
    Sender sender;
    if(inject&&!sender.send({})){std::fprintf(stderr,"pointer driver is not loaded; no input sent\n");vr::VR_Shutdown();return 1;}
    // Our invisible overlay only enables native mouse routing while we own the pointer.
    vr::VROverlayHandle_t routing=0;
    if(inject){
        auto e=overlays->CreateOverlay("framehands.input.routing","Frame Hands input",&routing);
        std::array<uint8_t,64> pixels{};vr::HmdMatrix34_t below{};below.m[0][0]=below.m[1][1]=below.m[2][2]=1;below.m[1][3]=-50;
        if(!e)e=overlays->SetOverlayRaw(routing,pixels.data(),4,4,4);
        if(!e)e=overlays->SetOverlayWidthInMeters(routing,.001f);
        if(!e)e=overlays->SetOverlayInputMethod(routing,vr::VROverlayInputMethod_Mouse);
        if(!e)e=overlays->SetOverlayFlag(routing,vr::VROverlayFlags_MakeOverlaysInteractiveIfVisible,true);
        if(!e)e=overlays->SetOverlayTransformTrackedDeviceRelative(routing,0,&below);
        if(e){std::fprintf(stderr,"routing overlay setup failed: %d\n",int(e));if(routing)overlays->DestroyOverlay(routing);vr::VR_Shutdown();return 1;}
    }
    fh::PinchGate pinch[2];fh::TouchGate contact[2];fh::ClaimPulse claim;Ray frozen{};int owner=-1;bool previous_click=false,route_visible=false;
    uint64_t last_controller=0,last_report=0;const auto start=fh::now_ns(),end=start+uint64_t(seconds)*1000000000;
    const std::string root="/run/user/"+std::to_string(getuid())+"/frametop-hands/";
    uint64_t frames=0,eyes=0,hand_frames=0,visible_frames=0,click_begins=0,touch_begins=0,hits=0,send_errors=0;
    int last_eye_error=0;bool keyboard_seen=false;double min_gap=1000;uint64_t last_sample=0,gaze_hits=0;
    double eye_yaw_min=1000,eye_yaw_max=-1000,eye_pitch_min=1000,eye_pitch_max=-1000;
    uint32_t our_index=vr::k_unTrackedDeviceIndexInvalid;
    for(uint32_t i=0;i<vr::k_unMaxTrackedDeviceCount;++i){char type[96]{};sys->GetStringTrackedDeviceProperty(i,vr::Prop_ControllerType_String,type,sizeof type);if(!std::strcmp(type,"fh_pointer"))our_index=i;}
    uint64_t primary_frames=0,hover_frames=0,corrected_frames=0,calibration_rejected=0,last_session_check=0;bool calibrated_head_seen=false;
    auto tick=std::chrono::steady_clock::now();
    while(!stopping&&fh::now_ns()<end) {
        const uint64_t now=fh::now_ns();++frames;
        vr::VREvent_t event{};while(sys->PollNextEvent(&event,sizeof event))if(event.eventType==vr::VREvent_Quit)stopping=1;
        vr::TrackedDevicePose_t poses[vr::k_unMaxTrackedDeviceCount]{};
        sys->GetDeviceToAbsoluteTrackingPose(vr::TrackingUniverseRawAndUncalibrated,0,poses,vr::k_unMaxTrackedDeviceCount);
        const bool head=poses[0].bPoseIsValid&&poses[0].bDeviceIsConnected;
        for(uint32_t i=1;i<vr::k_unMaxTrackedDeviceCount;++i)if(poses[i].bDeviceIsConnected&&sys->GetTrackedDeviceClass(i)==vr::TrackedDeviceClass_Controller){
            char type[96]{};sys->GetStringTrackedDeviceProperty(i,vr::Prop_ControllerType_String,type,sizeof type);
            if(!std::strcmp(type,"fh_pointer"))continue;
            vr::VRControllerState_t state{};
            if((sys->GetControllerState(i,&state,sizeof state)&&state.ulButtonPressed)||
               (poses[i].bPoseIsValid&&(vec(poses[i].vVelocity.v).length()>.35||vec(poses[i].vAngularVelocity.v).length()>2)))last_controller=now;
        }
        const bool controller_busy=last_controller&&now-last_controller<2000000000ull;
        const bool system_ui=vr::VRApplications()->GetCurrentSceneProcessId()==0||overlays->IsDashboardVisible();
        vr::VREyeTrackingData_t eye{};auto update=input->UpdateActionState(&active,sizeof active,1);
        last_eye_error=int(input->GetEyeTrackingDataRelativeToNow(action,vr::TrackingUniverseRawAndUncalibrated,0,&eye,sizeof eye));
        Ray gaze{vec(eye.vGazeOrigin.v),(vec(eye.vGazeTarget.v)-vec(eye.vGazeOrigin.v)).unit()};
        bool eye_valid=!update&&!last_eye_error&&eye.bActive&&eye.bTracked&&eye.bValid&&gaze.valid();eyes+=eye_valid;
        if(!calibration_path.empty()){
            if(head)calibrated_head_seen=true;
            if(calibrated_head_seen&&!head)calibration_ok=false;
            if(now-last_session_check>=1000000000){
                calibration_ok=calibration_ok&&fhg::eye_session()==calibration.session&&time(nullptr)>=calibration.created&&time(nullptr)-calibration.created<=900;
                last_session_check=now;
            }
            if(eye_valid&&head&&calibration_ok){
                auto local=fhg::angles(fhg::inverse_rotate(poses[0].mDeviceToAbsoluteTracking,gaze.direction));fhg::Angles corrected{};
                if(calibration.apply(local,corrected)){gaze.direction=fhg::rotate(poses[0].mDeviceToAbsoluteTracking,fhg::direction(corrected)).unit();++corrected_frames;}
                else eye_valid=false;
            }else eye_valid=false;
            calibration_rejected+=!eye_valid;
        }
        if(eye_valid&&head){
            const auto& m=poses[0].mDeviceToAbsoluteTracking.m;Vec d=gaze.direction;
            Vec local{m[0][0]*d.x+m[1][0]*d.y+m[2][0]*d.z,m[0][1]*d.x+m[1][1]*d.y+m[2][1]*d.z,m[0][2]*d.x+m[1][2]*d.y+m[2][2]*d.z};
            double yaw=std::atan2(-local.x,-local.z)*180/3.141592653589793,pitch=std::asin(std::clamp(local.y,-1.,1.))*180/3.141592653589793;
            eye_yaw_min=std::min(eye_yaw_min,yaw);eye_yaw_max=std::max(eye_yaw_max,yaw);eye_pitch_min=std::min(eye_pitch_min,pitch);eye_pitch_max=std::max(eye_pitch_max,pitch);
        }
        fh_hands_t hands{};fh_gestures_t gestures{};
        bool hands_valid=read_packet(root+"hands",FH_HANDS_MAGIC,FH_HANDS_VERSION,hands,now)&&hands.nhands<=2;
        bool gesture_valid=read_packet(root+"gestures",FH_GESTURES_MAGIC,FH_GESTURES_VERSION,gestures,now);
        if(hands_valid&&gesture_valid){auto a=hands.capture_ns,b=gestures.capture_ns;gesture_valid=(a>b?a-b:b-a)<50000000;}
        hand_frames+=hands_valid&&hands.nhands>0;
        bool eligible=head&&!controller_busy&&system_ui&&!stopping;
        vr::VROverlayHandle_t keyboard=0;bool keyboard_visible=false;
        for(const char* key:{"valve.steam.gamepadui.keyboard","system.keyboard"})if(!overlays->FindOverlay(key,&keyboard)&&overlays->IsOverlayVisible(keyboard)){keyboard_visible=true;break;}
        keyboard_seen|=keyboard_visible;
        auto primary=overlays->GetPrimaryDashboardDevice();bool our_primary=our_index!=vr::k_unTrackedDeviceIndexInvalid&&primary==our_index;
        primary_frames+=our_primary;hover_frames+=keyboard_visible&&overlays->IsHoverTargetOverlay(keyboard);
        if(keyboard_visible&&eye_valid){vr::VROverlayIntersectionParams_t p{};p.eOrigin=vr::TrackingUniverseRawAndUncalibrated;p.vSource=vrvec(gaze.origin);p.vDirection=vrvec(gaze.direction);vr::VROverlayIntersectionResults_t r{};gaze_hits+=overlays->ComputeOverlayIntersection(keyboard,&p,&r);}
        bool pinch_down[2]{},touch_down[2]{},near[2]{};Ray touch_ray[2]{};bool hand_present[2]{};
        vr::TrackedDevicePose_t captured{};
        if(hands_valid)sys->GetDeviceToAbsoluteTrackingPose(vr::TrackingUniverseRawAndUncalibrated,-float(now-hands.capture_ns)*1e-9f,&captured,1);
        const bool new_sample=hands_valid&&hands.capture_ns!=last_sample;
        for(int side=0;side<2;++side){
            const fh_hand_t* hand=nullptr;
            if(hands_valid)for(uint32_t j=0;j<hands.nhands;++j)if(bool(hands.hands[j].flags&FH_HAND_RIGHT)==bool(side)){
                if(hand){hand=nullptr;break;}hand=&hands.hands[j];
            }
            bool valid_hand=eligible&&hand&&hand->id&&std::isfinite(hand->confidence)&&hand->confidence>=.5;
            hand_present[side]=valid_hand;
            const auto& g=gestures.pinch[side];
            const bool gvalid=valid_hand&&eye_valid&&gesture_valid&&g.hand_id==hand->id&&(g.flags&FH_PINCH_TRACKED);
            pinch_down[side]=pinch[side].update(gvalid,g.hand_id,g.flags&FH_PINCH_DOWN);
            double gap=1000;
            if(valid_hand&&keyboard_visible&&captured.bPoseIsValid&&captured.bDeviceIsConnected&&(hand->flags&FH_HAND_STEREO)){
                Vec tip=transform(captured.mDeviceToAbsoluteTracking,vec(hand->pts[8]));
                Vec base=transform(captured.mDeviceToAbsoluteTracking,vec(hand->pts[6]));
                Vec direction=(tip-base).unit();Ray ray{tip-direction*.08,direction};
                vr::VROverlayIntersectionParams_t params{};params.eOrigin=vr::TrackingUniverseRawAndUncalibrated;params.vSource=vrvec(ray.origin);params.vDirection=vrvec(ray.direction);
                vr::VROverlayIntersectionResults_t result{};
                if(ray.valid()&&overlays->ComputeOverlayIntersection(keyboard,&params,&result)&&std::isfinite(result.fDistance)&&std::abs(direction.dot(vec(result.vNormal.v)))>.5){
                    gap=result.fDistance-.08;
                    if(new_sample){++hits;min_gap=std::min(min_gap,gap);}
                    near[side]=gap>=-.025&&gap<=.09;touch_ray[side]=ray;
                }
            }
            touch_down[side]=contact[side].update(touch&&near[side],hand?hand->id:0,gap);
            if(new_sample&&hand)std::printf("{\"pinch_geometry\":true,\"t_ns\":%llu,\"side\":%d,\"id\":%u,\"published_tip_gap_m\":%.5f,\"palm_width_m\":%.5f,\"model_tip_gap_m\":%.5f,\"begin_threshold_m\":%.5f,\"end_threshold_m\":%.5f,\"gesture_packet_fresh\":%s}\n",(unsigned long long)hands.capture_ns,side,hand->id,(vec(hand->pts[4])-vec(hand->pts[8])).length(),(vec(hand->pts[5])-vec(hand->pts[17])).length(),g.distance,gestures.begin_m,gestures.end_m,gesture_valid?"true":"false");
            if(new_sample&&hand)std::printf("{\"hand_input\":true,\"t_ns\":%llu,\"side\":%d,\"id\":%u,\"stereo\":%s,\"confidence_age\":%.3f,\"gesture_valid\":%s,\"gesture_id\":%u,\"gesture_flags\":%u,\"pinch_distance_m\":%.4f,\"pinch_begins\":%u,\"pinch_ends\":%u,\"pinch_armed\":%s,\"pinch_down\":%s,\"tip_gap_m\":%.4f}\n",(unsigned long long)hands.capture_ns,side,hand->id,(hand->flags&FH_HAND_STEREO)?"true":"false",hand->confidence,gvalid?"true":"false",g.hand_id,g.flags,g.distance,g.begins,g.ends,pinch[side].armed?"true":"false",pinch_down[side]?"true":"false",gap);
        }
        if(hands_valid)last_sample=hands.capture_ns;
        bool touch_mode=touch&&(near[0]||near[1]);
        // Preserve one owner through a press. Prefer right only when neither owns it.
        int desired=owner;
        if(!previous_click)desired=touch_mode?(near[1]?1:0):(pinch_down[1]?1:pinch_down[0]?0:hand_present[1]?1:hand_present[0]?0:-1);
        bool visible=eligible&&desired>=0&&(touch_mode?near[desired]:eye_valid&&hand_present[desired]);
        bool click=visible&&(touch_mode?touch_down[desired]:pinch_down[desired]);
        // Switching between near touch and gaze must pass through a release.
        static bool previous_touch=false;
        if(previous_click&&touch_mode!=previous_touch){click=false;visible=false;}
        Ray ray=touch_mode&&desired>=0?touch_ray[desired]:gaze;
        if(click&&!previous_click){frozen=ray;++click_begins;if(touch_mode)++touch_begins;}
        if(previous_click||click)ray=frozen;
        if(!visible){for(auto& p:pinch)p.update(false,0,false);for(auto& c:contact)c.update(false,0,0);click=false;desired=-1;}
        fh::Packet packet;
        bool claiming=claim.update(visible&&ray.valid(),now);
        if(visible&&ray.valid()){packet.flags=fh::Visible|(click?uint32_t(fh::Click):0u)|(claiming?uint32_t(fh::Claim):0u);set_ray(packet,ray);++visible_frames;}
        else{visible=false;click=false;}
        if(inject){
            if(visible!=route_visible){if(visible)overlays->ShowOverlay(routing);else overlays->HideOverlay(routing);route_visible=visible;}
            if(!sender.send(packet)){++send_errors;stopping=1;}
        }
        owner=desired;previous_click=click;previous_touch=touch_mode;
        if(now-last_report>=1000000000)std::printf("{\"routing\":true,\"our_index\":%u,\"primary_index\":%u,\"owns_primary\":%s,\"keyboard_hover\":%s,\"claiming\":%s}\n",our_index,primary,our_primary?"true":"false",keyboard_visible&&overlays->IsHoverTargetOverlay(keyboard)?"true":"false",claiming?"true":"false");
        if(now-last_report>=1000000000){std::printf("{\"elapsed_s\":%.2f,\"head\":%s,\"gaze\":%s,\"hands\":%u,\"keyboard\":%s,\"system_ui\":%s,\"physical_controller\":%s,\"pointer\":%s,\"click\":%s,\"touch_mode\":%s,\"injected\":%s}\n",(now-start)*1e-9,head?"true":"false",eye_valid?"true":"false",hands_valid?hands.nhands:0,keyboard_visible?"true":"false",system_ui?"true":"false",controller_busy?"true":"false",visible?"true":"false",click?"true":"false",touch_mode?"true":"false",inject?"true":"false");std::fflush(stdout);last_report=now;}
        tick+=std::chrono::microseconds(11111);auto current=std::chrono::steady_clock::now();if(tick<current)tick=current;std::this_thread::sleep_until(tick);
    }
    if(inject){sender.send({});overlays->HideOverlay(routing);overlays->DestroyOverlay(routing);}
    std::printf("{\"gaze_geometry\":true,\"keyboard_gaze_hits\":%llu,\"head_relative_eye_yaw_range_deg\":%.3f,\"head_relative_eye_pitch_range_deg\":%.3f}\n",(unsigned long long)gaze_hits,eye_yaw_max>=eye_yaw_min?eye_yaw_max-eye_yaw_min:0,eye_pitch_max>=eye_pitch_min?eye_pitch_max-eye_pitch_min:0);
    std::printf("{\"routing_summary\":true,\"primary_frames\":%llu,\"keyboard_hover_frames\":%llu}\n",(unsigned long long)primary_frames,(unsigned long long)hover_frames);
    std::printf("{\"calibration_summary\":true,\"requested\":%s,\"still_valid\":%s,\"corrected_frames\":%llu,\"rejected_frames\":%llu}\n",calibration_path.empty()?"false":"true",calibration_ok?"true":"false",(unsigned long long)corrected_frames,(unsigned long long)calibration_rejected);
    std::printf("{\"summary\":true,\"seconds\":%.3f,\"frames\":%llu,\"gaze_frames\":%llu,\"hand_frames\":%llu,\"pointer_frames\":%llu,\"click_begins\":%llu,\"touch_begins\":%llu,\"keyboard_seen\":%s,\"keyboard_ray_hits\":%llu,\"nearest_tip_gap_m\":%.4f,\"eye_error\":%d,\"send_errors\":%llu,\"injected\":%s}\n",(fh::now_ns()-start)*1e-9,(unsigned long long)frames,(unsigned long long)eyes,(unsigned long long)hand_frames,(unsigned long long)visible_frames,(unsigned long long)click_begins,(unsigned long long)touch_begins,keyboard_seen?"true":"false",(unsigned long long)hits,min_gap,last_eye_error,(unsigned long long)send_errors,inject?"true":"false");
    vr::VR_Shutdown();return send_errors?1:0;
}
