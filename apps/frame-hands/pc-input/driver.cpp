// No camera access or network sockets in vrserver. The opt-in relay owns transport.
#define NOMINMAX
#include "driver_platform.h"
#include <openvr_driver.h>
#include "skeleton.h"
#ifdef FRAME_HANDS_DRIVER_MODEL
#include "model_protocol.h"
#include "display_protocol.h"
#include "retarget_chain.h"
#include "retarget_hinge.h"
using InputPacket=fhs::ModelPacket;
using InputMailbox=fhs::ModelMailbox;
#else
using InputPacket=fhs::Packet;
using InputMailbox=fhs::Mailbox;
#endif
#ifdef FRAME_HANDS_DRIVER_PRESENTATION
#include "retarget_chain.h"
#include "presentation.h"
#ifndef FRAME_HANDS_DRIVER_STABILITY
#error Presentation requires the skeleton-only stability driver
#endif
#endif
#ifdef FRAME_HANDS_DRIVER_STABILITY
#include "stability.h"
#endif
#ifdef FRAME_HANDS_DRIVER_NAVIGATION
#include "dashboard_navigation.h"
#endif
#ifdef FRAME_HANDS_DRIVER_DISPLAY
#include "display_protocol.h"
#ifndef FRAME_HANDS_DRIVER_MODEL
#error Display requires the model driver
#endif
#endif
#include <algorithm>
#include <mutex>
#include <string>
#ifdef FRAME_HANDS_DRIVER_INPUT_MODE
#include "input_mode.h"
#ifndef FRAME_HANDS_DRIVER_ROLE_GRACE
#error Explicit input mode requires role-grace support
#endif
#endif
#if defined(FRAME_HANDS_DRIVER_ROLE_GRACE) && !defined(FRAME_HANDS_DRIVER_STABILITY)
#error Role grace requires the skeleton-only stability driver
#endif
#ifdef FRAME_HANDS_DRIVER_LINK
#include "link_gestures.h"
#ifndef FRAME_HANDS_DRIVER_MODEL
#error Link hand type requires the model driver
#endif
#endif
#ifdef FRAME_HANDS_DRIVER_GESTURE_CONTROLS
// PC build for controller-driven games (Synth Riders and the like): the index pinch drives the trigger and
// the closed hand drives the grip, so games without skeletal input still get a usable hand controller.
#include "link_gestures.h"
#ifndef FRAME_HANDS_DRIVER_MODEL
#error Gesture controls require the model driver
#endif
#endif
using namespace vr;
namespace {
fhs_platform::Path input_path;
uint64_t now_ms(){return fhs_platform::now_ms();}
bool read_packet(InputPacket& packet){return fhs_platform::read_exact(input_path,packet);}
class Device final:public ITrackedDeviceServerDriver {
public:
    int side=0;uint32_t index=k_unTrackedDeviceIndexInvalid;fhs::Smooth smooth;fhs::ControlsState controls;
    #ifdef FRAME_HANDS_DRIVER_STABILITY
    fhs::Stability stability;
    #endif
#ifdef FRAME_HANDS_DRIVER_PRESENTATION
    fhs::Presentation presentation;
#endif
#ifdef FRAME_HANDS_DRIVER_ROLE_GRACE
    uint64_t last_observation=0;
#endif
    uint64_t seen=0,last_frame=0;std::array<fhs::V,21> target{};bool target_valid=false;
    EVRInitError Activate(uint32_t i)override{
        index=i;props=VRProperties()->TrackedDeviceToPropertyContainer(i);auto p=VRProperties();
#ifdef FRAME_HANDS_DRIVER_DISPLAY
        // Optical hands never present a device render model: apps that draw their own skeletal hands would
        // otherwise show a second (controller) model next to them. The dashboard glove overlay covers
        // Home/dashboard, where no app draws hands.
        p->SetStringProperty(props,Prop_RenderModelName_String,"{fh_skeletal}fh_display_empty");
        display_hidden=true;
#endif
        p->SetStringProperty(props,Prop_ModelNumber_String,"Frame Hands Optical v1");
        p->SetStringProperty(props,Prop_ManufacturerName_String,"Frame Hands");
#ifdef FRAME_HANDS_DRIVER_LINK
        // Streamed PC VR: Valve's link (vrlink) mirrors the headset's controllers onto the PC, but only for
        // controller types it knows; "fh_skeletal" was refused ("Unsupported controller type"). Present the
        // optical hands as the link's own hand-tracking type, with the components its PC-side profile expects,
        // so a PC game receives them natively with nothing installed on the PC.
        p->SetStringProperty(props,Prop_ControllerType_String,"svl_hand_interaction_augmented");
        p->SetStringProperty(props,Prop_InputProfilePath_String,"{fh_skeletal}/input/fh_link_profile.json");
#else
        p->SetStringProperty(props,Prop_ControllerType_String,"fh_skeletal");
        p->SetStringProperty(props,Prop_InputProfilePath_String,"{fh_skeletal}/input/fh_skeletal_profile.json");
#endif
        p->SetStringProperty(props,Prop_SerialNumber_String,side?"frame-hands-optical-right":"frame-hands-optical-left");
        p->SetInt32Property(props,Prop_ControllerRoleHint_Int32,TrackedControllerRole_OptOut);
        p->SetInt32Property(props,Prop_ControllerHandSelectionPriority_Int32,10);
        p->SetBoolProperty(props,Prop_NeverTracked_Bool,false);
#ifdef FRAME_HANDS_DRIVER_NAVIGATION
        if(VRDriverInput()->CreateBooleanComponent(props,"/input/navigation/click",&navigation))return VRInitError_Driver_Failed;
#endif
        const char* names[]={"trigger","grip","a","b","application_menu","system","joystick"};
        for(int n=0;n<7;++n)if(VRDriverInput()->CreateBooleanComponent(props,(std::string("/input/")+names[n]+"/click").c_str(),&buttons[n]))return VRInitError_Driver_Failed;
        if(VRDriverInput()->CreateScalarComponent(props,"/input/trigger/value",&trigger,VRScalarType_Absolute,VRScalarUnits_NormalizedOneSided)||
           VRDriverInput()->CreateScalarComponent(props,"/input/grip/value",&grip,VRScalarType_Absolute,VRScalarUnits_NormalizedOneSided)||
           VRDriverInput()->CreateScalarComponent(props,"/input/joystick/x",&axes[0],VRScalarType_Absolute,VRScalarUnits_NormalizedTwoSided)||
           VRDriverInput()->CreateScalarComponent(props,"/input/joystick/y",&axes[1],VRScalarType_Absolute,VRScalarUnits_NormalizedTwoSided)||
           VRDriverInput()->CreateHapticComponent(props,"/output/haptic",&haptic))return VRInitError_Driver_Failed;
#ifdef FRAME_HANDS_DRIVER_LINK
        const char* pinch_names[]={"index_pinch","middle_pinch","ring_pinch","pinky_pinch"};
        for(int n=0;n<4;++n)if(VRDriverInput()->CreateScalarComponent(props,(std::string("/input/")+pinch_names[n]+"/value").c_str(),&pinches[n],VRScalarType_Absolute,VRScalarUnits_NormalizedOneSided))return VRInitError_Driver_Failed;
        if(VRDriverInput()->CreateBooleanComponent(props,"/input/index_point/touch",&index_point))return VRInitError_Driver_Failed;
#endif
        if(VRDriverInput()->CreateSkeletonComponent(props,side?"/input/skeleton/right":"/input/skeleton/left",side?"/skeleton/hand/right":"/skeleton/hand/left","/pose/raw",VRSkeletalTracking_Full,nullptr,0,&skeleton))return VRInitError_Driver_Failed;
        VRBoneTransform_t rest[31]{};for(int n=0;n<31;++n){auto t=fhs::rest[side][n];rest[n]={{float(t.p.x),float(t.p.y),float(t.p.z),1},{float(t.q.w),float(t.q.x),float(t.q.y),float(t.q.z)}};}
        for(auto range:{VRSkeletalMotionRange_WithController,VRSkeletalMotionRange_WithoutController})if(VRDriverInput()->UpdateSkeletonComponent(skeleton,range,rest,31))return VRInitError_Driver_Failed;
        VRDriverLog()->Log(side?"Frame Hands right skeleton ready (disconnected)":"Frame Hands left skeleton ready (disconnected)");
        return VRInitError_None;
    }
    void Deactivate()override{
        index=k_unTrackedDeviceIndexInvalid;smooth.reset();controls.reset();target_valid=false;
#ifdef FRAME_HANDS_DRIVER_STABILITY
        stability.reset();
#endif
#ifdef FRAME_HANDS_DRIVER_PRESENTATION
        presentation.reset();
#endif
#ifdef FRAME_HANDS_DRIVER_ROLE_GRACE
        last_observation=0;
#endif
    }
    void EnterStandby()override{}
    void* GetComponent(const char*)override{return nullptr;}
    void DebugRequest(const char*,char* text,uint32_t n)override{if(n)text[0]=0;}
    DriverPose_t GetPose()override{std::lock_guard<std::mutex> guard(lock);return pose;}
#ifdef FRAME_HANDS_DRIVER_MODEL
    // World-space fixed-rig output already contains the accepted motion filter.
    // Retarget only: never re-filter, recapture head pose, or renew observations.
#ifdef FRAME_HANDS_DRIVER_FOREARM
    // Forearm direction (elbow -> wrist) from a shoulder estimate below/beside the head and a two-link arm: the
    // device pose follows this instead of the palm, so a game's sleeve/forearm bends at the wrist like a real arm.
    static bool forearm_direction(const TrackedDevicePose_t& head,int side,const fhs::V& wrist,fhs::V& out){
        if(!head.bPoseIsValid||!head.bDeviceIsConnected)return false;
        const auto& m=head.mDeviceToAbsoluteTracking.m;
        fhs::V hp{m[0][3],m[1][3],m[2][3]},fwd{-m[0][2],-m[1][2],-m[2][2]},up{0,1,0};
        fwd={fwd.x,0,fwd.z};if(fwd.len()<.2)return false;fwd=fwd.unit();
        const fhs::V right=fwd.cross(up).unit();const double lateral=side?.18:-.18;
        const fhs::V shoulder=hp+right*lateral+up*-.22+fwd*-.03;
        const double upper=.28,lower=.26;fhs::V sw=wrist-shoulder;const double d=sw.len();if(d<.05)return false;
        const fhs::V u=sw.unit();fhs::V elbow;
        if(d>=upper+lower-.01)elbow=shoulder+u*upper;
        else{const double a=(upper*upper-lower*lower+d*d)/(2*d);const double h=std::sqrt(std::max(0.,upper*upper-a*a));
            fhs::V hint=up*-.8+right*(side?.3:-.3)+fwd*-.3;hint=hint-u*hint.dot(u);if(hint.len()<.05)hint=up*-1-u*(up*-1).dot(u);
            elbow=shoulder+u*a+hint.unit()*h;}
        out=(wrist-elbow).unit();return out.finite()&&out.len()>.5;
    }
#endif
    void frame_model(const fhs::ModelPacket& packet,bool fresh,bool head_valid,bool controller_busy,uint64_t now,bool retain_role,bool navigation_click=false,bool displayed=false,const TrackedDevicePose_t* head=nullptr){
        if(index==k_unTrackedDeviceIndexInvalid)return;
        const auto& hand=packet.hands[side];
        bool permitted=fresh&&head_valid&&!controller_busy;
        bool live=permitted&&(hand.flags&fhs::Present)&&fhs::model_hand(hand,packet.source_ms)&&now>=hand.capture_ms&&now<hand.until_ms;
        DriverPose_t next{};next.qWorldFromDriverRotation.w=next.qDriverFromHeadRotation.w=next.qRotation.w=1;
#if defined(FRAME_HANDS_DRIVER_LINK) || defined(FRAME_HANDS_DRIVER_GESTURE_CONTROLS)
        fhs::LinkGestures gestures;
#endif
        if(live){
            std::array<fhs::V,21> points{};
            for(int i=0;i<21;++i)points[i]={hand.points[i][0],hand.points[i][1],hand.points[i][2]};
#if defined(FRAME_HANDS_DRIVER_LINK) || defined(FRAME_HANDS_DRIVER_GESTURE_CONTROLS)
            gestures=fhs::link_gestures(points);
#endif
            // Tunable without a rebuild: "pose.txt" beside the mailbox holds "<yaw> <pitch> <roll> <hinge 0|1>":
            // degrees of a constant device-pose rotation in the palm's own axes (yaw about the palm normal, pitch
            // about the across-palm axis, roll about the finger direction; pitch and roll are mirrored for the left
            // hand) and the finger rotation model (1 = anatomical hinges, 0 = swing). Re-read once a second.
            if(now-pose_read_at>1000){pose_read_at=now;
                const auto knobs=fhs_platform::sibling(input_path,"pose.txt");
#ifdef _WIN32
                FILE* f=_wfopen(knobs.c_str(),L"r");
#else
                FILE* f=std::fopen(knobs.c_str(),"r");
#endif
                if(f){double a=0,b=0,c=0;int h=0;const int got=std::fscanf(f,"%lf %lf %lf %d",&a,&b,&c,&h);
                    if(got>=3&&std::abs(a)<=90&&std::abs(b)<=90&&std::abs(c)<=90){pose_fix_deg[0]=a;pose_fix_deg[1]=b;pose_fix_deg[2]=c;pose_hinge=got>=4&&h==1;}
                    std::fclose(f);}}
            const fhs::Q* fix=nullptr;fhs::Q fix_q;
            if(pose_fix_deg[0]!=0||pose_fix_deg[1]!=0||pose_fix_deg[2]!=0){
                const double rad=3.14159265358979323846/180,m=side?1:-1;
                fix_q=(fhs::axis_angle({0,0,1},pose_fix_deg[0]*rad)*fhs::axis_angle({1,0,0},m*pose_fix_deg[1]*rad)*fhs::axis_angle({0,1,0},m*pose_fix_deg[2]*rad)).unit();fix=&fix_q;}
#ifdef FRAME_HANDS_DRIVER_FOREARM
            fhs::V forearm;const bool have_forearm=head&&forearm_direction(*head,side,points[0],forearm);
            auto result=pose_hinge?fhs::retarget_hinge(points,side,have_forearm?&forearm:nullptr,fix):fhs::retarget_chain(points,side,have_forearm?&forearm:nullptr,fix);
#else
            (void)head;
            auto result=pose_hinge?fhs::retarget_hinge(points,side,nullptr,fix):fhs::retarget_chain(points,side,nullptr,fix);
#endif
            // Game-calibrated rigid correction C per side, in the device frame: D' = D*C, and the root-level bones are
            // re-based by C^-1 so the SteamVR dashboard glove (skinned to our positions) does not move. Measured inside
            // Maestro (UE4SS probe, 2026-10-12): the game orients its whole glove from the controller pose plus its own
            // offset and ignores the wrist bone. Files pose_left.txt / pose_right.txt beside the mailbox hold
            // "qw qx qy qz tx ty tz" (metres); re-read once a second.
            if(now-corr_read_at>1000){corr_read_at=now;
                const auto knobs=fhs_platform::sibling(input_path,side?"pose_right.txt":"pose_left.txt");
#ifdef _WIN32
                FILE* f=_wfopen(knobs.c_str(),L"r");
#else
                FILE* f=std::fopen(knobs.c_str(),"r");
#endif
                if(f){double v[7]{};const int got=std::fscanf(f,"%lf %lf %lf %lf %lf %lf %lf",&v[0],&v[1],&v[2],&v[3],&v[4],&v[5],&v[6]);std::fclose(f);
                    const double n=std::sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]+v[3]*v[3]);
                    if(got==7&&n>.5&&std::abs(v[4])<.5&&std::abs(v[5])<.5&&std::abs(v[6])<.5){corr_q=fhs::Q{v[0]/n,v[1]/n,v[2]/n,v[3]/n};corr_t={v[4],v[5],v[6]};corr_valid=true;}
                    else corr_valid=false;}
                else corr_valid=false;}
            if(corr_valid){
                const fhs::Q R=result.rotation;result.position=result.position+R.rotate(corr_t);result.rotation=(R*corr_q).unit();
                const fhs::Q ci=corr_q.inv();
                for(int b=1;b<31;++b)if(fhs::parents[b]==0){auto& bn=result.bones[b];bn.p=ci.rotate(bn.p-corr_t);bn.q=(ci*bn.q).unit();}
            }
            next.vecPosition[0]=result.position.x;next.vecPosition[1]=result.position.y;next.vecPosition[2]=result.position.z;
            next.qRotation={result.rotation.w,result.rotation.x,result.rotation.y,result.rotation.z};
#ifdef FRAME_HANDS_DRIVER_PREDICT
            // Streamed PC path: the pose is already old by the time it arrives (tracking + Wi-Fi), so give
            // SteamVR its age and a smoothed velocity and let it extrapolate to render time, as it does for
            // controllers. Finger bones are not predicted. Velocity is clamped and reset on a new observation gap.
            // Velocity is re-estimated only when a new observation arrives (about 30 Hz); the render frames in
            // between keep the last estimate instead of zeroing it, otherwise SteamVR extrapolates with zero
            // velocity two frames out of three and the hand stutters.
            // Tunable without a rebuild: "predict.txt" beside the mailbox holds "<fraction of the age to predict>
            // <velocity smoothing 0..1>" (defaults 1.0 0.35); re-read once a second so captures can A/B it live.
            if(now-predict_read_at>1000){predict_read_at=now;
                const auto knobs=fhs_platform::sibling(input_path,"predict.txt");
#ifdef _WIN32
                FILE* f=_wfopen(knobs.c_str(),L"r");
#else
                FILE* f=std::fopen(knobs.c_str(),"r");
#endif
                if(f){double a=1,b=.35,c=1;
                    const int got=std::fscanf(f,"%lf %lf %lf",&a,&b,&c);
                    if(got>=2&&a>=0&&a<=1.5&&b>=0&&b<=1){predict_fraction=a;velocity_smoothing=b;rotation_smoothing=(got>=3&&c>0&&c<=1)?c:1;}std::fclose(f);}}
            // Optional third knob: wrist orientation smoothing (1 = off). Orientation noise of a few degrees is what
            // makes a still hand visibly wander when a game hangs a long object on it.
            if(rotation_smoothing<1&&rotation_valid){
                double w=next.qRotation.w,x=next.qRotation.x,y=next.qRotation.y,z=next.qRotation.z;
                if(w*rotation_prev[0]+x*rotation_prev[1]+y*rotation_prev[2]+z*rotation_prev[3]<0){w=-w;x=-x;y=-y;z=-z;}
                const double k=rotation_smoothing;double q[4]={rotation_prev[0]+k*(w-rotation_prev[0]),rotation_prev[1]+k*(x-rotation_prev[1]),rotation_prev[2]+k*(y-rotation_prev[2]),rotation_prev[3]+k*(z-rotation_prev[3])};
                const double n=std::sqrt(q[0]*q[0]+q[1]*q[1]+q[2]*q[2]+q[3]*q[3]);if(n>1e-9){for(double& v:q)v/=n;next.qRotation={q[0],q[1],q[2],q[3]};}
            }
            rotation_prev[0]=next.qRotation.w;rotation_prev[1]=next.qRotation.x;rotation_prev[2]=next.qRotation.y;rotation_prev[3]=next.qRotation.z;rotation_valid=true;
            if(hand.capture_ms!=velocity_capture){
                if(velocity_valid&&hand.capture_ms>velocity_capture&&hand.capture_ms-velocity_capture<=100){
                    const double dt=double(hand.capture_ms-velocity_capture)/1000;
                    for(int i=0;i<3;++i){double v=(next.vecPosition[i]-velocity_position[i])/dt;v=std::max(-3.,std::min(3.,v));velocity[i]+=velocity_smoothing*(v-velocity[i]);}
                }else for(double& v:velocity)v=0;
                for(int i=0;i<3;++i)velocity_position[i]=next.vecPosition[i];velocity_capture=hand.capture_ms;velocity_valid=true;
            }
            // velocity smoothing 0 = report no velocity at all (the native headset driver behaviour): SteamVR then
            // renders the latest pose instead of extrapolating noisy optical velocity over the streaming horizon.
            for(int i=0;i<3;++i)next.vecVelocity[i]=velocity_smoothing>0?velocity[i]:0;
            next.poseTimeOffset=-predict_fraction*std::min(0.2,double(now>hand.capture_ms?now-hand.capture_ms:0)/1000);
#endif
            VRBoneTransform_t bones[31]{};
            for(int i=0;i<31;++i){auto t=result.bones[i];bones[i]={{float(t.p.x),float(t.p.y),float(t.p.z),1},{float(t.q.w),float(t.q.x),float(t.q.y),float(t.q.z)}};}
            for(auto range:{VRSkeletalMotionRange_WithController,VRSkeletalMotionRange_WithoutController})if(VRDriverInput()->UpdateSkeletonComponent(skeleton,range,bones,31))live=false;
        }
#ifdef FRAME_HANDS_DRIVER_NAVIGATION
        VRDriverInput()->UpdateBooleanComponent(navigation,side==1&&live&&navigation_click,0);
#else
        (void)navigation_click;
#endif
#ifdef FRAME_HANDS_DRIVER_DISPLAY
        // A fresh successful compositor submission is required for each side.
        // Never change physical controller properties or gameplay input.
        // The render model stays empty regardless of the overlay lease (see Activate); the lease only
        // tells the overlay renderer that the compositor accepted the gloves.
        (void)displayed;
        if(!display_hidden){
            auto error=VRProperties()->SetStringProperty(props,Prop_RenderModelName_String,"{fh_skeletal}fh_display_empty");
            if(error==TrackedProp_Success)display_hidden=true;
        }
#else
        (void)displayed;
#endif
        // Gameplay inputs stay skeleton-only; dashboard click has its own binding.
        for(auto button:buttons)VRDriverInput()->UpdateBooleanComponent(button,false,0);
        VRDriverInput()->UpdateScalarComponent(trigger,0,0);VRDriverInput()->UpdateScalarComponent(grip,0,0);
        for(auto axis:axes)VRDriverInput()->UpdateScalarComponent(axis,0,0);
#ifdef FRAME_HANDS_DRIVER_LINK
        // The link's hand type carries pinch/grip/point values; these are what a streamed PC game sees.
        for(int n=0;n<4;++n)VRDriverInput()->UpdateScalarComponent(pinches[n],live?gestures.pinch[n]:0.f,0);
        VRDriverInput()->UpdateScalarComponent(grip,live?gestures.grip:0.f,0);
        VRDriverInput()->UpdateBooleanComponent(index_point,live&&gestures.point,0);
#endif
#ifdef FRAME_HANDS_DRIVER_GESTURE_CONTROLS
        // Trigger = index pinch, grip = closed hand, each with a click hysteresis (on at 0.7, off at 0.45).
        if(!live){trigger_down=grip_down=false;}
        else{trigger_down=gestures.pinch[0]>=(trigger_down?.45f:.7f);grip_down=gestures.grip>=(grip_down?.45f:.7f);}
        VRDriverInput()->UpdateScalarComponent(trigger,live?gestures.pinch[0]:0.f,0);
        VRDriverInput()->UpdateScalarComponent(grip,live?gestures.grip:0.f,0);
        VRDriverInput()->UpdateBooleanComponent(buttons[0],trigger_down,0);
        VRDriverInput()->UpdateBooleanComponent(buttons[1],grip_down,0);
#endif
        bool connected=permitted&&(live||retain_role);
        if(connected&&!live){auto previous=GetPose();next.qRotation=previous.qRotation;for(int i=0;i<3;++i)next.vecPosition[i]=previous.vecPosition[i];}
        auto role=connected?(side?TrackedControllerRole_RightHand:TrackedControllerRole_LeftHand):TrackedControllerRole_OptOut;
        if(role!=last_role){VRProperties()->SetInt32Property(props,Prop_ControllerRoleHint_Int32,role);last_role=role;}
        next.deviceIsConnected=connected;next.poseIsValid=live;
        next.result=live?TrackingResult_Running_OK:(connected?TrackingResult_Running_OutOfRange:TrackingResult_Uninitialized);
        {std::lock_guard<std::mutex> guard(lock);pose=next;}
        VRServerDriverHost()->TrackedDevicePoseUpdated(index,next,sizeof next);
    }
#endif
    void frame(const fhs::Packet& packet,bool fresh,bool head_valid,bool controller_busy,uint64_t now,bool retain_role=false){
        (void)retain_role;
        if(index==k_unTrackedDeviceIndexInvalid)return;
        const auto& hand=packet.hands[side];
#ifdef FRAME_HANDS_DRIVER_STABILITY
        // Same observation-time world-space filter as the green diagnostic.
        // This candidate is skeleton-only; held/quarantined estimates never drive controls.
        controls.reset();
        const bool permitted=fresh&&head_valid&&!controller_busy;
        bool captured_valid=false;
        if(permitted){
            TrackedDevicePose_t captured{};
            VRServerDriverHost()->GetRawTrackedDevicePoses(-float(now-packet.capture_ms)/1000,&captured,1);
            captured_valid=captured.bPoseIsValid&&captured.bDeviceIsConnected;
            if(captured_valid&&(hand.flags&fhs::Present)){
                for(int n=0;n<21;++n){auto& m=captured.mDeviceToAbsoluteTracking.m;auto& v=hand.points[n];target[n]={m[0][0]*v[0]+m[0][1]*v[1]+m[0][2]*v[2]+m[0][3],m[1][0]*v[0]+m[1][1]*v[1]+m[1][2]*v[2]+m[1][3],m[2][0]*v[0]+m[2][1]*v[1]+m[2][2]*v[2]+m[2][3]};}
                stability.update(target,hand.id,packet.capture_ms,now);
            }
        }
        if(!permitted||!captured_valid)stability.reset();
        bool live=permitted&&captured_valid&&stability.visible(now);
        DriverPose_t next{};next.qWorldFromDriverRotation.w=next.qDriverFromHeadRotation.w=next.qRotation.w=1;
        if(live){
            const auto& points=stability.value;
#else
        bool live=fresh&&head_valid&&!controller_busy&&(hand.flags&fhs::Present);
        if(live&&packet.sequence!=seen){
            seen=packet.sequence;TrackedDevicePose_t captured{};
            VRServerDriverHost()->GetRawTrackedDevicePoses(-float(now-packet.capture_ms)/1000,&captured,1);
            target_valid=captured.bPoseIsValid&&captured.bDeviceIsConnected;
            if(target_valid)for(int n=0;n<21;++n){auto& m=captured.mDeviceToAbsoluteTracking.m;auto& v=hand.points[n];target[n]={m[0][0]*v[0]+m[0][1]*v[1]+m[0][2]*v[2]+m[0][3],m[1][0]*v[0]+m[1][1]*v[1]+m[1][2]*v[2]+m[1][3],m[2][0]*v[0]+m[2][1]*v[1]+m[2][2]*v[2]+m[2][3]};}
        }
        live=live&&target_valid;DriverPose_t next{};next.qWorldFromDriverRotation.w=next.qDriverFromHeadRotation.w=next.qRotation.w=1;
        if(live){
            controls.update(target,hand.id,(packet.options&fhs::Controls)&&(hand.flags&fhs::Stereo),now);
            auto points=smooth.update(target,hand.id,last_frame?double(now-last_frame)/1000:0);
#endif
#ifdef FRAME_HANDS_DRIVER_PRESENTATION
            auto result=presentation.update(fhs::retarget_chain(points,side),stability.id,stability.stamp,now);
#else
            auto result=fhs::retarget(points,side);
#endif
            next.vecPosition[0]=result.position.x;next.vecPosition[1]=result.position.y;next.vecPosition[2]=result.position.z;
            next.qRotation={result.rotation.w,result.rotation.x,result.rotation.y,result.rotation.z};
            VRBoneTransform_t bones[31]{};
            for(int n=0;n<31;++n){auto t=result.bones[n];bones[n]={{float(t.p.x),float(t.p.y),float(t.p.z),1},{float(t.q.w),float(t.q.x),float(t.q.y),float(t.q.z)}};}
            for(auto range:{VRSkeletalMotionRange_WithController,VRSkeletalMotionRange_WithoutController})if(VRDriverInput()->UpdateSkeletonComponent(skeleton,range,bones,31)){live=false;VRDriverLog()->Log("Frame Hands skeleton update rejected; disconnecting");}
        }
        if(!live){smooth.reset();controls.reset();target_valid=false;seen=0;}
#ifdef FRAME_HANDS_DRIVER_PRESENTATION
        if(!live)presentation.reset();
#endif
        last_frame=now;
        for(int n=0;n<7;++n)VRDriverInput()->UpdateBooleanComponent(buttons[n],live&&controls.buttons[n],0);
        VRDriverInput()->UpdateScalarComponent(trigger,live&&controls.buttons[0]?1:0,0);
        VRDriverInput()->UpdateScalarComponent(grip,live&&controls.buttons[1]?1:0,0);
        for(int n=0;n<2;++n)VRDriverInput()->UpdateScalarComponent(axes[n],live?float(controls.axes[n]):0,0);
        bool connected=live;
#ifdef FRAME_HANDS_DRIVER_ROLE_GRACE
        // Ownership is independent of pose validity. Never extend the 80 ms
        // valid-pose budget or renew ownership from duplicate/render frames.
        if(live)last_observation=stability.stamp;
        if(!head_valid||controller_busy)last_observation=0;
        connected=live||(last_observation&&now>=last_observation&&now-last_observation<=750);
#ifdef FRAME_HANDS_DRIVER_INPUT_MODE
        // Explicit Hands mode survives absent observations, but not a dead
        // transport/headset. Missing poses remain honestly invalid.
        if(retain_role&&fresh&&head_valid&&!controller_busy)connected=true;
#endif
        if(connected&&!live){
            // Invalid means unavailable, not a new hand at the tracking origin.
            auto previous=GetPose();next.qRotation=previous.qRotation;
            for(int n=0;n<3;++n)next.vecPosition[n]=previous.vecPosition[n];
        }
#endif
        auto role=connected?(side?TrackedControllerRole_RightHand:TrackedControllerRole_LeftHand):TrackedControllerRole_OptOut;
        if(role!=last_role){VRProperties()->SetInt32Property(props,Prop_ControllerRoleHint_Int32,role);last_role=role;}
        next.deviceIsConnected=connected;next.poseIsValid=live;
        next.result=live?TrackingResult_Running_OK:(connected?TrackingResult_Running_OutOfRange:TrackingResult_Uninitialized);
        {std::lock_guard<std::mutex> guard(lock);pose=next;}
        VRServerDriverHost()->TrackedDevicePoseUpdated(index,next,sizeof next);
    }
private:
#ifdef FRAME_HANDS_DRIVER_DISPLAY
    bool display_hidden=false;
#endif
#ifdef FRAME_HANDS_DRIVER_NAVIGATION
    VRInputComponentHandle_t navigation=0;
#endif
#ifdef FRAME_HANDS_DRIVER_LINK
    VRInputComponentHandle_t pinches[4]{},index_point=0;
#endif
#ifdef FRAME_HANDS_DRIVER_GESTURE_CONTROLS
    bool trigger_down=false,grip_down=false;
#endif
    // pose.txt (palm-axis yaw/pitch/roll + hinge flag) remains as a coarse knob; the exact game correction is the
    // per-side rigid transform in pose_left.txt / pose_right.txt (see frame_model).
    double pose_fix_deg[3]{};bool pose_hinge=true;uint64_t pose_read_at=0;
    fhs::Q corr_q;fhs::V corr_t;bool corr_valid=false;uint64_t corr_read_at=0;
#ifdef FRAME_HANDS_DRIVER_PREDICT
    bool velocity_valid=false;uint64_t velocity_capture=0;double velocity_position[3]{},velocity[3]{};
    double predict_fraction=1.0,velocity_smoothing=.35,rotation_smoothing=1.0;uint64_t predict_read_at=0;
    bool rotation_valid=false;double rotation_prev[4]{1,0,0,0};
#endif
    PropertyContainerHandle_t props=0;VRInputComponentHandle_t buttons[7]{},axes[2]{},trigger=0,grip=0,skeleton=0,haptic=0;
    ETrackedControllerRole last_role=TrackedControllerRole_OptOut;DriverPose_t pose{};std::mutex lock;
};
class Provider final:public IServerTrackedDeviceProvider {
    Device hands[2];InputMailbox mailbox;uint64_t busy_until=0;
#ifdef FRAME_HANDS_DRIVER_NAVIGATION
    fhs::DashboardPinch dashboard_pinch;
#endif
#ifdef FRAME_HANDS_DRIVER_INPUT_MODE
    fhs::ModeSelection selection;uint64_t mode_read_at=0;
    void read_mode(uint64_t now){
        if(now<mode_read_at)return;mode_read_at=now+100;
        auto path=fhs_platform::sibling(input_path,"mode.bin");
        fhs::ModeCommand command;
        bool ok=fhs_platform::read_exact(path,command);
        if(ok&&selection.accept(command)){
            busy_until=0;
            VRDriverLog()->Log(selection.mode==fhs::InputMode::Hands?"Frame Hands mode: hands":selection.mode==fhs::InputMode::Controllers?"Frame Hands mode: controllers":"Frame Hands mode: automatic");
        }
    }
#endif
public:
    EVRInitError Init(IVRDriverContext* context)override{
        VR_INIT_SERVER_DRIVER_CONTEXT(context);
        input_path=fhs_platform::mailbox_path();
        if(input_path.empty())return VRInitError_Driver_Failed;
        for(int side=0;side<2;++side){hands[side].side=side;if(!VRServerDriverHost()->TrackedDeviceAdded(side?"frame-hands-optical-right":"frame-hands-optical-left",TrackedDeviceClass_Controller,&hands[side]))return VRInitError_Driver_Failed;}
        return VRInitError_None;
    }
    void Cleanup()override{VR_CLEANUP_SERVER_DRIVER_CONTEXT();}
    const char* const* GetInterfaceVersions()override{return k_InterfaceVersions;}
    void RunFrame()override{
        uint64_t now=now_ms();InputPacket p;if(read_packet(p))mailbox.accept(p,now);
#ifdef FRAME_HANDS_DRIVER_INPUT_MODE
        read_mode(now);
#endif
        VREvent_t event{};
        while(VRServerDriverHost()->PollNextEvent(&event,sizeof event)){
            if(event.eventType==VREvent_ButtonPress&&event.trackedDeviceIndex!=hands[0].index&&event.trackedDeviceIndex!=hands[1].index){
#ifdef FRAME_HANDS_DRIVER_INPUT_MODE
                auto props=VRProperties()->TrackedDeviceToPropertyContainer(event.trackedDeviceIndex);
                if(VRProperties()->GetInt32Property(props,Prop_DeviceClass_Int32)==TrackedDeviceClass_Controller){
                    auto previous=selection.mode;selection.controller_press();busy_until=now+2000;
                    if(previous!=selection.mode)VRDriverLog()->Log("Frame Hands mode: controllers (physical button)");
                }
#else
                busy_until=now+2000;
#endif
            }
            // Bare optical hands have no haptic actuator; consume haptic events without fabricating feedback.
        }
        TrackedDevicePose_t poses[k_unMaxTrackedDeviceCount]{};VRServerDriverHost()->GetRawTrackedDevicePoses(0,poses,k_unMaxTrackedDeviceCount);
        for(uint32_t i=1;i<k_unMaxTrackedDeviceCount;++i){
            if(i==hands[0].index||i==hands[1].index||!poses[i].bPoseIsValid||!poses[i].bDeviceIsConnected)continue;
            auto props=VRProperties()->TrackedDeviceToPropertyContainer(i);
            if(VRProperties()->GetInt32Property(props,Prop_DeviceClass_Int32)!=TrackedDeviceClass_Controller)continue;
            auto v=poses[i].vVelocity,a=poses[i].vAngularVelocity;
            if(fhs::V{v.v[0],v.v[1],v.v[2]}.len()>.35||fhs::V{a.v[0],a.v[1],a.v[2]}.len()>2)busy_until=now+2000;
        }
#ifdef FRAME_HANDS_DRIVER_INPUT_MODE
        bool manual_hands=selection.mode==fhs::InputMode::Hands;
        bool yield=selection.mode==fhs::InputMode::Controllers||(selection.mode==fhs::InputMode::Auto&&now<busy_until);
#ifdef FRAME_HANDS_DRIVER_MODEL
        bool dashboard_click=false;
#ifdef FRAME_HANDS_DRIVER_NAVIGATION
        fhs::DashboardContext dashboard;
        const bool ui=fhs_platform::read_exact(fhs_platform::sibling(input_path,"dashboard.bin"),dashboard)&&fhs::dashboard_context(dashboard,now);
        dashboard_click=dashboard_pinch.update(mailbox.packet,now,ui&&manual_hands&&!yield&&poses[0].bPoseIsValid&&poses[0].bDeviceIsConnected);
#endif
        fhs::DisplayLease display{};
        bool display_valid=false;
#ifdef FRAME_HANDS_DRIVER_DISPLAY
        display_valid=fhs_platform::read_exact(fhs_platform::sibling(input_path,"display.bin"),display);
#endif
        for(auto& h:hands)h.frame_model(mailbox.packet,mailbox.fresh(now),poses[0].bPoseIsValid&&poses[0].bDeviceIsConnected,yield,now,manual_hands,dashboard_click,display_valid&&manual_hands&&fhs::display_lease(display,now,h.side),&poses[0]);
#else
        for(auto& h:hands)h.frame(mailbox.packet,mailbox.fresh(now),poses[0].bPoseIsValid&&poses[0].bDeviceIsConnected,yield,now,manual_hands);
#endif
#else
        for(auto& h:hands)h.frame(mailbox.packet,mailbox.fresh(now),poses[0].bPoseIsValid&&poses[0].bDeviceIsConnected,now<busy_until,now);
#endif
    }
    bool ShouldBlockStandbyMode()override{return false;}
    void EnterStandby()override{}
    void LeaveStandby()override{}
};
Provider provider;
}
extern "C" FHS_EXPORT void* HmdDriverFactory(const char* name,int* error){
    if(name&&!std::strcmp(name,IServerTrackedDeviceProvider_Version))return &provider;
    if(error)*error=VRInitError_Init_InterfaceNotFound;return nullptr;
}
