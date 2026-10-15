// Bounded visual calibration collector: public eye action, no input or cameras.
#include <openvr.h>
#include "gaze_math.h"
#include <chrono>
#include <csignal>
#include <cstdio>
#include <vector>
#include <thread>
#include <cstdint>
#include <cstring>
#include <cstdlib>

namespace {
volatile sig_atomic_t stopped=0;void stop(int){stopped=1;}
using fhg::Vec;using fhg::Angles;
constexpr int W=900,H=600;constexpr double width=1.2,depth=1.5;
// Original compact bitmap lettering, for readable instructions without fonts/dependencies.
const char* alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 /.-";
const uint8_t glyphs[][7]={
{14,17,17,31,17,17,17},{30,17,17,30,17,17,30},{14,17,16,16,16,17,14},{30,17,17,17,17,17,30},{31,16,16,30,16,16,31},{31,16,16,30,16,16,16},{14,17,16,23,17,17,15},{17,17,17,31,17,17,17},{31,4,4,4,4,4,31},{7,2,2,2,18,18,12},{17,18,20,24,20,18,17},{16,16,16,16,16,16,31},{17,27,21,21,17,17,17},{17,25,25,21,19,19,17},{14,17,17,17,17,17,14},{30,17,17,30,16,16,16},{14,17,17,17,21,18,13},{30,17,17,30,20,18,17},{15,16,16,14,1,1,30},{31,4,4,4,4,4,4},{17,17,17,17,17,17,14},{17,17,17,17,17,10,4},{17,17,17,21,21,21,10},{17,17,10,4,10,17,17},{17,17,10,4,4,4,4},{31,1,2,4,8,16,31},
{14,17,19,21,25,17,14},{4,12,4,4,4,4,14},{14,17,1,2,4,8,31},{30,1,1,14,1,1,30},{2,6,10,18,31,2,2},{31,16,16,30,1,1,30},{14,16,16,30,17,17,14},{31,1,2,4,8,8,8},{14,17,17,14,17,17,14},{14,17,17,15,1,1,14},{0,0,0,0,0,0,0},{1,2,2,4,8,8,16},{0,0,0,0,0,6,6},{0,0,0,31,0,0,0}};
struct Canvas {
    std::vector<uint8_t> pixels=std::vector<uint8_t>(W*H*4);
    void pixel(int x,int y,int r,int g,int b){if(x<0||y<0||x>=W||y>=H)return;auto p=&pixels[(y*W+x)*4];p[0]=r;p[1]=g;p[2]=b;p[3]=255;}
    void clear(){for(int y=0;y<H;++y)for(int x=0;x<W;++x)pixel(x,y,12,18,28);}
    void text(const std::string& s,int y,int scale=3){int x=(W-int(s.size())*6*scale)/2;for(char c:s){const char* p=std::strchr(alphabet,c);if(p)for(int row=0;row<7;++row)for(int col=0;col<5;++col)if(glyphs[p-alphabet][row]&(1<<(4-col)))for(int a=0;a<scale;++a)for(int b=0;b<scale;++b)pixel(x+col*scale+a,y+row*scale+b,220,232,248);x+=6*scale;}}
    void draw(int target,Vec point,double progress,bool settling,const std::string& message=""){
        clear();
        if(!message.empty()){text(message,230,4);text("NO CLICKS ARE SENT",285);return;}
        text(target<0?"EYE CALIBRATION":target<9?"CALIBRATE "+std::to_string(target+1)+" / 9":"CHECK "+std::to_string(target-8)+" / 5",20);
        text("LOOK AT THE DOT - KEEP YOUR HEAD STILL",550,2);
        text("NO CLICKING NEEDED",573,2);
        if(target==-2){text("MAGENTA RING SHOWS ESTIMATED GAZE",55,2);return;}
        if(target<0){text("FOLLOW EACH DOT WITH YOUR EYES",245);text("STARTING IN A MOMENT",285);return;}
        double cx=W*.5+point.x*W/width,cy=H*.5-point.y*W/width;
        double ring=settling?20-10*std::clamp(progress,0.,1.):10;
        for(int y=int(cy)-23;y<=int(cy)+23;++y)for(int x=int(cx)-23;x<=int(cx)+23;++x){double d=std::hypot(x-cx,y-cy);if(d<3)pixel(x,y,255,255,255);else if(std::abs(d-ring)<1.5)pixel(x,y,60,210,235);}
        int filled=int(780*std::clamp(progress,0.,1.));for(int y=523;y<529;++y)for(int x=60;x<840;++x)pixel(x,y,x<60+filled?60:35,x<60+filled?210:45,x<60+filled?235:60);
    }
};
struct Sample{double raw[2],target[2];};
Vec fromvr(vr::HmdVector3_t p){return{p.v[0],p.v[1],p.v[2]};}
}
int main(int argc,char** argv){
    std::string manifest,output,result,preview,feedback_profile;int timeout=110;bool verify_only=false;
    for(int i=1;i<argc;++i){std::string a=argv[i];if(a=="--manifest"&&i+1<argc)manifest=argv[++i];else if(a=="--output"&&i+1<argc)output=argv[++i];else if(a=="--result"&&i+1<argc)result=argv[++i];else if(a=="--preview"&&i+1<argc)preview=argv[++i];else if(a=="--timeout"&&i+1<argc)timeout=std::atoi(argv[++i]);else if(a=="--verify-only")verify_only=true;else if(a=="--feedback-profile"&&i+1<argc)feedback_profile=argv[++i];else return 2;}
    if(!preview.empty()){
        Canvas canvas;auto d=fhg::direction({-16,10});canvas.draw(1,d*(-depth/d.z),.65,false);
        std::ofstream out(preview,std::ios::binary);out<<"P6\n"<<W<<' '<<H<<"\n255\n";
        for(int i=0;i<W*H;++i)out.write(reinterpret_cast<char*>(&canvas.pixels[i*4]),3);
        return out?0:1;
    }
    if(timeout<1||timeout>120||manifest.empty()||(output.empty()&&result.empty()))return 2;
    fhg::Calibration feedback_calibration;
    if(!feedback_profile.empty()&&!feedback_calibration.load(feedback_profile,fhg::eye_session(),time(nullptr),true)){std::fprintf(stderr,"Preview profile does not match the eye-tracker session\n");return 2;}
    signal(SIGINT,stop);signal(SIGTERM,stop);
    vr::EVRInitError error;auto sys=vr::VR_Init(&error,vr::VRApplication_Overlay);if(error||!sys)return 1;
    auto overlay=vr::VROverlay();auto input=vr::VRInput();vr::VROverlayHandle_t panel=0,dot=0,feedback=0;vr::VRActionSetHandle_t set=0;vr::VRActionHandle_t eye_action=0;
    auto finish=[&](int code){if(feedback)overlay->DestroyOverlay(feedback);if(dot)overlay->DestroyOverlay(dot);if(panel)overlay->DestroyOverlay(panel);vr::VR_Shutdown();return code;};
    if(input->SetActionManifestPath(manifest.c_str())||input->GetActionSetHandle("/actions/main",&set)||input->GetActionHandle("/actions/main/in/gaze",&eye_action))return finish(1);
    if(overlay->CreateOverlay("framehands.gaze.calibration","Frame Hands gaze calibration",&panel))return finish(1);
    vr::HmdMatrix34_t pose{};pose.m[0][0]=pose.m[1][1]=pose.m[2][2]=1;pose.m[2][3]=-depth;
    if(overlay->SetOverlayWidthInMeters(panel,width)||overlay->SetOverlayTransformTrackedDeviceRelative(panel,0,&pose)||overlay->SetOverlayInputMethod(panel,vr::VROverlayInputMethod_None)||overlay->SetOverlaySortOrder(panel,10000))return finish(1);
    Canvas canvas;canvas.draw(-2,{},0,false,result.empty()?"":result=="pass"?"CALIBRATION CHECK PASSED":"CALIBRATION NEEDS ANOTHER TRY");
    if(overlay->SetOverlayRaw(panel,canvas.pixels.data(),W,H,4)||overlay->ShowOverlay(panel))return finish(1);
    // Upload each texture once. Repeated SetOverlayRaw uploads failed with
    // RequestFailed on this runtime; target changes now update only its pose.
    std::vector<uint8_t> dot_pixels(64*64*4,0),ring_pixels(64*64*4,0);
    if(result.empty()){
        if(overlay->CreateOverlay("framehands.gaze.calibration.dot","Frame Hands fixation dot",&dot))return finish(1);
        for(int y=0;y<64;++y)for(int x=0;x<64;++x){double radius=std::hypot(x-31.5,y-31.5);if(radius<3||std::abs(radius-14)<1.5){auto p=&dot_pixels[(y*64+x)*4];p[0]=p[1]=p[2]=255;p[3]=255;}}
        if(overlay->SetOverlayRaw(dot,dot_pixels.data(),64,64,4)||overlay->SetOverlayWidthInMeters(dot,.04f)||overlay->SetOverlayInputMethod(dot,vr::VROverlayInputMethod_None)||overlay->SetOverlaySortOrder(dot,10001))return finish(1);
        if(overlay->CreateOverlay("framehands.gaze.calibration.feedback","Frame Hands recorded gaze estimate",&feedback))return finish(1);
        for(int y=0;y<64;++y)for(int x=0;x<64;++x)if(std::abs(std::hypot(x-31.5,y-31.5)-20)<1.5){auto p=&ring_pixels[(y*64+x)*4];p[0]=235;p[1]=110;p[2]=235;p[3]=180;}
        if(overlay->SetOverlayRaw(feedback,ring_pixels.data(),64,64,4)||overlay->SetOverlayWidthInMeters(feedback,.045f)||overlay->SetOverlayInputMethod(feedback,vr::VROverlayInputMethod_None)||overlay->SetOverlaySortOrder(feedback,10002))return finish(1);
    }
    vr::VRActiveActionSet_t active{};active.ulActionSet=set;
    const auto session=fhg::eye_session();
    std::ofstream out;if(result.empty()){out.open(output);if(!out)return finish(1);out<<"{\"session\":\""<<session<<"\",\"created\":"<<time(nullptr)<<",\"sensor_timestamp_available\":false,\"input_injected\":false}\n";}
    // Shuffled grid avoids learning a simple sweep; holdout locations differ from all fit points.
    Angles targets[]={{0,0},{-16,10},{16,-10},{-16,-10},{16,10},{0,-10},{-16,0},{0,10},{16,0},{-9,6},{9,-6},{-9,-6},{9,6},{3,2}};
    if(verify_only){const Angles checks[]={{-12,3},{12,-3},{-4,-7},{4,7},{0,-2}};for(int i=0;i<5;++i)targets[i+9]=checks[i];}
    auto start=std::chrono::steady_clock::now(),phase=start,gap_start=start,feedback_until=start;bool began=false,head_gap=false,showing_feedback=false;
    std::string failure="timeout";
    int index=-1,drawn=-1,attempt=0,total=0,eye_good=0,motion_good=0;std::vector<Sample> samples;auto tick=start;
    while(!stopped){
        auto now=std::chrono::steady_clock::now();double elapsed=std::chrono::duration<double>(now-start).count();if(elapsed>timeout)break;
        vr::VREvent_t event{};while(sys->PollNextEvent(&event,sizeof event))if(event.eventType==vr::VREvent_Quit)stopped=1;
        if(stopped){failure="interrupted_or_runtime_quit";break;}
        if(!result.empty()){if(elapsed>4)return finish(0);std::this_thread::sleep_for(std::chrono::milliseconds(20));continue;}
        if(showing_feedback){
            if(now<feedback_until){std::this_thread::sleep_for(std::chrono::milliseconds(11));continue;}
            overlay->HideOverlay(feedback);showing_feedback=false;
            if(index==14){bool same=fhg::eye_session()==session;out<<"{\"completed\":"<<(same?"true":"false")<<",\"points\":"<<(verify_only?5:14)<<"}\n";out.close();return finish(same?0:3);}
            phase=now;
        }
        vr::TrackedDevicePose_t head{};sys->GetDeviceToAbsoluteTrackingPose(vr::TrackingUniverseRawAndUncalibrated,0,&head,1);
        if(began&&(!head.bPoseIsValid||!head.bDeviceIsConnected)){
            if(!head_gap){gap_start=now;head_gap=true;out<<"{\"head_tracking_gap\":true,\"point\":"<<index<<"}\n";}
            if(std::chrono::duration<double>(now-gap_start).count()>2){failure="head_tracking_lost_over_two_seconds";break;}
            // Discard this dot's partial sample window and allow a short recovery.
            phase=now;total=eye_good=motion_good=0;samples.clear();
            std::this_thread::sleep_for(std::chrono::milliseconds(11));continue;
        }
        head_gap=false;
        int ue=input->UpdateActionState(&active,sizeof active,1);vr::VREyeTrackingData_t eye{};
        int ee=input->GetEyeTrackingDataRelativeToNow(eye_action,vr::TrackingUniverseRawAndUncalibrated,0,&eye,sizeof eye);
        bool valid=!ue&&!ee&&head.bPoseIsValid&&head.bDeviceIsConnected&&eye.bActive&&eye.bTracked&&eye.bValid;
        Vec local_origin{},local_direction{};
        if(valid){local_origin=fhg::inverse_point(head.mDeviceToAbsoluteTracking,fromvr(eye.vGazeOrigin));local_direction=fhg::inverse_rotate(head.mDeviceToAbsoluteTracking,fromvr(eye.vGazeTarget)-fromvr(eye.vGazeOrigin)).unit();valid=local_origin.finite()&&local_direction.finite()&&local_direction.length()>.99&&fromvr(head.vAngularVelocity).length()<.25;}
        if(!began){if(valid){began=true;phase=now;}else phase=now;}
        double age=std::chrono::duration<double>(now-phase).count();
        if(index<0){if(began&&age>=5){index=verify_only?9:0;phase=now;age=0;}}
        Vec target{};
        if(index>=0){
            Vec d=fhg::direction(targets[index]);target=d*(-depth/d.z);
            if(age>=.9){++total;eye_good+=!ue&&!ee&&eye.bActive&&eye.bTracked&&eye.bValid;motion_good+=fromvr(head.vAngularVelocity).length()<.25;if(valid){auto a=fhg::angles(local_direction),b=fhg::angles(target-local_origin);samples.push_back({{a[0],a[1]},{b[0],b[1]}});}}
            if(age>=2.2){
                std::vector<Angles> raw_angles;for(auto s:samples)raw_angles.push_back({s.raw[0],s.raw[1]});
                double spread=fhg::fixation_spread(raw_angles);
                if(samples.size()<45||samples.size()<.7*total||spread>1.2){
                    out<<"{\"rejected_attempt\":true,\"target_index\":"<<index<<",\"attempt\":"<<attempt+1<<",\"total_polls\":"<<total<<",\"valid_polls\":"<<samples.size()<<",\"eye_valid_polls\":"<<eye_good<<",\"head_still_polls\":"<<motion_good<<",\"fixation_spread_deg\":"<<spread<<"}\n";out.flush();
                    if(++attempt>=3){failure=spread>1.2?"fixation_unstable_at_target":"insufficient_valid_or_head_still_samples_at_target";break;}
                    phase=now;total=eye_good=motion_good=0;samples.clear();age=0;
                }
                else{
                    out<<"{\"point\":"<<index<<",\"validation\":"<<(index>=9?"true":"false")<<",\"total_polls\":"<<total<<",\"valid_polls\":"<<samples.size()<<",\"nominal_target\":["<<targets[index][0]<<","<<targets[index][1]<<"],\"samples\":[";
                    for(size_t j=0;j<samples.size();++j){if(j)out<<',';auto s=samples[j];out<<'['<<s.raw[0]<<','<<s.raw[1]<<','<<s.target[0]<<','<<s.target[1]<<']';}out<<"]}\n";out.flush();
                    auto estimate=fhg::fixation_center(raw_angles);bool estimate_ok=true;
                    if(!feedback_profile.empty()){fhg::Angles corrected{};estimate_ok=feedback_calibration.apply(estimate,corrected);if(estimate_ok)estimate=corrected;}
                    auto ray=fhg::direction(estimate);Vec echo=local_origin+ray*((-depth-local_origin.z)/ray.z);
                    if(estimate_ok&&echo.finite()&&std::abs(echo.x)<1&&std::abs(echo.y)<.7){
                        vr::HmdMatrix34_t echo_pose=pose;echo_pose.m[0][3]=echo.x;echo_pose.m[1][3]=echo.y;echo_pose.m[2][3]=echo.z;
                        if(overlay->SetOverlayTransformTrackedDeviceRelative(feedback,0,&echo_pose)||overlay->ShowOverlay(feedback)){failure="feedback_overlay_error";break;}
                    }
                    // Freeze the sampled median, never a live cursor during collection.
                    out<<"{\"feedback\":true,\"target_index\":"<<index<<",\"corrected\":"<<(!feedback_profile.empty()?"true":"false")<<",\"estimate_deg\":["<<estimate[0]<<','<<estimate[1]<<"]}\n";
                    ++index;attempt=0;total=eye_good=motion_good=0;samples.clear();phase=now;
                    showing_feedback=true;feedback_until=now+std::chrono::milliseconds(800);continue;
                }
            }
        }
        if(index>=0&&drawn!=index){
            vr::HmdMatrix34_t dot_pose=pose;dot_pose.m[0][3]=target.x;dot_pose.m[1][3]=target.y;dot_pose.m[2][3]=target.z;
            auto draw_error=overlay->SetOverlayTransformTrackedDeviceRelative(dot,0,&dot_pose);
            if(!draw_error)draw_error=overlay->SetOverlayColor(dot,index<9?.25f:1.f,index<9?.85f:.65f,index<9?1.f:.25f);
            if(!draw_error)draw_error=overlay->ShowOverlay(dot);
            if(draw_error){failure="dot_transform_error_"+std::to_string(int(draw_error));break;}
            drawn=index;phase=std::chrono::steady_clock::now();
        }
        tick+=std::chrono::microseconds(11111);if(tick<now)tick=now;std::this_thread::sleep_until(tick);
    }
    if(out)out<<"{\"completed\":false,\"points\":"<<std::max(index,0)<<",\"reason\":\""<<failure<<"\"}\n";
    return finish(3);
}
