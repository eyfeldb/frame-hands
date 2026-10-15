// Read the actual application-facing skeleton; never recompute its retargeting.
#define NOMINMAX
#include <windows.h>
#include <openvr.h>
#include "sample_input.h"
#include "stability.h"
#include "trace.h"
#include "action_watchdog.h"
#include <cstdio>
#include <string>
#include <thread>
#include <chrono>
using namespace vr;
using fhs::V;
template<class T>T symbol(HMODULE lib,const char* name){auto raw=GetProcAddress(lib,name);T result=nullptr;static_assert(sizeof raw==sizeof result);std::memcpy(&result,&raw,sizeof result);return result;}
V transform(const HmdMatrix34_t& m,V p){return {m.m[0][0]*p.x+m.m[0][1]*p.y+m.m[0][2]*p.z+m.m[0][3],m.m[1][0]*p.x+m.m[1][1]*p.y+m.m[1][2]*p.z+m.m[1][3],m.m[2][0]*p.x+m.m[2][1]*p.y+m.m[2][2]*p.z+m.m[2][3]};}
V untransform(const HmdMatrix34_t& m,V p){p=p-V{m.m[0][3],m.m[1][3],m.m[2][3]};return {m.m[0][0]*p.x+m.m[1][0]*p.y+m.m[2][0]*p.z,m.m[0][1]*p.x+m.m[1][1]*p.y+m.m[2][1]*p.z,m.m[0][2]*p.x+m.m[1][2]*p.y+m.m[2][2]*p.z};}
using fhs::View;
struct Diagnostics{unsigned pose_error=0,inactive=0,disconnected=0,invalid_pose=0,skeleton_error=0,skeleton_inactive=0,bone_error=0,source_without_pose=0,source_without_skeleton=0;int last_pose_error=0,last_skeleton_error=0,last_bone_error=0;};
void point(float* out,V v){out[0]=float(v.x);out[1]=float(v.y);out[2]=float(v.z);}
bool read(const char* path,fhs::Packet& p){auto f=CreateFileA(path,GENERIC_READ,FILE_SHARE_READ|FILE_SHARE_WRITE|FILE_SHARE_DELETE,nullptr,OPEN_EXISTING,0,nullptr);if(f==INVALID_HANDLE_VALUE)return false;DWORD n=0;LARGE_INTEGER size{};bool ok=GetFileSizeEx(f,&size)&&size.QuadPart==sizeof p&&ReadFile(f,&p,sizeof p,&n,nullptr)&&n==sizeof p;CloseHandle(f);return ok;}
int main(int argc,char** argv){
    if(argc!=6&&argc!=7)return 2; // DLL, manifest, input, output, seconds, optional stop file
    HMODULE lib=LoadLibraryA(argv[1]);if(!lib)return 3;
    auto init=symbol<uint32_t(*)(EVRInitError*,EVRApplicationType,const char*)>(lib,"VR_InitInternal2");auto iface=symbol<void*(*)(const char*,EVRInitError*)>(lib,"VR_GetGenericInterface");auto shutdown=symbol<void(*)()>(lib,"VR_ShutdownInternal");if(!init||!iface||!shutdown)return 4;
    EVRInitError error{};init(&error,VRApplication_Background,nullptr);if(error){std::printf("init_error=%d\n",int(error));return 5;}
    auto system=static_cast<IVRSystem*>(iface(IVRSystem_Version,&error));auto input=static_cast<IVRInput*>(iface(IVRInput_Version,&error));if(!system||!input)return 6;
    EVRInitError overlay_error{};
    auto overlay=static_cast<IVROverlay*>(iface(IVROverlay_Version,&overlay_error));
    std::printf("overlay_interface_error=%d available=%d\n",int(overlay_error),int(overlay!=nullptr));
    if(auto e=input->SetActionManifestPath(argv[2])){std::printf("manifest_error=%d\n",int(e));shutdown();return 7;}
    VRActionSetHandle_t set{};
    if(auto e=input->GetActionSetHandle("/actions/main",&set)){std::printf("action_set_error=%d\n",int(e));shutdown();return 7;}
    VRActiveActionSet_t active{};active.ulActionSet=set;
    VRActionHandle_t skeleton[2]{},pose[2]{};
    for(int side=0;side<2;++side){
        auto se=input->GetActionHandle(side?"/actions/main/in/skeletonright":"/actions/main/in/skeletonleft",&skeleton[side]);
        auto pe=input->GetActionHandle(side?"/actions/main/in/controllerright":"/actions/main/in/controllerleft",&pose[side]);
        if(se||pe){std::printf("action_handle_error side=%d skeleton=%d pose=%d\n",side,int(se),int(pe));shutdown();return 7;}
    }
    auto end=GetTickCount64()+1000*std::clamp(std::atoi(argv[5]),5,400);std::string temp=std::string(argv[4])+".tmp";unsigned frames=0,raw[2]{},rendered[2]{},drops=0;
    fhs::SampleInput source;unsigned head_missing=0,captured_head_missing=0,action_errors=0;
    bool preview=GetEnvironmentVariableA("FRAME_HANDS_STABILITY_PREVIEW",nullptr,0)>0;
    fhs::Stability stability[2];Diagnostics diagnostics[2];unsigned candidate_frames[2]{},held_frames[2]{};
    // Query actual devices independently of action bindings. This separates a
    // missing hand driver pose from an application-facing action selection error.
    int device_kind[k_unMaxTrackedDeviceCount]{}; // 0 unknown, 1 other controller, 2/3 optical L/R
    unsigned optical_connected[2]{},optical_valid[2]{},other_controller_motion_frames=0;
    uint64_t inventory_at=0,action_audit_at=0;
    fhs::ActionWatchdog action_watchdog;int diagnostic_exit=0;
    const bool require_steam=GetEnvironmentVariableA("FRAME_HANDS_REQUIRE_STEAM",nullptr,0)>0;
    const uint64_t started=GetTickCount64();
    uint64_t event_at[2]={started,started};std::string previous_event[2];unsigned event_count=0;
    HANDLE trace=INVALID_HANDLE_VALUE;bool trace_failed=false;uint64_t trace_frames=0;
    if(const char* trace_path=std::getenv("FRAME_HANDS_TRACE")){
        trace=CreateFileA(trace_path,GENERIC_WRITE,FILE_SHARE_READ,nullptr,CREATE_NEW,FILE_ATTRIBUTE_NORMAL,nullptr);
        if(trace==INVALID_HANDLE_VALUE){std::printf("trace_open_error=%lu\n",GetLastError());shutdown();return 8;}
    }
    std::setvbuf(stdout,nullptr,_IOLBF,0);
    while(GetTickCount64()<end&&(argc!=7||GetFileAttributesA(argv[6])==INVALID_FILE_ATTRIBUTES)){
        View view;view.time=GetTickCount64();fhs::Packet incoming;
        bool read_ok=read(argv[3],incoming);if(read_ok)source.observe(incoming,view.time);
        const auto& packet=source.mailbox.packet;bool valid=source.mailbox.fresh(view.time);
        TrackedDevicePose_t devices[k_unMaxTrackedDeviceCount]{};
        system->GetDeviceToAbsoluteTrackingPose(TrackingUniverseRawAndUncalibrated,0,devices,k_unMaxTrackedDeviceCount);
        const auto& head=devices[0];
        TrackedDevicePose_t captured{};
        if(view.time>=inventory_at){
            inventory_at=view.time+1000;
            for(uint32_t i=1;i<k_unMaxTrackedDeviceCount;++i){
                device_kind[i]=0;if(system->GetTrackedDeviceClass(i)!=TrackedDeviceClass_Controller)continue;
                char serial[128]{};ETrackedPropertyError property_error{};
                system->GetStringTrackedDeviceProperty(i,Prop_SerialNumber_String,serial,sizeof serial,&property_error);
                if(property_error!=TrackedProp_Success)continue;
                device_kind[i]=!std::strcmp(serial,"frame-hands-optical-left")?2:(!std::strcmp(serial,"frame-hands-optical-right")?3:1);
            }
        }
        bool other_controller_motion=false;
        for(uint32_t i=1;i<k_unMaxTrackedDeviceCount;++i){const auto& p=devices[i];
            if(device_kind[i]>=2){int side=device_kind[i]-2;optical_connected[side]+=p.bDeviceIsConnected;optical_valid[side]+=p.bDeviceIsConnected&&p.bPoseIsValid;}
            if(device_kind[i]==1&&p.bDeviceIsConnected&&p.bPoseIsValid){auto v=p.vVelocity,a=p.vAngularVelocity;
                other_controller_motion|=V{v.v[0],v.v[1],v.v[2]}.len()>.35||V{a.v[0],a.v[1],a.v[2]}.len()>2.;}
        }
        other_controller_motion_frames+=other_controller_motion;
        // Binding diagnostics must advance even when the headset is off or untracked.
        const auto action_update_error=input->UpdateActionState(&active,sizeof active,1);
        action_errors+=action_update_error!=VRInputError_None;
        if(head.bPoseIsValid&&head.bDeviceIsConnected){
            if(valid)system->GetDeviceToAbsoluteTrackingPose(TrackingUniverseRawAndUncalibrated,-float(view.time-packet.capture_ms)/1000,&captured,1);
            captured_head_missing+=valid&&!(captured.bPoseIsValid&&captured.bDeviceIsConnected);
            for(int side=0;side<2;++side){auto& out=view.hands[side];auto to_view=[&](V world){return untransform(head.mDeviceToAbsoluteTracking,world);};
                if(valid&&captured.bPoseIsValid&&captured.bDeviceIsConnected&&(packet.hands[side].flags&fhs::Present)){
                    out.id=packet.hands[side].id;out.flags|=1;++raw[side];
                    std::array<V,21> world{};
                    for(int i=0;i<21;++i){auto p=packet.hands[side].points[i];world[i]=transform(captured.mDeviceToAbsoluteTracking,{p[0],p[1],p[2]});point(out.raw[i],to_view(world[i]));}
                    if(preview)stability[side].update(world,out.id,packet.capture_ms,view.time);
                }
                if(preview){auto& estimate=stability[side];
                    if(!valid||(valid&&!(captured.bPoseIsValid&&captured.bDeviceIsConnected)))estimate.reset();
                    if(estimate.visible(view.time)){
                        out.flags|=4;out.candidate_id=estimate.id;++candidate_frames[side];
                        if(!(out.flags&1)||estimate.stamp!=packet.capture_ms){out.flags|=8;++held_frames[side];}
                        for(int i=0;i<21;++i)point(out.candidate[i],to_view(estimate.value[i]));
                    }
                }
                InputPoseActionData_t located{};VRBoneTransform_t bones[31]{};InputSkeletalActionData_t data{};
                auto& d=diagnostics[side];
                int pe=int(input->GetPoseActionDataRelativeToNow(pose[side],TrackingUniverseRawAndUncalibrated,0,&located,sizeof located,k_ulInvalidInputValueHandle));
                int se=int(input->GetSkeletalActionData(skeleton[side],&data,sizeof data));
                int be=data.bActive&&!se?int(input->GetSkeletalBoneData(skeleton[side],VRSkeletalTransformSpace_Model,VRSkeletalMotionRange_WithoutController,bones,31)):0;
                if(pe){++d.pose_error;d.last_pose_error=pe;}else{d.inactive+=!located.bActive;d.disconnected+=located.bActive&&!located.pose.bDeviceIsConnected;d.invalid_pose+=located.bActive&&located.pose.bDeviceIsConnected&&!located.pose.bPoseIsValid;}
                if(se){++d.skeleton_error;d.last_skeleton_error=se;}else d.skeleton_inactive+=!data.bActive;
                if(be){++d.bone_error;d.last_bone_error=be;}
                if(out.flags&1){d.source_without_pose+=pe||!located.bActive||!located.pose.bDeviceIsConnected||!located.pose.bPoseIsValid;
                    d.source_without_skeleton+=se||!data.bActive||be;}
                if(!pe&&located.bActive&&located.pose.bDeviceIsConnected&&located.pose.bPoseIsValid&&!se&&data.bActive&&!be){
                    out.flags|=2;++rendered[side];auto m=located.pose.mDeviceToAbsoluteTracking;
                    for(int i=0;i<31;++i){auto p=bones[i].position;point(out.bones[i],to_view(transform(m,{p.v[0],p.v[1],p.v[2]})));}
                    auto wrist=bones[1];fhs::Q q{wrist.orientation.w,wrist.orientation.x,wrist.orientation.y,wrist.orientation.z};V origin{wrist.position.v[0],wrist.position.v[1],wrist.position.v[2]};
                    for(int a=0;a<3;++a){V axis{};if(a==0)axis.x=.06;if(a==1)axis.y=.06;if(a==2)axis.z=.06;point(out.axes[a],to_view(transform(m,origin+q.rotate(axis))));}
                }
            }
        } else {++head_missing;for(auto& estimate:stability)estimate.reset();}
        if(view.time>=action_audit_at){
            action_audit_at=view.time+1000;
            std::printf("action_audit t_ms=%llu input_available=%d update_error=%d head_valid=%d\n",(unsigned long long)(view.time-started),int(system->IsInputAvailable()),int(action_update_error),int(head.bPoseIsValid&&head.bDeviceIsConnected));
            std::printf("input_focus t_ms=%llu dashboard_visible=%d should_pause=%d system_drawing_controllers=%d\n",(unsigned long long)(view.time-started),overlay?int(overlay->IsDashboardVisible()):-1,int(system->ShouldApplicationPause()),int(system->IsSteamVRDrawingControllers()));
            for(int side=0;side<2;++side){
                InputPoseActionData_t current{},next{};InputSkeletalActionData_t skeletal{};
                auto ce=input->GetPoseActionDataRelativeToNow(pose[side],TrackingUniverseRawAndUncalibrated,0,&current,sizeof current,k_ulInvalidInputValueHandle);
                auto ne=input->GetPoseActionDataForNextFrame(pose[side],TrackingUniverseRawAndUncalibrated,&next,sizeof next,k_ulInvalidInputValueHandle);
                auto se=input->GetSkeletalActionData(skeleton[side],&skeletal,sizeof skeletal);
                std::printf("action_state side=%s current_error=%d current_active=%d current_valid=%d next_error=%d next_active=%d next_valid=%d skeleton_error=%d skeleton_active=%d skeleton_origin=%llu role_device=%u\n",side?"right":"left",int(ce),int(current.bActive),int(current.pose.bPoseIsValid),int(ne),int(next.bActive),int(next.pose.bPoseIsValid),int(se),int(skeletal.bActive),(unsigned long long)skeletal.activeOrigin,system->GetTrackedDeviceIndexForControllerRole(side?TrackedControllerRole_RightHand:TrackedControllerRole_LeftHand));
                for(int type=0;type<2;++type){
                    auto action=type?skeleton[side]:pose[side];const char* kind=type?"skeleton":"pose";
                    VRInputValueHandle_t origins[16]{};auto oe=input->GetActionOrigins(set,action,origins,16);
                    InputBindingInfo_t bindings[8]{};uint32_t count=0;auto be=input->GetActionBindingInfo(action,bindings,sizeof(InputBindingInfo_t),8,&count);
                    std::printf("action_binding side=%s kind=%s origins_error=%d binding_error=%d binding_count=%u\n",side?"right":"left",kind,int(oe),int(be),count);
                    if(!oe)for(auto origin:origins)if(origin){InputOriginInfo_t info{};auto e=input->GetOriginTrackedDeviceInfo(origin,&info,sizeof info);std::printf("action_origin side=%s kind=%s origin=%llu error=%d device=%u device_path=%llu\n",side?"right":"left",kind,(unsigned long long)origin,int(e),info.trackedDeviceIndex,(unsigned long long)info.devicePath);}
                    if(!be)for(uint32_t i=0;i<std::min(count,8u);++i)std::printf("action_binding_path side=%s kind=%s device=%.127s input=%.127s mode=%.127s\n",side?"right":"left",kind,bindings[i].rchDevicePathName,bindings[i].rchInputPathName,bindings[i].rchModeName);
                }
            }
            for(uint32_t i=1;i<k_unMaxTrackedDeviceCount;++i)if(device_kind[i]>=2){
                auto role=system->GetControllerRoleForTrackedDeviceIndex(i);
                auto hint=system->GetInt32TrackedDeviceProperty(i,Prop_ControllerRoleHint_Int32);
                auto priority=system->GetInt32TrackedDeviceProperty(i,Prop_ControllerHandSelectionPriority_Int32);
                std::printf("optical_role device=%u side=%d connected=%d valid=%d role=%d hint=%d priority=%d\n",i,device_kind[i]-2,int(devices[i].bDeviceIsConnected),int(devices[i].bPoseIsValid),int(role),int(hint),int(priority));
            }
        }
        // Transition metadata only: no camera pixels or joint trajectories. A loss
        // interval can now be correlated with rejection, identity or hold expiry.
        for(int side=0;side<2;++side){
            const auto& out=view.hands[side];const auto& s=stability[side];
            const char* reason=!valid?"transport_stale":
                !(head.bPoseIsValid&&head.bDeviceIsConnected)?"head_pose_missing":
                source.last_duplicate?"duplicate_identity":
                !(source.last_flags[side]&fhs::Present)?"source_absent":
                source.last_issue[side]!=fhs::HandIssue::None?fhs::hand_issue_name(source.last_issue[side]):
                !(out.flags&1)?"capture_head_missing":
                preview&&s.pending_stamp==packet.capture_ms?"spike_quarantine":
                preview&&!(out.flags&4)?"capture_age_expired":"measured";
            std::string event=std::string(reason)+":"+std::to_string(source.last_id[side])+":"+
                std::to_string(out.flags&13)+":"+std::to_string(out.candidate_id)+":"+std::to_string(source.bad_segments[side]);
            if(event!=previous_event[side]){
                if(event_count<4096)std::printf("loss_event t_ms=%llu side=%s reason=%s source_id=%u candidate_id=%u raw=%d visible=%d held=%d bad_segments=0x%x previous_ms=%llu capture_age_ms=%llu\n",
                    (unsigned long long)(view.time-started),side?"right":"left",reason,source.last_id[side],out.candidate_id,
                    !!(out.flags&1),!!(out.flags&4),!!(out.flags&8),source.bad_segments[side],
                    (unsigned long long)(view.time-event_at[side]),
                    (unsigned long long)(source.observations&&view.time>=packet.capture_ms?view.time-packet.capture_ms:0));
                else if(event_count==4096)std::puts("loss_event_limit=4096; aggregate counters continue");
                ++event_count;previous_event[side]=event;event_at[side]=view.time;
            }
        }
        FILE* f=std::fopen(temp.c_str(),"wb");if(!f){trace_failed=true;break;}bool ok=std::fwrite(&view,1,sizeof view,f)==sizeof view;std::fclose(f);
        bool published=ok&&MoveFileExA(temp.c_str(),argv[4],MOVEFILE_REPLACE_EXISTING);if(!published)++drops;
        if(trace!=INVALID_HANDLE_VALUE){
            fhs::Trace record;record.time=view.time;record.flags=uint32_t(read_ok)|(uint32_t(valid)<<1)|(uint32_t(preview)<<2);
            record.head_flags=uint32_t(head.bDeviceIsConnected)|(uint32_t(head.bPoseIsValid)<<1);
            record.capture_flags=uint32_t(captured.bDeviceIsConnected)|(uint32_t(captured.bPoseIsValid)<<1);record.published=published;
            std::memcpy(record.head,head.mDeviceToAbsoluteTracking.m,sizeof record.head);
            std::memcpy(record.captured_head,captured.mDeviceToAbsoluteTracking.m,sizeof record.captured_head);
            record.incoming=incoming;record.mailbox=packet;record.view=view;DWORD written=0;
            if(!WriteFile(trace,&record,sizeof record,&written,nullptr)||written!=sizeof record){trace_failed=true;break;}
            ++trace_frames;
        }
        ++frames;
        if(require_steam&&action_watchdog.update(view.time,((view.hands[0].flags|view.hands[1].flags)&1)!=0,((view.hands[0].flags|view.hands[1].flags)&2)!=0)){
            std::puts("STEAM_ACTIONS_INACTIVE: source hands visible for 8 seconds without application-facing skeletons; stopping diagnostic trial");diagnostic_exit=10;break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    DeleteFileA(temp.c_str());std::printf("frames=%u raw_left=%u raw_right=%u steam_left=%u steam_right=%u write_drops=%u head_missing=%u captured_head_missing=%u\n",frames,raw[0],raw[1],rendered[0],rendered[1],drops,head_missing,captured_head_missing);
    std::printf("unique_input_observations=%llu duplicate_identity=%llu\n",(unsigned long long)source.observations,(unsigned long long)source.duplicate_identity);
    for(int side=0;side<2;++side){
        std::printf("loss_event_end side=%s last_interval_ms=%llu state=%s\n",side?"right":"left",(unsigned long long)(GetTickCount64()-event_at[side]),previous_event[side].c_str());
        std::printf("side=%s source_absent=%llu",side?"right":"left",(unsigned long long)source.absent[side]);
        for(int issue=1;issue<7;++issue)std::printf(" %s=%llu",fhs::hand_issue_name(fhs::HandIssue(issue)),(unsigned long long)source.issues[side][issue]);
        std::puts("");
        const auto& d=diagnostics[side];const auto& s=stability[side];
        std::printf("side=%s pose_error=%u(%d) inactive=%u disconnected=%u invalid_pose=%u skeleton_error=%u(%d) skeleton_inactive=%u bone_error=%u(%d) action_errors=%u role_device=%u\n",side?"right":"left",d.pose_error,d.last_pose_error,d.inactive,d.disconnected,d.invalid_pose,d.skeleton_error,d.last_skeleton_error,d.skeleton_inactive,d.bone_error,d.last_bone_error,action_errors,system->GetTrackedDeviceIndexForControllerRole(side?TrackedControllerRole_RightHand:TrackedControllerRole_LeftHand));
        std::printf("side=%s candidate_frames=%u held_frames=%u accepted_observations=%u quarantined=%u filter_fallbacks=%u\n",side?"right":"left",candidate_frames[side],held_frames[side],s.accepted,s.rejected,s.fallbacks);
        std::printf("side=%s optical_device_connected=%u optical_device_valid=%u source_without_action_pose=%u source_without_skeleton=%u other_controller_motion_frames=%u\n",side?"right":"left",optical_connected[side],optical_valid[side],d.source_without_pose,d.source_without_skeleton,other_controller_motion_frames);
    }
    if(trace!=INVALID_HANDLE_VALUE){if(!FlushFileBuffers(trace))trace_failed=true;CloseHandle(trace);}
    std::printf("trace_frames=%llu trace_failed=%d\n",(unsigned long long)trace_frames,trace_failed);
    shutdown();return trace_failed?9:diagnostic_exit;
}
