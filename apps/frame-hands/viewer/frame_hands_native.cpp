// Optical observations only; never injects input or changes runtime settings.
#define XR_USE_TIMESPEC
#include <ctime>
#include <openxr/openxr.h>
#include <openxr/openxr_platform.h>
#include <godot_cpp/classes/open_xr_extension_wrapper.hpp>
#include <godot_cpp/classes/open_xrapi_extension.hpp>
#include <godot_cpp/classes/engine.hpp>
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/godot.hpp>
#include <godot_cpp/variant/dictionary.hpp>
#include <godot_cpp/variant/array.hpp>
#include <godot_cpp/variant/packed_vector3_array.hpp>
#include <godot_cpp/variant/transform3d.hpp>
#include "fh_hands.h"
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cmath>
#include <cstdio>
#include <cstring>
using namespace godot;

class FrameHandsNative final:public OpenXRExtensionWrapper {
    GDCLASS(FrameHandsNative,OpenXRExtensionWrapper)
    bool time_enabled=false;
    XrInstance instance=XR_NULL_HANDLE;
    XrSpace view=XR_NULL_HANDLE;
    PFN_xrConvertTimespecTimeToTimeKHR convert=nullptr;
    PFN_xrConvertTimeToTimespecTimeKHR convert_back=nullptr;
    PFN_xrCreateReferenceSpace create_space=nullptr;
    PFN_xrDestroySpace destroy_space=nullptr;
    PFN_xrLocateSpace locate=nullptr;
    XrResult view_result=XR_ERROR_SESSION_NOT_RUNNING;
    static int64_t mono_ns(){timespec t{};if(clock_gettime(CLOCK_MONOTONIC,&t))return 0;return int64_t(t.tv_sec)*1000000000+t.tv_nsec;}
protected:
    static void _bind_methods(){ClassDB::bind_method(D_METHOD("sample"),&FrameHandsNative::sample);}
public:
    Dictionary _get_requested_extensions(uint64_t) override {
        Dictionary d;d["XR_KHR_convert_timespec_time"]=uint64_t(&time_enabled);return d;
    }
    void _on_instance_created(uint64_t handle) override {
        instance=reinterpret_cast<XrInstance>(handle);
        auto api=get_openxr_api();
        if(time_enabled)convert=reinterpret_cast<PFN_xrConvertTimespecTimeToTimeKHR>(api->get_instance_proc_addr("xrConvertTimespecTimeToTimeKHR"));
        if(time_enabled)convert_back=reinterpret_cast<PFN_xrConvertTimeToTimespecTimeKHR>(api->get_instance_proc_addr("xrConvertTimeToTimespecTimeKHR"));
        create_space=reinterpret_cast<PFN_xrCreateReferenceSpace>(api->get_instance_proc_addr("xrCreateReferenceSpace"));
        destroy_space=reinterpret_cast<PFN_xrDestroySpace>(api->get_instance_proc_addr("xrDestroySpace"));
        locate=reinterpret_cast<PFN_xrLocateSpace>(api->get_instance_proc_addr("xrLocateSpace"));
    }
    void _on_session_created(uint64_t handle) override {
        if(!create_space)return;
        XrReferenceSpaceCreateInfo ci{XR_TYPE_REFERENCE_SPACE_CREATE_INFO};
        ci.referenceSpaceType=XR_REFERENCE_SPACE_TYPE_VIEW;ci.poseInReferenceSpace.orientation.w=1;
        view_result=create_space(reinterpret_cast<XrSession>(handle),&ci,&view);
    }
    void _on_session_destroyed() override {if(view && destroy_space)destroy_space(view);view=XR_NULL_HANDLE;}
    void _on_instance_destroyed() override {
        instance=XR_NULL_HANDLE;view=XR_NULL_HANDLE;convert=nullptr;convert_back=nullptr;create_space=nullptr;
        destroy_space=nullptr;locate=nullptr;time_enabled=false;
    }
    Dictionary sample() {
        Dictionary out;out["valid"]=false;out["reason"]="OpenXR capture-time transform unavailable";
        const int64_t sampled_now=mono_ns();
        out["now_seconds"]=double(sampled_now)/1e9;
        out["display_seconds"]=double(sampled_now)/1e9;
        out["view_result"]=int64_t(view_result);
        if(!instance || !view || !convert || !locate)return out;
        if(convert_back){
            const XrTime display=get_openxr_api()->get_predicted_display_time();timespec target{};
            if(display>0 && XR_SUCCEEDED(convert_back(instance,display,&target))){
                const int64_t ns=int64_t(target.tv_sec)*1000000000+target.tv_nsec;
                if(ns>=sampled_now && ns-sampled_now<=100000000)out["display_seconds"]=double(ns)/1e9;
            }
        }
        char path[128];std::snprintf(path,sizeof(path),"/run/user/%u/frametop-hands/hands",unsigned(getuid()));
        int fd=open(path,O_RDONLY|O_CLOEXEC|O_NOFOLLOW);
        out["reason"]="Tracker is not publishing";
        if(fd<0)return out;
        fh_hands_t packet{};uint64_t before=0,after=0;struct stat st{};
        bool ok=fstat(fd,&st)==0 && S_ISREG(st.st_mode) && st.st_uid==getuid() && st.st_size==sizeof(packet);
        if(ok)ok=pread(fd,&before,sizeof(before),16)==sizeof(before) &&
                 pread(fd,&packet,sizeof(packet),0)==sizeof(packet) &&
                 pread(fd,&after,sizeof(after),16)==sizeof(after);
        close(fd);
        out["reason"]="Incomplete or invalid tracking packet";
        if(!ok || !before || before!=after || before!=packet.seq || before%2 ||
            std::memcmp(packet.magic,FH_HANDS_MAGIC,8) || packet.version!=FH_HANDS_VERSION ||
            packet.size!=sizeof(packet) || packet.nhands>2 || packet.ncapsules>64)return out;
        const int64_t now=mono_ns(),capture=int64_t(packet.capture_ns),published=int64_t(packet.publish_ns);
        out["reason"]="Stale hand observations";
        if(capture<=0 || published<capture || now<published || now-capture>100000000)return out;
        out["age_ms"]=double(now-capture)/1e6;out["pipeline_ms"]=double(published-capture)/1e6;
        out["sequence"]=int64_t(packet.seq);
        out["capture_seconds"]=double(capture)/1e9;
        bool sides[2]{};
        for(unsigned h=0;h<packet.nhands;++h){const auto& hand=packet.hands[h];
            unsigned side=hand.flags&FH_HAND_RIGHT?1:0;
            if((hand.flags&~3u) || sides[side] || !hand.id || !std::isfinite(hand.confidence) ||
                hand.confidence<0 || hand.confidence>1 || (h && hand.id==packet.hands[0].id))return out;
            sides[side]=true;
            for(auto& point:hand.pts)for(float x:point)if(!std::isfinite(x)||std::abs(x)>5)return out;
        }
        timespec ts{time_t(capture/1000000000),long(capture%1000000000)};XrTime time{};
        out["reason"]="Capture clock conversion failed";
        if(XR_FAILED(convert(instance,&ts,&time)))return out;
        XrSpace base=reinterpret_cast<XrSpace>(get_openxr_api()->get_play_space());
        if(!base)return out;
        XrSpaceLocation location{XR_TYPE_SPACE_LOCATION};XrResult result=locate(view,base,time,&location);
        out["pose_result"]=int64_t(result);out["pose_flags"]=int64_t(location.locationFlags);
        out["reason"]="Head pose at exposure is not tracked";
        constexpr XrSpaceLocationFlags required=XR_SPACE_LOCATION_POSITION_VALID_BIT|XR_SPACE_LOCATION_ORIENTATION_VALID_BIT|
            XR_SPACE_LOCATION_POSITION_TRACKED_BIT|XR_SPACE_LOCATION_ORIENTATION_TRACKED_BIT;
        if(XR_FAILED(result) || (location.locationFlags&required)!=required)return out;
        const auto& q=location.pose.orientation;const auto& p=location.pose.position;
        const double norm=double(q.x)*q.x+double(q.y)*q.y+double(q.z)*q.z+double(q.w)*q.w;
        if(!std::isfinite(norm)||std::abs(norm-1)>.002||!std::isfinite(p.x)||!std::isfinite(p.y)||!std::isfinite(p.z))return out;
        Transform3D head(Basis(Quaternion(q.x,q.y,q.z,q.w).normalized()),Vector3(p.x,p.y,p.z));
        Array hands;
        for(unsigned h=0;h<packet.nhands;++h){const auto& hand=packet.hands[h];Dictionary info;PackedVector3Array points;
            for(auto& point:hand.pts)points.append(head.xform(Vector3(point[0],point[1],point[2])));
            info["right"]=bool(hand.flags&FH_HAND_RIGHT);info["stereo"]=bool(hand.flags&FH_HAND_STEREO);
            info["id"]=int64_t(hand.id);info["points"]=points;hands.append(info);
        }
        out["valid"]=true;out["reason"]=packet.nhands?"Optical observations":"No hands detected";out["hands"]=hands;
        return out;
    }
};
static FrameHandsNative* wrapper=nullptr;
static void initialize(ModuleInitializationLevel level){
    if(level!=MODULE_INITIALIZATION_LEVEL_CORE)return;
    GDREGISTER_CLASS(FrameHandsNative);wrapper=memnew(FrameHandsNative);wrapper->register_extension_wrapper();
    Engine::get_singleton()->register_singleton("FrameHandsNative",wrapper);
}
static void terminate(ModuleInitializationLevel level){
    if(level==MODULE_INITIALIZATION_LEVEL_SCENE)Engine::get_singleton()->unregister_singleton("FrameHandsNative");
}
extern "C" GDExtensionBool GDE_EXPORT frame_hands_init(GDExtensionInterfaceGetProcAddress proc,GDExtensionClassLibraryPtr library,GDExtensionInitialization* init){
    GDExtensionBinding::InitObject object(proc,library,init);object.register_initializer(initialize);
    object.register_terminator(terminate);object.set_minimum_library_initialization_level(MODULE_INITIALIZATION_LEVEL_CORE);return object.init();
}
