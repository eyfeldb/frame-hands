#define FRAME_HANDS_DRIVER_MODEL
#include "graphics.h"
#include "../pc-input/driver_platform.h"
#include "../pc-input/input_mode.h"
#include "../pc-input/display_protocol.h"
#include "../pc-input/dashboard_navigation.h"
#include <openvr.h>
#include <filesystem>
#include <chrono>
#include <thread>
#include <csignal>
#ifndef _WIN32
#include <dlfcn.h>
#include <unistd.h>
#endif
using namespace fhdisplay;
namespace fs=std::filesystem;
static volatile std::sig_atomic_t running=1;
static void stop(int){running=0;}
static Matrix identity(){return {{1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1}};}
static Matrix from(const vr::HmdMatrix34_t&m){Matrix r=identity();for(int i=0;i<3;++i)for(int j=0;j<4;++j)r.m[i*4+j]=m.m[i][j];return r;}
static vr::HmdMatrix34_t to(const Matrix&m){vr::HmdMatrix34_t r{};for(int i=0;i<3;++i)for(int j=0;j<4;++j)r.m[i][j]=m.m[i*4+j];return r;}
static Matrix inverse(const Matrix&m){Matrix r=identity();for(int i=0;i<3;++i){for(int j=0;j<3;++j)r.m[4*i+j]=m.m[4*j+i];r.m[4*i+3]=-(r.m[4*i]*m.m[3]+r.m[4*i+1]*m.m[7]+r.m[4*i+2]*m.m[11]);}return r;}
static bool rigid(const Matrix&m){for(float v:m.m)if(!std::isfinite(v))return false;for(int i=0;i<3;++i)for(int j=0;j<3;++j){double d=0;for(int k=0;k<3;++k)d+=m.m[k*4+i]*m.m[k*4+j];if(std::abs(d-(i==j?1.:0.))>.003)return false;}V x{m.m[0],m.m[4],m.m[8]},y{m.m[1],m.m[5],m.m[9]},z{m.m[2],m.m[6],m.m[10]};return x.cross(y).dot(z)>.997;}
// Plane-fixed stereo camera, exactly matching the physical SBS overlay quad.
// plane: distance of the overlay quad in front of the head (metres); the quad is 3.6*plane wide so the mapping
// from the head viewpoint is the same for any plane, but the compositor's reprojection under head motion is only
// exact for content AT the plane, so the plane follows the hands (2026-10-12).
static Matrix projection(const Matrix& eye_to_head,double plane=1.){
    const double half=1.8*plane,distance=plane+eye_to_head.m[11],near_z=.025,far_z=10;
    require(distance>.1,"Eye outside projection plane");
    double l=(-half-eye_to_head.m[3])/distance,r=(half-eye_to_head.m[3])/distance,b=(-half-eye_to_head.m[7])/distance,t=(half-eye_to_head.m[7])/distance;
    Matrix p{};p.m[0]=float(2/(r-l));p.m[2]=float((r+l)/(r-l));p.m[5]=float(2/(t-b));p.m[6]=float((t+b)/(t-b));p.m[10]=float(-(far_z+near_z)/(far_z-near_z));p.m[11]=float(-2*far_z*near_z/(far_z-near_z));p.m[14]=-1;return p;
}
// Projection axes match the plane/head; only physical eye translation is used.
static Matrix parallel_eye(const Matrix& eye){Matrix r=identity();r.m[3]=eye.m[3];r.m[7]=eye.m[7];r.m[11]=eye.m[11];return r;}
static fhs::Skeleton fixture(int side,int variant){
    fhs::Skeleton s;s.position={side?.13:-.13,-.10,-.50};s.rotation={1,0,0,0};
    for(int b=0;b<31;++b)s.bones[b]={fhs::rest[side][b].p,fhs::rest[side][b].q};
    auto rest=fhs::forward(s.bones);std::array<V,21> reference;
    for(int i=0;i<21;++i)reference[i]=rest[fhs::landmark_bone[i]].p;
    s.rotation=(side?Q{0,0,1,0}:Q{})*fhs::palm(reference).inv();
    if(variant)for(int b:{3,4,8,9,10,13,14,15,18,19,20,23,24,25})s.bones[b].q=(s.bones[b].q*Q{.9238795325,0,0,.3826834324}).unit();
    // Reconstruct landmarks and run the same retargeting used by the live driver.
    auto bones=fhs::forward(s.bones);std::array<V,21> points;
    for(int i=0;i<21;++i)points[i]=s.position+s.rotation.rotate(bones[fhs::landmark_bone[i]].p);
    return fhs::retarget_chain(points,side);
}
static void image(const fs::path& path,const std::vector<unsigned char>& pixels){
    std::ofstream out(path,std::ios::binary);out<<"P6\n"<<Graphics::width*2<<' '<<Graphics::height<<"\n255\n";
    for(int y=Graphics::height-1;y>=0;--y)for(int x=0;x<Graphics::width*2;++x){size_t i=(y*Graphics::width*2+x)*4;float a=pixels[i+3]/255.f;int bg=((x/32+y/32)%2)?42:62;for(int c=0;c<3;++c)out.put(char(std::clamp(int(pixels[i+c]+bg*(1-a)),0,255)));}
    require(bool(out),"Proof image write");
}
template<class T>static bool read(const fs::path& p,T& value){return fhs_platform::read_exact(p.native(),value);}
template<class T>static void publish(const fs::path& path,const T& lease){auto temp=path;temp+=".display.tmp";
    std::ofstream out(temp,std::ios::binary|std::ios::trunc);out.write(reinterpret_cast<const char*>(&lease),sizeof lease);out.close();require(bool(out),"Display lease write");
#ifdef _WIN32
    require(MoveFileExW(temp.c_str(),path.c_str(),MOVEFILE_REPLACE_EXISTING),"Display lease publish");
#else
    require(!std::rename(temp.c_str(),path.c_str()),"Display lease publish");
#endif
}
int main(int argc,char** argv){
    // --proof assets output-directory: graphics only, no OpenVR session or live input.
    // openvr-library assets input.bin stop-file: background display only.
    const bool proof=argc==4&&std::string(argv[1])=="--proof";
    const bool check=argc==6&&std::string(argv[5])=="--check";
    if(!proof&&!check&&argc!=5)return 2;
    std::signal(SIGTERM,stop);std::signal(SIGINT,stop);
    vr::IVROverlay* overlay=nullptr;vr::VROverlayHandle_t handle=0;void(*shutdown)()=nullptr;
    fs::path lease,context_path;Graphics graphics;bool initialized=false;
    try{
        graphics.init();std::array<Mesh,2> meshes;
        auto assets=fs::u8path(argv[2]);for(int s=0;s<2;++s)meshes[s].load(assets/(s?"right.mesh":"left.mesh"),s);
        if(proof){
            fs::path output=fs::u8path(argv[3]);fs::create_directories(output);
            double maximum=0;std::array<std::vector<unsigned char>,2> proofs;
            for(int variant=0;variant<2;++variant){
                std::array<std::vector<DrawVertex>,2> draw;
                auto start=std::chrono::steady_clock::now();
                for(int frame=0;frame<30;++frame){for(int s=0;s<2;++s)draw[s]=meshes[s].skin(fixture(s,variant));graphics.begin();for(int eye=0;eye<2;++eye){Matrix e=identity();e.m[3]=eye?.032f:-.032f;graphics.eye(eye,projection(e),inverse(e));for(int s=0;s<2;++s)graphics.draw(meshes[s],draw[s]);}graphics.finish();glFinish();}
                double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count()/30;maximum=std::max(maximum,ms);
                proofs[variant]=graphics.pixels();image(output/(variant?"curl.ppm":"open.ppm"),proofs[variant]);
                size_t visible=0,edge=0;for(size_t i=3;i<proofs[variant].size();i+=4){visible+=proofs[variant][i]>0;edge+=proofs[variant][i]>0&&proofs[variant][i]<255;}
                require(visible>1000&&edge>0,"Glove proof has no geometry or MSAA edges");std::printf("proof variant=%d visible=%zu edges=%zu skin_render_ms=%.3f\n",variant,visible,edge,ms);
            }
            size_t changed=0;for(size_t i=0;i<proofs[0].size();++i)changed+=proofs[0][i]!=proofs[1][i];require(changed>1000,"Glove did not articulate");
            // Intersecting hands use one depth buffer; drawing order cannot choose the winner.
            std::array<std::vector<DrawVertex>,2> crossing;
            for(int side=0;side<2;++side){auto pose=fixture(side,0);pose.position.x=side?.01:-.01;pose.position.z+=side?.025:0;crossing[side]=meshes[side].skin(pose);}
            std::array<std::vector<unsigned char>,2> depth_proof;
            for(int order=0;order<2;++order){graphics.begin();for(int eye=0;eye<2;++eye){Matrix e=identity();e.m[3]=eye?.032f:-.032f;graphics.eye(eye,projection(e),inverse(e));for(int i=0;i<2;++i){int side=order?1-i:i;graphics.draw(meshes[side],crossing[side]);}}graphics.finish();depth_proof[order]=graphics.pixels();}
            require(depth_proof[0]==depth_proof[1],"Crossing depth depends on draw order");image(output/"crossing.ppm",depth_proof[0]);
            std::printf("PASS graphics-only glove articulation changed_channels=%zu max_skin_render_ms=%.3f\n",changed,maximum);return 0;
        }
#ifdef _WIN32
        auto library=LoadLibraryW(fs::u8path(argv[1]).c_str());require(library,"OpenVR library");auto symbol=[&](const char*n){return reinterpret_cast<void*>(GetProcAddress(library,n));};
#else
        auto library=dlopen(argv[1],RTLD_NOW|RTLD_LOCAL);require(library,"OpenVR library");auto symbol=[&](const char*n){return dlsym(library,n);};
#endif
        auto init=reinterpret_cast<uint32_t(*)(vr::EVRInitError*,vr::EVRApplicationType,const char*)>(symbol("VR_InitInternal2"));auto iface=reinterpret_cast<void*(*)(const char*,vr::EVRInitError*)>(symbol("VR_GetGenericInterface"));shutdown=reinterpret_cast<void(*)()>(symbol("VR_ShutdownInternal"));require(init&&iface&&shutdown,"OpenVR entrypoints");
        vr::EVRInitError error{};init(&error,vr::VRApplication_Overlay,nullptr);require(!error,"OpenVR initialize");initialized=true;
        auto system=static_cast<vr::IVRSystem*>(iface(vr::IVRSystem_Version,&error));auto apps=static_cast<vr::IVRApplications*>(iface(vr::IVRApplications_Version,&error));overlay=static_cast<vr::IVROverlay*>(iface(vr::IVROverlay_Version,&error));require(system&&apps&&overlay&&!error,"OpenVR interfaces");
        require(!overlay->CreateOverlay("framehands.animated-gloves.v1","Frame Hands",&handle),"Create hand overlay (another renderer may be running)");
        require(!overlay->SetOverlayInputMethod(handle,vr::VROverlayInputMethod_None),"Overlay input");
        require(!overlay->SetOverlayFlag(handle,vr::VROverlayFlags_SortWithNonSceneOverlays,true),"Overlay group");
        require(!overlay->SetOverlayFlag(handle,vr::VROverlayFlags_IsPremultiplied,true),"Overlay alpha");
        require(!overlay->SetOverlayFlag(handle,vr::VROverlayFlags_SideBySide_Parallel,true),"Overlay stereo");
        require(!overlay->SetOverlayWidthInMeters(handle,3.6f)&&!overlay->SetOverlayTexelAspect(handle,1.f)&&!overlay->SetOverlaySortOrder(handle,40),"Overlay layout");
        if(check){
            graphics.begin();graphics.finish();vr::Texture_t tex{reinterpret_cast<void*>(uintptr_t(graphics.texture)),vr::TextureType_OpenGL,vr::ColorSpace_Gamma};
            auto status=overlay->SetOverlayTexture(handle,&tex);
            std::printf("Hidden compositor texture submission=%d\n",int(status));require(!status,"Hidden texture submission");
            Matrix plane=identity();plane.m[11]=-1;auto transform=to(plane);
            require(!overlay->SetOverlayTransformAbsolute(handle,vr::TrackingUniverseStanding,&transform),"Check transform");
            require(!overlay->ShowOverlay(handle),"Transparent compositor check");
            for(int i=0;i<3;++i)overlay->WaitFrameSync(20);
            vr::HmdMatrix34_t a{},b{};
            auto ca=overlay->GetTransformForOverlayCoordinates(handle,vr::TrackingUniverseStanding,{{0,0}},&a);
            auto cb=overlay->GetTransformForOverlayCoordinates(handle,vr::TrackingUniverseStanding,{{1,1}},&b);
            std::printf("Hidden quad span x=%.4f y=%.4f errors=%d/%d\n",b.m[0][3]-a.m[0][3],b.m[1][3]-a.m[1][3],int(ca),int(cb));
            require(!ca&&!cb&&std::abs(std::abs(b.m[0][3]-a.m[0][3])-3.6)<.01&&std::abs(std::abs(b.m[1][3]-a.m[1][3])-3.6)<.01,"Stereo overlay geometry differs from projection");
            overlay->DestroyOverlay(handle);handle=0;shutdown();initialized=false;return 0;
        }
        auto models=static_cast<vr::IVRRenderModels*>(iface(vr::IVRRenderModels_Version,&error));require(models&&!error,"Render-model interface");
        vr::RenderModel_t* empty=nullptr;vr::EVRRenderModelError loaded=vr::VRRenderModelError_Loading;
        for(int i=0;i<100&&loaded==vr::VRRenderModelError_Loading;++i){loaded=models->LoadRenderModel_Async("{fh_skeletal}fh_display_empty",&empty);if(loaded==vr::VRRenderModelError_Loading)std::this_thread::sleep_for(std::chrono::milliseconds(20));}
        if(empty){models->FreeRenderModel(empty);}require(!loaded,"Display driver resource unavailable; restart runtime after installation");
        fs::path source=fs::u8path(argv[3]),mode_path=source.parent_path()/"mode.bin",stop_path=fs::u8path(argv[4]);lease=source.parent_path()/"display.bin";context_path=source.parent_path()/"dashboard.bin";
        uint32_t last_mask=0;fhs::ModelMailbox mailbox;std::array<Matrix,2> eyes;
        // 2026-10-12 wobble fix. The gloves are a head-oriented quad re-placed every frame: with the head pose sampled
        // at "now" and a quad 1 m away, the compositor reprojects the stale image with the photon-time head pose, so
        // every head micro-motion made the hands swim against the dashboard (the wearer's "wobble" on every tracker
        // configuration; the published joints were static). Now: head pose predicted to photon time, and the quad
        // placed at the hands' own depth so reprojection is exact where the hands are. Knobs without a rebuild in
        // "display.txt" beside the mailbox: "<predict 0|1> <extra ms> <plane m, 0 = follow the hands>"; defaults 1 0 0.
        int knob_predict=1;double knob_extra_ms=0,knob_plane=0;uint64_t knob_read_at=0;double plane=1,width_set=3.6;
        const fs::path knob_path=source.parent_path()/"display.txt";
        while(running&&!fs::exists(stop_path)){
            vr::VREvent_t event{};while(system->PollNextEvent(&event,sizeof event))if(event.eventType==vr::VREvent_Quit){system->AcknowledgeQuit_Exiting();running=0;}
            uint64_t now=fhs_platform::now_ms();fhs::ModeCommand mode{};fhs::ModeSelection select;fhs::ModelPacket packet{};
            if(now-knob_read_at>1000){knob_read_at=now;if(FILE* f=std::fopen(knob_path.string().c_str(),"r")){int a=1;double b=0,c=0;const int got=std::fscanf(f,"%d %lf %lf",&a,&b,&c);if(got>=1&&(a==0||a==1))knob_predict=a;if(got>=2&&b>=-50&&b<=100)knob_extra_ms=b;if(got>=3&&c>=0&&c<=3)knob_plane=c;std::fclose(f);}}
            uint32_t scene=apps->GetCurrentSceneProcessId();char key[256]{};if(scene)apps->GetApplicationKeyByProcessId(scene,key,sizeof key);
            bool dashboard=overlay->IsDashboardVisible();
            bool surface=dashboard||scene==0||!std::strcmp(key,"openvr.tool.steamvr_environments");
            bool allowed=running&&surface&&read(mode_path,mode)&&select.accept(mode)&&mode.mode==fhs::InputMode::Hands;
            if(read(source,packet))mailbox.accept(packet,now);
            allowed=allowed&&mailbox.fresh(now);packet=mailbox.packet;
            float ahead=0;
            if(knob_predict){float vsync=0;if(system->GetTimeSinceLastVsync(&vsync,nullptr)){vr::ETrackedPropertyError pe{};const float freq=system->GetFloatTrackedDeviceProperty(vr::k_unTrackedDeviceIndex_Hmd,vr::Prop_DisplayFrequency_Float,&pe);const float photons=system->GetFloatTrackedDeviceProperty(vr::k_unTrackedDeviceIndex_Hmd,vr::Prop_SecondsFromVsyncToPhotons_Float,nullptr);if(freq>1&&std::isfinite(vsync)&&vsync>=0&&vsync<1)ahead=1.f/freq-vsync+(std::isfinite(photons)?photons:0.f);}ahead+=float(knob_extra_ms/1000);if(!(ahead>=-.05f&&ahead<=.2f))ahead=0;}
            vr::TrackedDevicePose_t head{};if(allowed)system->GetDeviceToAbsoluteTrackingPose(vr::TrackingUniverseRawAndUncalibrated,ahead,&head,1);
            Matrix h=from(head.mDeviceToAbsoluteTracking);allowed=allowed&&head.bDeviceIsConnected&&head.bPoseIsValid&&rigid(h);
            uint32_t mask=0;std::array<std::vector<DrawVertex>,2> draw;double depth_sum=0;int depth_n=0;Matrix head_inverse=inverse(h);
            if(allowed)for(int s=0;s<2;++s){auto& hand=packet.hands[s];if((hand.flags&fhs::Present)&&fhs::model_hand(hand,packet.source_ms)&&now>=hand.capture_ms&&now<hand.until_ms){std::array<V,21> points;for(int i=0;i<21;++i)points[i]={hand.points[i][0],hand.points[i][1],hand.points[i][2]};draw[s]=meshes[s].skin(fhs::retarget_chain(points,s));mask|=1u<<s;
                // depth of the palm centre (wrist/middle-MCP midpoint) in front of the head, for the quad plane
                const V centre=(points[0]+points[9])*.5;const double z=head_inverse.m[8]*centre.x+head_inverse.m[9]*centre.y+head_inverse.m[10]*centre.z+head_inverse.m[11];if(std::isfinite(z)&&-z>.15&&-z<3){depth_sum+=-z;++depth_n;}}}
            if(mask){
                const double target=knob_plane>0?knob_plane:(depth_n?std::clamp(depth_sum/depth_n,.25,1.5):plane);
                plane+=(target-plane)*.3;   // ease the plane so it never jumps
                for(int eye=0;eye<2;++eye)eyes[eye]=parallel_eye(from(system->GetEyeToHeadTransform(vr::EVREye(eye))));
                graphics.begin();for(int eye=0;eye<2;++eye){graphics.eye(eye,projection(eyes[eye],plane),inverse(multiply(h,eyes[eye])));for(int s=0;s<2;++s)if(mask&(1u<<s))graphics.draw(meshes[s],draw[s]);}graphics.finish();
                if(std::abs(3.6*plane-width_set)>.002){width_set=3.6*plane;require(!overlay->SetOverlayWidthInMeters(handle,float(width_set)),"Overlay width");}
                Matrix local=identity();local.m[11]=float(-plane);auto origin=from(system->GetRawZeroPoseToStandingAbsoluteTrackingPose());require(rigid(origin),"Standing origin invalid");auto quad=to(multiply(origin,multiply(h,local)));vr::Texture_t tex{reinterpret_cast<void*>(uintptr_t(graphics.texture)),vr::TextureType_OpenGL,vr::ColorSpace_Gamma};
                require(!overlay->SetOverlayTransformAbsolute(handle,vr::TrackingUniverseStanding,&quad)&&!overlay->SetOverlayTexture(handle,&tex)&&!overlay->ShowOverlay(handle),"Submit hand overlay");
            }else require(!overlay->HideOverlay(handle),"Hide hand overlay");
            fhs::DisplayLease displayed;displayed.stamp_ms=fhs_platform::now_ms();displayed.hands=mask;publish(lease,displayed);
            fhs::DashboardContext context;context.stamp_ms=displayed.stamp_ms;context.enabled=dashboard&&running;publish(context_path,context);if(mask!=last_mask){std::printf("display hands_mask=%u\n",mask);std::fflush(stdout);last_mask=mask;}
            if(mask)overlay->WaitFrameSync(20);else std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        overlay->HideOverlay(handle);overlay->DestroyOverlay(handle);handle=0;std::error_code ec;fs::remove(lease,ec);fs::remove(context_path,ec);shutdown();shutdown=nullptr;return 0;
    }catch(const std::exception& error){std::fprintf(stderr,"Hand display: %s\n",error.what());if(overlay&&handle){overlay->HideOverlay(handle);overlay->DestroyOverlay(handle);}if(!lease.empty()){std::error_code ec;fs::remove(lease,ec);fs::remove(context_path,ec);}if(initialized&&shutdown)shutdown();return 1;}
}
