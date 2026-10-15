#ifdef NDEBUG
#undef NDEBUG
#endif
#include "tracker_rig_hook.h"
#include <tracker.h>
#include <cassert>
#include <fstream>
#include <iostream>
#include <sstream>
int main(int argc,char**argv){try{
 if(argc!=6&&argc!=8&&argc!=10)return 2;
 std::ifstream g(argv[1]),l(argv[2]),r(argv[3]),c(argv[4]),input(argv[5]);
 const auto generic=fhn::HandRig::load(g);std::array<fhn::HandRig,2> profiles{fhn::HandRig::load(l),fhn::HandRig::load(r)};
 const auto cameras=fhn::load_rig_cameras(c);
 std::int64_t stamp;int id,side;input>>stamp>>id>>side;
 std::array<fhn::RigObservation,21> points{};std::array<fhn::PixelObservation,21> pixels{};
 for(auto& p:points){int present;input>>present>>p.point.x>>p.point.y>>p.point.z;p.present=present;}
 for(auto& p:pixels)for(int v=0;v<2;++v){int present;input>>present>>p.point[v].x>>p.point[v].y>>p.sigma[v];p.present[v]=present;}
 assert(input.good());Hand hand;hand.id=11;hand.right_score=side?.99:.01;hand.nviews=2;
 std::vector<Seen> views(2);views[0].cam="slam_left";views[1].cam="slam_right";
 for(auto& v:views){v.hand=hand.id;v.lm.presence=.9;}
 for(int m=0;m<21;++m){int j=fhn::HandRig::mediapipe_to_rig[m];if(j<0){hand.pts[m]={1e6,1e6,1e6};continue;}
  hand.pts[m]={points[j].point.x*.001,points[j].point.y*.001,points[j].point.z*.001};
  for(int v=0;v<2;++v)views[v].lm.pts[m]={pixels[j].point[v].x,pixels[j].point[v].y};}
 std::map<std::string,Camera> native_cameras;
 for(int i=0;i<2;++i){const auto& expected=cameras[i];Camera camera;
  for(int j=0;j<3;++j){auto row=expected.rotation[j];camera.R[0][j]=row.x;camera.R[1][j]=row.y;camera.R[2][j]=row.z;}
  const auto& rows=expected.rotation;double determinant=rows[0].dot(rows[1].cross(rows[2]));
  auto origin=(rows[1].cross(rows[2])*expected.translation.x+rows[2].cross(rows[0])*expected.translation.y+rows[0].cross(rows[1])*expected.translation.z)*(-.001/determinant);
  camera.origin={origin.x,origin.y,origin.z};camera.fx=expected.fx;camera.fy=expected.fy;camera.cx=expected.cx;camera.cy=expected.cy;
  for(int j=0;j<4;++j)camera.k[j]=expected.k[j];auto actual=fhn::TrackerRigAdapter::convert_camera(camera);native_cameras[i?"slam_right":"slam_left"]=camera;
  for(const auto& p:points)if(p.present){fhn::V x,y;assert(expected.project(p.point,x)&&actual.project(p.point,y));assert((x-y).len()<1e-9);}
 }
 auto adapter=[&](){return fhn::TrackerRigAdapter(generic,profiles,cameras);};auto a=adapter();
 const Hand original=hand;const auto original_views=views;
 a.update(std::vector<const Hand*>{&hand},views,1000000000);assert(a.outputs().size()==1&&a.outputs()[0].status=="fitted");
 fhn::RigStage direct;const auto reference=direct.fit(generic,profiles[side],side,points,cameras,pixels,1000000000,hand.id);
 auto difference=[](const fhn::RigStageResult& a,const fhn::RigStageResult& b){double d=(a.refined.fit.pose.wrist.t-b.refined.fit.pose.wrist.t).len();for(int i=0;i<20;++i)d=std::max(d,std::abs(a.refined.fit.pose.angles[i]-b.refined.fit.pose.angles[i]));return d;};
 assert(difference(a.outputs()[0].result,reference)<.001);
 for(int j=0;j<21;++j){assert(hand.pts[j]==original.pts[j]);for(int v=0;v<2;++v)assert(views[v].lm.pts[j]==original_views[v].lm.pts[j]);}
 assert(hand.right_score==original.right_score&&hand.frames==original.frames&&hand.nviews==original.nviews);
 const auto first=a.outputs()[0].result;
 // Thumb CMC has no rig correspondence and must not contaminate the fit.
 hand.pts[1]={-1e8,1e8,0};auto dummy=adapter();dummy.update(std::vector<const Hand*>{&hand},views,1000000000);assert(difference(dummy.outputs()[0].result,first)==0);
 // Two sources of the same labeled side retain independent state, not one side slot.
 Hand other=hand;other.id=22;auto both_views=views;for(auto v:views){v.hand=22;both_views.push_back(v);}
 a.update(std::vector<const Hand*>{&hand,&other},both_views,1033333333);assert(a.outputs().size()==2&&a.outputs()[0].source==11&&a.outputs()[1].source==22);
 assert(difference(a.outputs()[1].result,first)==0);
 // Empty, monocular, missing view and duplicate camera evidence invalidate history.
 std::int64_t now=1066666666;
 for(int mode=0;mode<4;++mode){
  if(mode==0){a.update(std::vector<const Hand*>{},std::vector<Seen>{},now);assert(a.outputs().empty());}
  if(mode==1){hand.nviews=1;a.update(std::vector<const Hand*>{&hand},views,now);assert(a.outputs()[0].status=="monocular");hand.nviews=2;}
  if(mode==2){a.update(std::vector<const Hand*>{&hand},std::vector<Seen>{views[0]},now);assert(a.outputs()[0].status=="missing_or_ambiguous_views");}
  if(mode==3){auto duplicate=views;duplicate.push_back(views[0]);a.update(std::vector<const Hand*>{&hand},duplicate,now);assert(a.outputs()[0].status=="missing_or_ambiguous_views");}
  now+=33333333;a.update(std::vector<const Hand*>{&hand},views,now);assert(difference(a.outputs()[0].result,first)==0);now+=33333333;
 }
 hand.right_score=side?.01:.99;a.update(std::vector<const Hand*>{&hand},views,now);auto changed=adapter();changed.update(std::vector<const Hand*>{&hand},views,now);
 assert(difference(a.outputs()[0].result,changed.outputs()[0].result)==0);
 bool rejected=false;try{a.update(std::vector<const Hand*>{&hand},views,now);}catch(const std::invalid_argument&){rejected=true;}assert(rejected);
 auto invalid=adapter();rejected=false;try{invalid.update(std::vector<const Hand*>{&hand,&hand},views,now);}catch(const std::invalid_argument&){rejected=true;}assert(rejected);
 // Optional whole-pose path: actual tracker types, injected capture contexts, no OpenVR.
 hand=original;views=original_views;
 fhn::TrackerRigAdapter temporal(generic,profiles,cameras,true);now=10000000000LL;
 auto context=[&](std::uint64_t generation){fhn::CaptureHeadPose h;h.valid=true;h.raw_ns=now;h.capture_ns=now-10000000;h.query_ns=now;h.returned_ns=now+1000000;h.generation=generation;return h;};
 auto update=[&](const fhn::CaptureHeadPose* h){temporal.update(std::vector<const Hand*>{&hand},views,now,h);};
 auto head=context(1);update(&head);assert(!temporal.outputs()[0].whole_eligible);
 now+=33333333;head=context(1);update(&head);assert(temporal.outputs()[0].whole_eligible);
 now+=33333333;head=context(1);head.raw_ns--;update(&head);assert(!temporal.outputs()[0].whole_eligible);
 now+=33333333;head=context(1);update(&head);assert(!temporal.outputs()[0].whole_eligible);
 now+=33333333;head=context(1);update(&head);assert(temporal.outputs()[0].whole_eligible);
 now+=33333333;head=context(2);update(&head);assert(!temporal.outputs()[0].whole_eligible);
 now+=33333333;head=context(2);update(&head);assert(temporal.outputs()[0].whole_eligible);
 other=hand;other.id=22;both_views=views;for(auto v:views){v.hand=22;both_views.push_back(v);}
 now+=33333333;head=context(2);temporal.update(std::vector<const Hand*>{&hand,&other},both_views,now,&head);
 for(const auto& o:temporal.outputs())assert(!o.whole_eligible);
 now+=33333333;head=context(2);update(&head);assert(!temporal.outputs()[0].whole_eligible);
 now+=33333333;head=context(2);update(&head);assert(temporal.outputs()[0].whole_eligible);
 now+=33333333;temporal.update(std::vector<const Hand*>{},std::vector<Seen>{},now);
 now+=33333333;head=context(2);update(&head);assert(!temporal.outputs()[0].whole_eligible);
 now+=33333333;head=context(2);update(&head);assert(temporal.outputs()[0].whole_eligible);
 now+=33333333;head=context(2);head.capture_ns=1;update(&head);assert(!temporal.outputs()[0].whole_eligible);
 std::cout<<"PASS opt-in head-pose matching, stale/missing/epoch reset, side ambiguity, absence and recovery\n";
 if(argc>=8){
  auto environment=[](const char* key,const char* value){
#ifdef _WIN32
   _putenv_s(key,value);
#else
   if(*value)setenv(key,value,1);else unsetenv(key);
#endif
  };
  for(const char* key:{"FRAME_HANDS_RIG_WHOLE_POSE","FRAME_HANDS_RIG_REPLAY_HEADS","FRAME_HANDS_RIG_OPENVR","FRAME_HANDS_RIG_SNAPSHOT"})environment(key,"");
  environment("FRAME_HANDS_RIG_DIR","");environment("FRAME_HANDS_RIG_TRACE","");
  {fhn::TrackerRigHook disabled;disabled.configure(native_cameras);assert(!disabled.enabled());disabled.update(std::vector<const Hand*>{&hand},views,now);assert(disabled.outputs().empty());}
  environment("FRAME_HANDS_RIG_DIR",argv[6]);environment("FRAME_HANDS_RIG_TRACE",argv[7]);
  {fhn::TrackerRigHook enabled;enabled.configure(native_cameras);assert(enabled.enabled());enabled.update(std::vector<const Hand*>{&hand},views,now);assert(enabled.outputs().size()==1&&enabled.outputs()[0].status=="fitted");enabled.update(std::vector<const Hand*>{},std::vector<Seen>{},now+33333333);assert(enabled.outputs().empty());}
  std::ifstream trace(argv[7]);std::string line;int lines=0;while(std::getline(trace,line)){++lines;if(lines==2)assert(line.rfind("rig 11 ",0)==0);if(lines==3)assert(line.back()=='0');}assert(lines==3);
  bool refused=false;try{fhn::TrackerRigHook again;again.configure(native_cameras);}catch(const std::invalid_argument&){refused=true;}assert(refused);
  if(argc==10){
   if(std::ifstream(argv[8]).good()||std::ifstream(argv[9]).good())throw std::invalid_argument("Existing replay test output");
   {std::ofstream saved(argv[8]);saved<<"FHHEAD 1 CLOCK_MONOTONIC raw_world_mm "<<std::string(64,'a')<<'\n';
    for(int i=0;i<4;++i)saved<<now+i*33333333LL<<' '<<now-10000000+i*33333333LL<<' '<<(i!=2)<<" 1 0 0 0 0 0 0\n";}
   environment("FRAME_HANDS_RIG_TRACE",argv[9]);environment("FRAME_HANDS_RIG_WHOLE_POSE","1");environment("FRAME_HANDS_RIG_REPLAY_HEADS",argv[8]);
   {fhn::TrackerRigHook replay;replay.configure(native_cameras);
    for(int i=0;i<4;++i){replay.update(std::vector<const Hand*>{&hand},views,now+i*33333333LL);assert(replay.outputs()[0].whole_eligible==(i==1));}
    bool blocked=false;try{replay.capture_context(now,now);}catch(const std::invalid_argument&){blocked=true;}assert(blocked);
    blocked=false;try{replay.publish(now,now,now);}catch(const std::invalid_argument&){blocked=true;}assert(blocked);
   }
   std::ifstream captured(argv[9]);int heads=0,whole=0;while(std::getline(captured,line)){if(line.rfind("head_recorded ",0)==0)++heads;if(line.rfind("whole ",0)==0)++whole;}
   assert(heads==4&&whole==4);
   for(const char* key:{"FRAME_HANDS_RIG_WHOLE_POSE","FRAME_HANDS_RIG_REPLAY_HEADS"})environment(key,"");
   std::cout<<"PASS recorded hook without OpenVR, exact timestamps, missing join recovery, live capture/publication refused\n";
  }
  environment("FRAME_HANDS_RIG_DIR","");environment("FRAME_HANDS_RIG_TRACE","");
  std::cout<<"PASS optional hook disabled/enabled, fitted and empty frame trace, existing trace refused\n";
 }
 std::cout<<"PASS actual tracker types; camera conversion, mapping, no input mutation, missing thumb CMC, duplicate sides, loss/mono/view resets, side reset, timestamps and duplicate IDs\n";return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 7;}}
