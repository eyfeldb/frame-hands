// Isolated saved-model observation fitting. No device/runtime or input state.
#include "rig_stage.h"
#include "finger_correspondence.h"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
struct Track {int side=-1;fhn::RigStage stage;};
bool triangulate(const std::array<fhn::RigCamera,2>& cameras,const fhn::PixelObservation& pixel,fhn::V& point){
 if(!pixel.present[0]||!pixel.present[1])return false;
 std::array<fhn::StereoRay,2> rays;
 for(int v=0;v<2;++v)if(!fhn::rig_pixel_ray(cameras[v],pixel.point[v],rays[v]))return false;
 auto w=rays[0].origin-rays[1].origin;double b=rays[0].direction.dot(rays[1].direction),d=rays[0].direction.dot(w),e=rays[1].direction.dot(w),den=1-b*b;
 if(den<1e-8)return false;double t=(b*e-d)/den,u=(e-b*d)/den;
 if(t<=0||u<=0)return false;
 point=(rays[0].origin+rays[0].direction*t+rays[1].origin+rays[1].direction*u)*.5;return point.finite();
}
struct FitOutput {fhn::RigStageResult fitted{};fhn::CorrespondenceResult corrected{};int stereo=0;};
FitOutput fit_observations(fhn::RigStage& stage,const fhn::HandRig& reference,const fhn::HandRig& rig,
 const std::array<fhn::RigCamera,2>& cameras,const std::array<fhn::PixelObservation,21>& pixels,std::int64_t stamp,int source,int side){
 FitOutput out;std::array<fhn::RigObservation,21> points{};
 for(int i=0;i<21;++i){points[i].present=triangulate(cameras,pixels[i],points[i].point);out.stereo+=points[i].present;}
 out.fitted=stage.fit(reference,rig,side,points,cameras,pixels,stamp,source);
 out.corrected.control=out.fitted.refined.fit;out.corrected.fit=out.fitted.refined.fit;
 if(out.fitted.initialized)out.corrected=fhn::refine_index_middle_hypotheses(rig,out.fitted.initial.pose,out.fitted.refined.fit,cameras,pixels,out.fitted.initial.pose.angles);
 return out;
}
bool read_observations(std::istream& in,std::array<fhn::PixelObservation,21>& pixels){
 for(auto& p:pixels)for(int v=0;v<2;++v){int available;if(!(in>>available>>p.point[v].x>>p.point[v].y)||available<0||available>1)return false;p.present[v]=available;p.validate();}return true;
}
void write_fit(std::ostream& out,const fhn::HandRig& rig,std::int64_t stamp,int source,int side,const FitOutput& fit){
 const auto& fitted=fit.fitted;const auto& corrected=fit.corrected;
 out<<stamp<<' '<<source<<' '<<side<<' '<<fitted.initialized<<' '<<corrected.chosen<<' '<<fitted.refined.used_history<<' '<<fit.stereo;
 auto finite=[](double v){return std::isfinite(v)?v:-1.;};
 out<<' '<<finite(corrected.original_disagreement);for(double v:corrected.alternative_disagreement)out<<' '<<finite(v);for(bool v:corrected.eligible)out<<' '<<v;for(double v:corrected.alternative_cost)out<<' '<<finite(v);
 for(const auto* result:std::array<const fhn::PixelFitResult*,2>{&corrected.control,&corrected.fit}){
  if(!fitted.initialized){for(int i=0;i<92;++i)out<<" 0";continue;}
  const auto& p=result->pose;out<<' '<<result->cost<<' '<<result->observed_rmse_px<<' '<<p.wrist.q.w<<' '<<p.wrist.q.x<<' '<<p.wrist.q.y<<' '<<p.wrist.q.z<<' '<<p.wrist.t.x<<' '<<p.wrist.t.y<<' '<<p.wrist.t.z;
  for(double angle:p.angles)out<<' '<<angle;for(auto point:rig.landmarks(p.angles,p.wrist,side))out<<' '<<point.x<<' '<<point.y<<' '<<point.z;
 }out<<'\n';
}
int main(int argc,char**argv){try{
 if(argc!=7&&argc!=10)return 2;bool isolated=argc==10;std::ifstream generic(argv[1]),left(argv[2]),right(argv[3]),calibration(argv[4]),input(argv[5]);
 if(!generic||!left||!right||!calibration||!input||std::ifstream(argv[6]).good())return 3;
 std::ifstream baseline,decisions;std::ofstream baseline_output;
 if(isolated){baseline.open(argv[7]);decisions.open(argv[8]);if(!baseline||!decisions||std::ifstream(argv[9]).good())return 3;baseline_output.open(argv[9]);baseline_output<<std::setprecision(17);}
 auto reference=fhn::HandRig::load(generic);std::array<fhn::HandRig,2> rigs{fhn::HandRig::load(left),fhn::HandRig::load(right)};auto cameras=fhn::load_rig_cameras(calibration);
 std::ofstream out(argv[6]);out<<std::setprecision(17);std::map<int,Track> tracks;std::string tag,other;std::int64_t stamp,last=0,other_stamp;int count,total=0,other_count;
 while(input>>tag){if(tag!="set"||!(input>>stamp>>count)||stamp<=last||count<0||count>20)return 4;last=stamp;std::set<int> present;
  if(isolated&&(!(baseline>>other>>other_stamp>>other_count)||other!=tag||other_stamp!=stamp||other_count!=count))return 4;
  for(int n=0;n<count;++n){int source,side;if(!(input>>tag>>source>>side)||tag!="hand"||source<=0||side<0||side>1||!present.insert(source).second)return 4;
   std::array<fhn::PixelObservation,21> pixels{},normal{};if(!read_observations(input,pixels))return 4;int choice=0;
   if(isolated){int other_source,other_side;if(!(baseline>>other>>other_source>>other_side)||other!=tag||other_source!=source||other_side!=side||!read_observations(baseline,normal))return 4;
    if(!(decisions>>other_stamp>>other_source>>choice)||other_stamp!=stamp||other_source!=source||choice<0||choice>3)return 4;
    for(int i=0;i<12;++i){double value;if(!(decisions>>value)||!std::isfinite(value))return 4;}
    if(choice==0)for(int i=0;i<21;++i)for(int v=0;v<2;++v)if(pixels[i].present[v]!=normal[i].present[v]||(pixels[i].point[v]-normal[i].point[v]).len()!=0)return 4;
   }else normal=pixels;
   auto& track=tracks[source];if(track.side!=side){track=Track{};track.side=side;}
   // Copy the pre-observation normal state. Alternate fits never update normal history.
   auto temporary=track.stage;auto control=fit_observations(track.stage,reference,rigs[side],cameras,normal,stamp,source,side);auto selected=control;
   if(isolated&&choice)selected=fit_observations(temporary,reference,rigs[side],cameras,pixels,stamp,source,side);
   if(!control.fitted.initialized)tracks.erase(source);
   write_fit(out,rigs[side],stamp,source,side,selected);if(isolated)write_fit(baseline_output,rigs[side],stamp,source,side,control);++total;
  }
  for(auto it=tracks.begin();it!=tracks.end();)if(!present.count(it->first))it=tracks.erase(it);else ++it;
 }
 if(!input.eof()||!out.good()||!total)return 5;if(isolated&&(baseline>>other||decisions>>other||!baseline_output.good()))return 5;
 std::cout<<"hands="<<total<<'\n';return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
