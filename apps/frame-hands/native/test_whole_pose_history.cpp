#include "whole_pose_fitter.h"
#include <fstream>
#include <iostream>
void require(bool b){if(!b)throw std::runtime_error("History control failed");}
int main(int argc,char**argv){try{
 if(argc!=4)return 2;std::ifstream l(argv[1]),r(argv[2]),c(argv[3]);std::array<fhn::HandRig,2> rigs{fhn::HandRig::load(l),fhn::HandRig::load(r)};auto cameras=fhn::load_rig_cameras(c);
 int checks=0;fhn::WholePoseFitter fitter;fhn::Rigid head;
 auto call=[&](std::int64_t stamp,int side,int source,const fhn::Rigid* h,bool continuous=true,int present=20,bool palm=true){
  fhn::RigPose pose;pose.side=side;pose.wrist.t={0,0,-450};for(int j=0;j<20;++j)pose.angles[j]=.5*(rigs[side].joints[j].lower+rigs[side].joints[j].upper);
  auto points=rigs[side].landmarks(pose.angles,pose.wrist,side);std::array<fhn::RigObservation,21> obs{};std::array<fhn::PixelObservation,21> pixels{};
  for(int i=0;i<20;++i){obs[i].point=points[i];obs[i].present=true;for(int v=0;v<2;++v){require(cameras[v].project(points[i],pixels[i].point[v]));pixels[i].present[v]=i<present;}}
  if(!palm)obs[11].present=false;
  fhn::PixelFitResult baseline;baseline.sufficient=true;baseline.pose=pose;
  return fitter.refine(rigs[side],pose,baseline,cameras,obs,pixels,stamp,source,h,continuous);
 };
 auto check=[&](bool b){require(b);++checks;};std::int64_t t=1000000000;
 check(!call(t,0,1,&head).eligible);t+=33333333;check(call(t,0,1,&head).eligible);
 bool threw=false;try{call(t,0,1,&head);}catch(const std::invalid_argument&){threw=true;}check(threw);
 t+=33333333;check(!call(t,0,1,&head,false).eligible);t+=33333333;check(call(t,0,1,&head).eligible);
 t+=33333333;check(!call(t,0,1,nullptr).eligible);t+=33333333;check(!call(t,0,1,&head).eligible);t+=33333333;check(call(t,0,1,&head).eligible);
 t+=33333333;check(!call(t,0,2,&head).eligible);t+=100000000;check(call(t,0,2,&head).eligible);t+=100000001;check(!call(t,0,2,&head).eligible);
 t+=33333333;check(!call(t,0,2,&head,true,5).eligible);t+=33333333;check(!call(t,0,2,&head).eligible);
 t+=33333333;check(!call(t,0,2,&head,true,20,false).eligible);t+=33333333;check(!call(t,0,2,&head).eligible);
 fitter.clear(0);threw=false;try{call(t,0,2,&head);}catch(const std::invalid_argument&){threw=true;}check(threw);
 t+=33333333;check(!call(t,0,2,&head).eligible);
 t+=33333333;check(!call(t,1,2,&head).eligible);t+=33333333;check(!call(t,0,2,&head).eligible);
 auto bad=head;bad.q.w=2;threw=false;try{call(t+1,0,2,&bad);}catch(const std::invalid_argument&){threw=true;}check(threw);
 t+=33333333;check(call(t,0,2,&head).eligible);fitter.clear();t+=33333333;check(!call(t,0,2,&head).eligible);
 std::cout<<"controls="<<checks<<" failures=0\n";return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
