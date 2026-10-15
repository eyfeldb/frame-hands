#include "stereo_model_choice.h"
#include <fstream>
#include <iostream>
int main(int argc,char**argv){try{
 if(argc!=3)return 2;std::ifstream r(argv[1]),c(argv[2]);auto rig=fhn::HandRig::load(r);auto cameras=fhn::load_rig_cameras(c);int checks=0;
 auto check=[&](bool value){if(!value)throw std::runtime_error("Model-choice control "+std::to_string(checks+1));++checks;};
 fhn::RigPose pose;pose.side=rig.subject_side;pose.wrist.t={0,-70,-450};auto points=rig.landmarks(pose.angles,pose.wrist,pose.side);
 std::array<fhn::PixelObservation,21> good{};for(int i=0;i<20;++i)for(int v=0;v<2;++v){good[i].present[v]=true;check(cameras[v].project(points[i],good[i].point[v]));}
 std::array<double,2> presence{1,1};auto clean=fhn::choose_stereo_model_views(cameras,good,good,presence,presence);check(clean.choice==0);check(clean.maximum_chain[0]<1e-3);
 for(int view=0;view<2;++view){auto bad=good;for(int i:{8,9,10,1})bad[i].point[view].y+=100;
  auto selected=fhn::choose_stereo_model_views(cameras,bad,good,presence,presence);check(selected.maximum_chain[0]>3);check(selected.choice==(view==0?2:1));
  for(int i=0;i<21;++i)for(int v=0;v<2;++v)check((selected.pixels[i].point[v]-good[i].point[v]).len()==0);
  auto low=fhn::choose_stereo_model_views(cameras,bad,good,presence,{.1,.1});check(low.choice==0);
  auto both_bad=fhn::choose_stereo_model_views(cameras,bad,bad,presence,presence);check(both_bad.choice==0);
 }
 auto mono=good;for(auto& p:mono)p.present[1]=false;auto missing=fhn::choose_stereo_model_views(cameras,mono,mono,presence,presence);check(missing.choice==0);for(bool e:missing.eligible)check(!e);
 auto below=fhn::choose_stereo_model_views(cameras,good,good,{.1,.1},presence);check(below.choice==0);
 bool rejected=false;try{fhn::choose_stereo_model_views(cameras,good,good,{2,1},presence);}catch(const std::invalid_argument&){rejected=true;}check(rejected);
 std::cout<<"controls="<<checks<<" failures=0\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
