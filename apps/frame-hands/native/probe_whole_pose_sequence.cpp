#include "rig_stage.h"
#include "whole_pose_fitter.h"
#include <fstream>
#include <iostream>
#include <iomanip>
#include <random>
int main(int argc,char**argv){try{
 if(argc!=6)return 2;std::ifstream l(argv[1]),r(argv[2]),g(argv[3]),c(argv[4]);if(std::ifstream(argv[5]).good())return 3;
 std::array<fhn::HandRig,2> rigs{fhn::HandRig::load(l),fhn::HandRig::load(r)};auto reference=fhn::HandRig::load(g);auto cameras=fhn::load_rig_cameras(c);
 std::ofstream out(argv[5]);out<<std::setprecision(17);std::mt19937 random(41872);std::normal_distribution<double> noise(0,1);int cases=0;
 for(int mode=0;mode<4;++mode){fhn::RigStage stage;fhn::WholePoseFitter history;
  std::array<bool,2> last_valid{};std::array<std::array<fhn::V,21>,2> old_truth{};std::array<std::array<std::array<fhn::V,21>,2>,2> old_output{};
  for(int frame=0;frame<120;++frame)for(int side=0;side<2;++side){
   auto stamp=1000000000LL+frame*33333333LL+(mode==3&&frame>=80?150000000LL:0);int source=10+side+(mode==3&&frame>=90?10:0);
   fhn::Rigid head{fhn::rig_exp({.1*std::sin(frame*.07),.13*std::sin(frame*.03),.04*std::cos(frame*.04)}),{5*std::sin(frame*.09),4*std::cos(frame*.1),3*std::sin(frame*.06)}};
   fhn::RigPose truth;truth.side=side;truth.wrist={fhn::rig_exp({.3*std::sin(frame*.05),.2*std::cos(frame*.04),frame<60?0.:1.2}),{side?70.:-70.,10*std::sin(frame*.08),-450+20*std::sin(frame*.04)}};
   auto blend=[](double x){return std::clamp(x,0.,1.);};
   for(int j=0;j<20;++j){double fraction=.2+.6*blend((frame-25)/2.);if(frame>=50&&j/4==1)fraction=.8-.6*blend((frame-50)/2.);if(frame>=85)fraction=.8-.6*blend((frame-85)/2.);
    truth.angles[j]=(1-fraction)*rigs[side].joints[j].lower+fraction*rigs[side].joints[j].upper;}
   auto truth_world=rigs[side].landmarks(truth.angles,truth.wrist,side);auto local=truth;local.wrist=fhn::rig_inverse(head).then(truth.wrist);local.wrist.q=local.wrist.q.unit();
   auto actual=rigs[side].landmarks(local.angles,local.wrist,side);auto measured=actual;
   if(mode==2&&(frame==30||frame==31||frame==90||frame==91)){auto wrong=local;wrong.wrist.q=(fhn::rig_exp({0,3.141592653589793,0})*wrong.wrist.q).unit();measured=rigs[side].landmarks(wrong.angles,wrong.wrist,side);}
   std::array<fhn::RigObservation,21> obs{};std::array<fhn::PixelObservation,21> pixels{};
   bool absent=mode==3&&frame>=45&&frame<=47;
   for(int i=0;i<20;++i){obs[i].point=measured[i];obs[i].present=true;
    if(mode==1)obs[i].point=obs[i].point+fhn::V{3*noise(random),3*noise(random),3*noise(random)};
    for(int v=0;v<2;++v){if(!cameras[v].project(actual[i],pixels[i].point[v]))throw std::runtime_error("Synthetic point behind camera");pixels[i].present[v]=!absent||v==0;
     if(mode==1){pixels[i].point[v].x+=2*noise(random);pixels[i].point[v].y+=2*noise(random);}}
   }
   auto baseline=stage.fit(reference,rigs[side],side,obs,cameras,pixels,stamp,source);
   if(!baseline.initialized){history.clear(side);last_valid[side]=false;out<<side<<' '<<mode<<' '<<frame<<" 0 0 0 0 0 0 0 0 0\n";++cases;continue;}
   bool head_valid=!(mode==3&&frame==75);
   auto selected=history.refine(rigs[side],baseline.initial.pose,baseline.refined.fit,cameras,obs,pixels,stamp,source,head_valid?&head:nullptr);
   if(selected.fit.cost>baseline.refined.fit.cost)throw std::runtime_error("Objective regression");
   std::array<double,2> error{},motion{};
   for(int variant=0;variant<2;++variant){auto p=variant?selected.fit.pose:baseline.refined.fit.pose;auto points=rigs[side].landmarks(p.angles,p.wrist,side);
    for(int i=0;i<21;++i){points[i]=head.apply(points[i]);auto e=points[i]-truth_world[i];error[variant]+=e.dot(e);
     if(last_valid[side]){auto d=(points[i]-old_output[side][variant][i])-(truth_world[i]-old_truth[side][i]);motion[variant]+=d.dot(d);}}
    error[variant]=std::sqrt(error[variant]/21);motion[variant]=std::sqrt(motion[variant]/21);old_output[side][variant]=points;}
   old_truth[side]=truth_world;last_valid[side]=true;
   out<<side<<' '<<mode<<' '<<frame<<" 1 "<<selected.eligible<<' '<<selected.used<<' '<<baseline.refined.fit.cost<<' '<<selected.fit.cost<<' '<<error[0]<<' '<<error[1]<<' '<<motion[0]<<' '<<motion[1]<<'\n';++cases;
  }
 }
 if(!out.good())return 5;std::cout<<"sequence_cases="<<cases<<'\n';return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
