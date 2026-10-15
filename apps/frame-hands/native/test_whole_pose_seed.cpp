#include "whole_pose_seed.h"
#include "fit_rig_pixels.h"
#include <fstream>
#include <iostream>
void require(bool b){if(!b)throw std::runtime_error("Seed control failed");}
int main(int argc,char**argv){try{
 if(argc!=4)return 2;int cases=0;std::ifstream calibration(argv[3]);auto cameras=fhn::load_rig_cameras(calibration);
 for(int side=0;side<2;++side){std::ifstream input(argv[1+side]);auto rig=fhn::HandRig::load(input);
  fhn::RigPose previous;previous.side=side;previous.wrist={fhn::rig_exp({.2,-.4,.1}),{50,100,-450}};
  for(int j=0;j<20;++j)previous.angles[j]=.4*rig.joints[j].lower+.6*rig.joints[j].upper;
  for(auto head: {fhn::Rigid{},fhn::Rigid{fhn::rig_exp({-.4,.3,1.2}),{120,-30,40}}}){
   auto target=previous;target.wrist.t=target.wrist.t+fhn::V{15,-8,22};target.wrist=fhn::rig_inverse(head).then(target.wrist);
   auto cloud=rig.landmarks(target.angles,target.wrist,side);std::array<fhn::RigObservation,21> obs{};
   for(int i:{5,8,11,14,17}){obs[i].point=cloud[i];obs[i].present=true;}
   fhn::RigPose seeded;require(fhn::whole_pose_seed(rig,previous,head,obs,seeded));
   auto actual=rig.landmarks(seeded.angles,seeded.wrist,side);
   for(int i=0;i<21;++i)require((actual[i]-cloud[i]).len()<1e-8);
   require(seeded.angles==previous.angles);++cases;
   obs[11].present=false;require(!fhn::whole_pose_seed(rig,previous,head,obs,seeded));++cases;
  }
  auto invalid=fhn::Rigid{};invalid.q.w=2;bool threw=false;
  try{std::array<fhn::RigObservation,21> obs{};fhn::RigPose seed;fhn::whole_pose_seed(rig,previous,invalid,obs,seed);}catch(const std::invalid_argument&){threw=true;}
  require(threw);++cases;
  for(int gesture=0;gesture<4;++gesture){
   auto truth=previous;truth.wrist.q=(fhn::rig_exp({0,0,1.5707963267948966})*truth.wrist.q).unit();
   for(int j=0;j<20;++j){double fraction=gesture==0?.2:.8;if(gesture==2&&j/4==1)fraction=.2;
    truth.angles[j]=(1-fraction)*rig.joints[j].lower+fraction*rig.joints[j].upper;}
   auto cloud=rig.landmarks(truth.angles,truth.wrist,side);std::array<fhn::PixelObservation,21> pixels{};std::array<fhn::RigObservation,21> observations{};
   for(int i=0;i<20;++i){observations[i].point=cloud[i];observations[i].present=true;
    for(int v=0;v<2;++v){require(cameras[v].project(cloud[i],pixels[i].point[v]));pixels[i].present[v]=true;}}
   // Exact fresh evidence must win during deliberate rapid changes. Conversely,
   // a correct full-pose seed can rescue a deliberately flipped cold start.
   auto fresh=truth;if(gesture==3)fresh.wrist.q=(fhn::rig_exp({0,3.141592653589793,0})*truth.wrist.q).unit();
   auto recent=gesture==3?truth:previous;fhn::RigPose warm;
   require(fhn::whole_pose_seed(rig,recent,{},observations,warm));
   auto baseline=fhn::fit_rig_pixels(rig,fresh,cameras,pixels);
   auto alternate=fhn::fit_rig_pixels(rig,warm,cameras,pixels,{},0,&fresh.angles);
   auto selected=alternate.cost<baseline.cost?alternate:baseline;
   auto result=rig.landmarks(selected.pose.angles,selected.pose.wrist,side);
   for(int i=0;i<21;++i)require((result[i]-cloud[i]).len()<1e-5);
   if(gesture<3)require(baseline.cost<=alternate.cost);
   else require(alternate.cost<baseline.cost);
   ++cases;
  }
 }
 std::cout<<"controls="<<cases<<" failures=0\n";return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
