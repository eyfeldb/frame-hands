#define main saved_probe_main
#include "probe_hand_scale_rig.cpp"
#undef main
int main(int argc,char**argv){try{
 if(argc!=4)return 2;std::ifstream f(argv[1]),c(argv[2]);auto rig=fhn::HandRig::load(f);auto cameras=fhn::load_rig_cameras(c);int checks=0;double maximum_error=0;
 auto check=[&](bool ok){if(!ok)throw std::runtime_error("Triangulation control failed at check "+std::to_string(checks+1));++checks;};
 for(int pose=0;pose<3;++pose){fhn::RigPose p;p.side=rig.subject_side;p.wrist.t={double(pose*20),-70,-450-double(pose*50)};
  for(int j=0;j<20;++j)p.angles[j]=std::clamp(pose*.15,rig.joints[j].lower,rig.joints[j].upper);
  for(auto truth:rig.landmarks(p.angles,p.wrist,p.side)){fhn::PixelObservation o;for(int v=0;v<2;++v){o.present[v]=true;check(cameras[v].project(truth,o.point[v]));}
   fhn::V recovered;check(triangulate(cameras,o,recovered));maximum_error=std::max(maximum_error,(recovered-truth).len());check((recovered-truth).len()<.001);
   o.present[1]=false;check(!triangulate(cameras,o,recovered));o.present[1]=true;
   auto same=cameras;same[1]=same[0];o.point[1]=o.point[0];check(!triangulate(same,o,recovered));
  }
 }
 std::ofstream out(argv[3]);out<<"controls="<<checks<<" failures=0 maximum_error_mm="<<std::setprecision(17)<<maximum_error<<"\n";std::cout<<"controls="<<checks<<" failures=0 maximum_error_mm="<<std::setprecision(17)<<maximum_error<<"\n";return out.good()?0:3;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
