#include "stereo_disagreement.h"
#include <fstream>
#include <iostream>
int main(int argc,char**argv){try{
 if(argc!=2)return 2;std::ifstream input(argv[1]);auto cams=fhn::load_rig_cameras(input);int checks=0;auto check=[&](bool ok){if(!ok)throw std::runtime_error("Stereo disagreement control failed");++checks;};
 for(int z:{300,500,800})for(int x:{-120,0,120})for(int y:{-180,-50,50}){
  fhn::V point{double(x),double(y),-double(z)};fhn::PixelObservation o;
  for(int v=0;v<2;++v){o.present[v]=true;check(cams[v].project(point,o.point[v]));fhn::StereoRay ray;check(fhn::rig_pixel_ray(cams[v],o.point[v],ray));check((point-ray.origin).cross(ray.direction).len()<.001);}
  auto d=fhn::stereo_disagreement(cams,o);check(d.valid&&d.rms_px<.001);check(fhn::stereo_uncertainty_factor(d,o)==1);
 }
 fhn::PixelObservation o;for(int v=0;v<2;++v){o.present[v]=true;check(cams[v].project({0,-100,-450},o.point[v]));}o.point[1].y+=60;auto d=fhn::stereo_disagreement(cams,o);check(d.valid&&d.rms_px>15);double factor=fhn::stereo_uncertainty_factor(d,o);check(factor>1&&factor<=4);
 o.present[1]=false;check(!fhn::stereo_disagreement(cams,o).valid);check(fhn::stereo_uncertainty_factor({},o)==1);
 o.present[1]=true;check(!fhn::stereo_disagreement({cams[0],cams[0]},o).valid);
 o.sigma[0]=0;bool threw=false;try{fhn::stereo_disagreement(cams,o);}catch(const std::invalid_argument&){threw=true;}check(threw);
 std::cout<<"controls="<<checks<<" failures=0\n";return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
