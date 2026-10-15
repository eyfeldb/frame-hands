// Build candidate observation input only; does not alter any runtime tracker.
#include "stereo_model_choice.h"
#include <fstream>
#include <iostream>
#include <iomanip>
bool read_pixels(std::istream& in,std::array<fhn::PixelObservation,21>& points){
 for(auto& p:points)for(int v=0;v<2;++v){int available;if(!(in>>available>>p.point[v].x>>p.point[v].y)||available<0||available>1)return false;p.present[v]=available;}return true;
}
int main(int argc,char**argv){try{
 if(argc!=7)return 2;std::ifstream calibration(argv[1]),full(argv[2]),lite(argv[3]),presence(argv[4]);
 if(!calibration||!full||!lite||!presence||std::ifstream(argv[5]).good()||std::ifstream(argv[6]).good())return 3;
 auto cams=fhn::load_rig_cameras(calibration);std::ofstream output(argv[5]),decisions(argv[6]);output<<std::setprecision(17);decisions<<std::setprecision(17);
 std::string tag,other;std::int64_t stamp,other_stamp,last=0;int count,other_count,hands=0;
 while(full>>tag){if(tag!="set"||!(full>>stamp>>count)||stamp<=last||count<0||count>20||!(lite>>other>>other_stamp>>other_count)||other!=tag||other_stamp!=stamp||other_count!=count)return 4;last=stamp;
  output<<"set "<<stamp<<' '<<count<<'\n';
  for(int h=0;h<count;++h){int source,side,other_source,other_side;if(!(full>>tag>>source>>side)||tag!="hand"||!(lite>>other>>other_source>>other_side)||other!=tag||other_source!=source||other_side!=side)return 4;
   std::array<fhn::PixelObservation,21> f{},l{};if(!read_pixels(full,f)||!read_pixels(lite,l))return 4;
   std::array<double,2> fp{},lp{};if(!(presence>>other_stamp>>other_source)||other_stamp!=stamp||other_source!=source)return 4;
   for(auto& p:fp)if(!(presence>>p))return 4;for(auto& p:lp)if(!(presence>>p))return 4;
   auto selected=fhn::choose_stereo_model_views(cams,f,l,fp,lp);
   output<<"hand "<<source<<' '<<side;for(const auto& p:selected.pixels)for(int v=0;v<2;++v)output<<' '<<p.present[v]<<' '<<p.point[v].x<<' '<<p.point[v].y;output<<'\n';
   decisions<<stamp<<' '<<source<<' '<<selected.choice;for(int i=0;i<4;++i)decisions<<' '<<(std::isfinite(selected.score[i])?selected.score[i]:-1)<<' '<<(std::isfinite(selected.maximum_chain[i])?selected.maximum_chain[i]:-1)<<' '<<selected.eligible[i];decisions<<'\n';++hands;
  }
 }
 if(!full.eof()||lite>>other||presence>>other||!output.good()||!decisions.good()||!hands)return 5;std::cout<<"hands="<<hands<<'\n';return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
