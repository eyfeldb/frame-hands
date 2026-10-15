#include "nets.h"
#include "record.h"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
struct Request {long long stamp;std::string camera;Roi roi;};
int main(int argc,char**argv){try{
 if(argc!=5)return 2;std::ifstream input(argv[3]);if(!input||std::ifstream(argv[4]).good())return 3;std::vector<Request> req;Request r;
 while(input>>r.stamp>>r.camera>>r.roi.center[0]>>r.roi.center[1]>>r.roi.size>>r.roi.rotation)req.push_back(r);
 if(!input.eof()||req.empty()||req.size()>200)return 4;Nets nets;std::string error;if(!nets.load(argv[2],false,error))throw std::runtime_error(error);Contrast hand;hand.mode=Contrast::None;nets.set_contrast({},hand);
 SetReader reader;if(!reader.open(argv[1],error))throw std::runtime_error(error);std::ofstream out(argv[4]);out<<std::setprecision(17);std::vector<fh_set_cam_t> cams;std::vector<std::vector<uint8_t>> pixels;int cases=0;
 while(reader.next(cams,pixels)){uint64_t stamp=UINT64_MAX;for(const auto& c:cams)stamp=std::min(stamp,c.capture_ns);for(int j=0;j<int(req.size());++j)for(int k=0;k<int(cams.size());++k)if(req[j].stamp==static_cast<long long>(stamp)&&req[j].camera==cams[k].name){
  const auto& camera=cams[k];Image image{pixels[k].data(),int(camera.width),int(camera.height),int(camera.width)};
  for(double scale:{1.,1.3}){Roi roi=req[j].roi;roi.size*=scale;auto result=nets.landmarks(image,roi);out<<j<<' '<<scale<<' '<<result.presence<<' '<<result.right;for(auto point:result.pts)out<<' '<<point[0]<<' '<<point[1];out<<'\n';}++cases;
 }}
 if(cases!=int(req.size())||!out.good())throw std::runtime_error("Missing requested input");std::cout<<"cases="<<cases<<"\n";return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
