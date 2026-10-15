// Export byte-identical native sampler inputs and original landmark outputs.
#include "nets.cpp"
#include "record.h"
#include <fstream>
#include <iomanip>
#include <iostream>
struct Request{long long stamp;std::string camera;Roi roi;};
int main(int argc,char**argv){try{
 if(argc!=5)return 2;std::ifstream in(argv[3]);std::vector<Request> requests;Request r;
 while(in>>r.stamp>>r.camera>>r.roi.center[0]>>r.roi.center[1]>>r.roi.size>>r.roi.rotation)requests.push_back(r);
 if(!in.eof()||requests.empty()||requests.size()>10000)return 3;
 Nets nets;std::string error;if(!nets.load(argv[2],false,error))throw std::runtime_error(error);
 Contrast hand;hand.mode=Contrast::None;nets.set_contrast({},hand);
 SetReader reader;if(!reader.open(argv[1],error))throw std::runtime_error(error);
 std::ofstream out(std::string(argv[4])+"/native.txt");out<<std::setprecision(17);std::vector<fh_set_cam_t> cams;std::vector<std::vector<uint8_t>> pixels;int cases=0;
 while(reader.next(cams,pixels)){uint64_t stamp=UINT64_MAX;for(const auto& c:cams)stamp=std::min(stamp,c.capture_ns);
  for(int j=0;j<int(requests.size());++j)for(int k=0;k<int(cams.size());++k)if(requests[j].stamp==static_cast<long long>(stamp)&&requests[j].camera==cams[k].name){
   auto& c=cams[k];Image im{pixels[k].data(),int(c.width),int(c.height),int(c.width)};const auto& roi=requests[j].roi;
   float tm[6];crop_matrix(roi.center,roi.size,roi.rotation,224,tm);std::vector<uint8_t> patch(224*224);
   ncnn::warpaffine_bilinear_c1(im.data,im.width,im.height,im.stride,patch.data(),224,224,224,tm,0,0);
   auto path=std::string(argv[4])+"/"+std::to_string(j)+".gray";if(std::ifstream(path).good())throw std::runtime_error("Existing crop output");
   std::ofstream file(path,std::ios::binary);file.write(reinterpret_cast<const char*>(patch.data()),patch.size());if(!file.good())throw std::runtime_error("Crop write failed");
   auto lm=nets.landmarks(im,roi);out<<j<<' '<<lm.presence<<' '<<lm.right;for(float x:tm)out<<' '<<x;for(auto p:lm.pts)out<<' '<<p[0]<<' '<<p[1];out<<'\n';++cases;
  }
 }
 if(cases!=int(requests.size())||!out.good())throw std::runtime_error("Incomplete crops");std::cout<<"cases="<<cases<<'\n';return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
