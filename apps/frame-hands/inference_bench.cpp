// Synthetic crops measure inference cost, never hand accuracy or sensor latency.
#include "nets.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

int main(int argc,char** argv) {
 if(argc!=3){std::fprintf(stderr,"usage: inference-bench MODELS fp16|int8\n");return 2;}
 bool int8=std::string(argv[2])=="int8";
 if(!int8 && std::string(argv[2])!="fp16")return 2;
 Nets nets;std::string err;
 if(!nets.load(argv[1],int8,err)){std::fprintf(stderr,"%s\n",err.c_str());return 1;}
 Contrast p,h;Contrast::parse_pair("clahe:2/none",p,h);nets.set_contrast(p,h);
 std::vector<uint8_t> pixels(640*480);
 for(int y=0;y<480;++y)for(int x=0;x<640;++x)pixels[y*640+x]=uint8_t(40+(x*7+y*3)%160);
 Image img{pixels.data(),640,480,640};Roi roi{{320,240},300,0};
 std::vector<double> palm,hand;bool finite=true;
 for(int n=0;n<70;++n){
   auto t=std::chrono::steady_clock::now();auto palms=nets.palms(img,{320,240},480,0);
   auto u=std::chrono::steady_clock::now();auto lm=nets.landmarks(img,roi);
   auto v=std::chrono::steady_clock::now();
   finite &= std::isfinite(lm.presence)&&std::isfinite(lm.right);
   for(int j=0;j<21;++j){for(int k=0;k<2;++k)finite &= std::isfinite(lm.pts[j][k]);for(int k=0;k<3;++k)finite &= std::isfinite(lm.world[j][k]);}
   if(n>=10){palm.push_back(std::chrono::duration<double,std::milli>(u-t).count());hand.push_back(std::chrono::duration<double,std::milli>(v-u).count());}
 }
 std::sort(palm.begin(),palm.end());std::sort(hand.begin(),hand.end());
 std::printf("{\"input\":\"synthetic_crop_not_accuracy_test\",\"mode\":\"%s\",\"warmup\":10,\"samples\":60,\"palm_p50_ms\":%.3f,\"palm_p95_ms\":%.3f,\"landmarks_p50_ms\":%.3f,\"landmarks_p95_ms\":%.3f,\"finite_outputs\":%s}\n",argv[2],palm[29],palm[56],hand[29],hand[56],finite?"true":"false");
 return finite?0:1;
}
