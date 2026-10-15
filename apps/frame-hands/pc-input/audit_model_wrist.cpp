#include "retarget_chain.h"
#include "model_protocol.h"
#include <fstream>
#include <cstdio>
int main(int argc,char** argv){if(argc!=2)return 2;std::ifstream file(argv[1],std::ios::binary);fhs::ModelPacket p;fhs::Q first[2];bool initialized=false;double max_degrees[2]{};int count=0;
while(file.read(reinterpret_cast<char*>(&p),sizeof p)){for(int s=0;s<2;++s){std::array<fhs::V,21> points{};for(int i=0;i<21;++i)points[i]={p.hands[s].points[i][0],p.hands[s].points[i][1],p.hands[s].points[i][2]};auto q=fhs::retarget_chain(points,s).rotation;if(!initialized)first[s]=q;else{auto delta=q*first[s].inv();max_degrees[s]=std::max(max_degrees[s],2*std::acos(std::clamp(std::abs(delta.w),0.,1.))*180/3.141592653589793);}}initialized=true;++count;}std::printf("fixed_model_wrist_poses=%d reconstructed_wrist_rotation_max_deg=%.6f,%.6f\n",count,max_degrees[0],max_degrees[1]);}
