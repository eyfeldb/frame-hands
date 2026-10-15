#include "source_filter.h"
#include <fstream>
#include <iomanip>
#include <map>
#include <string>
int main(int argc,char** argv){
 if(argc!=3&&argc!=4)return 2;bool guard=argc==3;
 if(argc==4&&std::string(argv[3])!="--historical-unguarded")return 2;std::ifstream in(argv[1]);if(!in || std::ifstream(argv[2]).good())return 3;
 std::ofstream out(argv[2]);if(!out)return 4;out<<std::setprecision(17);
 std::map<unsigned,std::array<fhn::SourceFilter,2>> filters;
 int64_t t;unsigned id;
 while(in>>t>>id){std::array<fhs::V,21> p{};for(auto& v:p)if(!(in>>v.x>>v.y>>v.z))return 5;
  if(!id)return 6;out<<t<<' '<<id;
  for(int mode=0;mode<2;++mode){auto& f=filters[id][mode];if(!f.update(p,t,mode,guard))return 7;out<<' '<<f.guard_resets;for(auto v:f.value)out<<' '<<v.x<<' '<<v.y<<' '<<v.z;}out<<'\n';
 }
 return in.eof()&&out.good()?0:8;
}
