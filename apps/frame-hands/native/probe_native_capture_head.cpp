#include "native_head_pose.h"
#include <charconv>
#include <iostream>
int main(int argc,char**argv){try{
 if(argc!=4)return 2;std::uint64_t raw=0,mono=0;
 auto parse=[](const char* s,std::uint64_t& value){auto end=s+std::char_traits<char>::length(s);auto p=std::from_chars(s,end,value);return p.ec==std::errc{}&&p.ptr==end&&value>0;};
 if(!parse(argv[2],raw)||!parse(argv[3],mono)||raw>INT64_MAX)return 2;
 fhn::NativeHeadPose provider(argv[1]);auto p=provider.query(std::int64_t(raw),mono);
 std::cout.precision(17);std::cout<<p.valid<<' '<<p.raw_ns<<' '<<p.capture_ns<<' '<<p.query_ns<<' '<<p.returned_ns<<' '<<p.generation<<' '<<p.head.q.w<<' '<<p.head.q.x<<' '<<p.head.q.y<<' '<<p.head.q.z<<' '<<p.head.t.x<<' '<<p.head.t.y<<' '<<p.head.t.z<<'\n';return p.valid?0:4;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
