#ifdef NDEBUG
#undef NDEBUG
#endif
#include "stable_lengths.h"
#include "../pc-input/skeleton.h"
#include <cassert>
#include <iostream>
std::array<fhs::V,21> reference(){
 std::array<fhs::Transform,31> local{};
 for(int i=0;i<31;++i)local[i]={fhs::rest[0][i].p,fhs::rest[0][i].q};
 auto b=fhs::forward(local);std::array<fhs::V,21> p{};
 for(int i=0;i<21;++i)p[i]=b[fhs::landmark_bone[i]].p;return p;
}
int main(){
 auto p=reference();assert(fhs::Stability::geometry(p));std::array<fhs::V,21> out{};
 fhn::StableLengths s;
 for(int i=0;i<20;++i){assert(s.apply(p,p,1,1000+i*33,false,out));assert(s.count()==0);assert(fhs::Stability::distance(p,out)<1e-12);}
 for(int i=0;i<15;++i)assert(s.apply(p,p,1,1660+i*33,true,out));assert(s.count()==15);
 auto stretched=p;
 for(int i=1;i<21;++i)if(s.finger(i)){int par=fhs::Stability::parent[i];stretched[i]=stretched[par]+(p[i]-p[par])*1.3;}
 assert(s.apply(stretched,stretched,1,2155,false,out));assert(fhs::Stability::distance(p,out)<1e-12);
 // Actual rigid motion and finger rotation must remain free after calibration.
 auto moving=p;fhs::Q q{std::cos(.2),0,std::sin(.2),0};for(auto& v:moving)v=q.rotate(v)+fhs::V{.1,.2,-.3};
 assert(s.apply(moving,moving,1,2188,false,out));assert(fhs::Stability::distance(moving,out)<1e-12);
 auto invalid=p;invalid[8]=invalid[7];assert(!s.apply(invalid,invalid,1,2221,true,out));assert(s.stamp==2188);
 assert(!s.apply(p,p,1,2188,true,out));assert(s.stamp==2188);
 assert(s.apply(stretched,stretched,2,2221,false,out));assert(s.count()==0);assert(fhs::Stability::distance(stretched,out)<1e-12);
 for(int i=0;i<15;++i)s.apply(p,p,2,2254+i*33,true,out);assert(s.count()==15);
 assert(s.apply(stretched,stretched,2,2900,false,out));assert(s.count()==0);
 // Median tolerates isolated calibration outliers; this does not prove real accuracy.
 s.reset();for(int i=0;i<15;++i)s.apply(i==7?stretched:p,i==7?stretched:p,1,4000+i*33,true,out);
 assert(s.count()==15);assert(s.apply(stretched,stretched,1,4495,false,out));assert(fhs::Stability::distance(p,out)<1e-12);
 std::cout<<"passed: calibration admission, rigid motion, length noise, invalid/duplicate time, ID/gap reset, robust median\n";
}
