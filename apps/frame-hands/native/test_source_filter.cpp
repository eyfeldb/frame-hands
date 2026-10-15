#ifdef NDEBUG
#undef NDEBUG
#endif
#include "source_filter.h"
#include "../pc-input/skeleton.h"
#include "../pc-input/stability.h"
#include "source_filter_reference.h"
#include <cassert>
#include <iostream>
#include <iomanip>
std::array<fhs::V,21> pose(){
 std::array<fhs::Transform,31> local{};for(int i=0;i<31;++i)local[i]={fhs::rest[0][i].p,fhs::rest[0][i].q};
 auto b=fhs::forward(local);std::array<fhs::V,21> p{};for(int i=0;i<21;++i)p[i]=b[fhs::landmark_bone[i]].p+fhs::V{0,0,-.5};return p;
}
int main(){
 auto p=pose();assert(fhn::SourceFilter::plausible(p));
 ReferenceHand ref;fhn::SourceFilter control;double parity=0;
 for(int n=0;n<600;++n){auto raw=p;double a=.4*std::sin(n*.035);fhs::Q q{std::cos(a/2),std::sin(a/2),0,0};
  for(int j=0;j<21;++j){raw[j]=p[0]+q.rotate(raw[j]-p[0])+fhs::V{.1*std::sin(n*.02),0,0};if(j==8)raw[j].x+=.004*std::sin(n*.7);ref.pts[j]={raw[j].x,raw[j].y,raw[j].z};}
  int64_t t=1000000000LL+n*33333333LL+(n>=300?500000000LL:0);++ref.frames;reference_smooth(ref,t);assert(control.update(raw,t,false));
  for(int j=0;j<21;++j)parity=std::max(parity,(control.value[j]-fhs::V{ref.smooth[j][0],ref.smooth[j][1],ref.smooth[j][2]}).len());
 }
 assert(parity<1e-12);
 double error[2]{},noise[2]{},final_error[2]{};unsigned guards[2]{};
 for(int mode=0;mode<2;++mode){fhn::SourceFilter s;fhs::Stability display;
  for(int n=0;n<360;++n){auto truth=p;double a=.8*std::sin(n*2*3.141592653589793/30.);fhs::Q q{std::cos(a/2),std::sin(a/2),0,0};
   for(int j=6;j<=8;++j)truth[j]=p[5]+q.rotate(p[j]-p[5]);assert(s.update(truth,1000000000LL+n*33333333LL,mode));
   if(n>30)error[mode]+=std::pow((s.value[8]-truth[8]).len(),2);
   bool ok=display.update(s.value,1,1000+n*33,1000+n*33);assert(ok);
   if(n>30)final_error[mode]+=std::pow((display.value[8]-truth[8]).len(),2);
   for(int j:{0,5,9,13,17})assert((s.value[j]-p[j]).len()<1e-10);
  }
  guards[mode]=s.guard_resets;s.reset();
  for(int n=0;n<360;++n){auto raw=p;raw[8].x+=(n%2?.001:-.001);assert(s.update(raw,20000000000LL+n*33333333LL,mode));if(n>30)noise[mode]+=std::pow((s.value[8]-p[8]).len(),2);}
  auto stamp=s.stamp;assert(!s.update(p,stamp,mode));assert(!s.update(p,stamp-1,mode));auto bad=p;bad[8].x=std::nan("");assert(!s.update(bad,stamp+33333333,mode));assert(s.stamp==stamp);
  assert(s.update(p,stamp+500000000,mode));assert((s.value[8]-p[8]).len()<1e-12);
 }
 assert(error[1]<error[0]);assert(final_error[1]<final_error[0]);
 std::cout<<std::setprecision(12)<<"parity_max_m="<<parity<<" source_bend_rmse_mm="<<1000*std::sqrt(error[0]/329)<<","<<1000*std::sqrt(error[1]/329)<<" downstream_bend_rmse_mm="<<1000*std::sqrt(final_error[0]/329)<<","<<1000*std::sqrt(final_error[1]/329)<<" noise_rms_mm="<<1000*std::sqrt(noise[0]/329)<<","<<1000*std::sqrt(noise[1]/329)<<" guard_resets="<<guards[0]<<","<<guards[1]<<"\n";
}
