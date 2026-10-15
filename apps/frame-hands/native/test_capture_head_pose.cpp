#include "capture_head_pose.h"
#include "native_head_pose.h"
#include <iostream>
#include <limits>
int main(){int n=0;auto check=[&](bool b){++n;if(!b)throw std::runtime_error("Capture head control failed");};try{
 float age=0;check(fhn::head_request_age(1,250000001,age));check(!fhn::head_request_age(1,250000002,age));check(!fhn::head_request_age(0,1,age));check(!fhn::head_request_age(2,1,age));
 double m[12]{1,0,0,.01,0,1,0,-.02,0,0,1,.03};fhn::Rigid head;check(fhn::head_from_matrix(m,head));check((head.t-fhn::V{10,-20,30}).len()<1e-12);
 m[0]=-1;check(!fhn::head_from_matrix(m,head));m[0]=1;m[1]=.1;check(!fhn::head_from_matrix(m,head));m[1]=0;m[3]=NAN;check(!fhn::head_from_matrix(m,head));
 fhn::CaptureHeadPose x;x.raw_ns=20;x.capture_ns=10;x.query_ns=15;x.returned_ns=17;x.generation=1;x.valid=true;check(x.usable(20));check(!x.usable(21));
 auto bad=x;bad.generation=0;check(!bad.usable(20));bad=x;bad.query_ns=9;check(!bad.usable(20));bad=x;bad.returned_ns=14;check(!bad.usable(20));bad=x;bad.returned_ns=250000011;check(!bad.usable(20));bad=x;bad.head.q.w=2;check(!bad.usable(20));
#if !defined(FRAME_HANDS_WITH_OPENVR) || defined(_WIN32)
 bool threw=false;try{fhn::NativeHeadPose unavailable("unused");}catch(const std::runtime_error&){threw=true;}check(threw);
#endif
 std::cout<<"controls="<<n<<" failures=0\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
