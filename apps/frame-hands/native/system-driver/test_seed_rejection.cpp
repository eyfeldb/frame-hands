#include "native/whole_pose_fitter.h"
#include "native/causal_rig_fit.h"
#include <fstream>
#include <iostream>
using namespace fhn;
void require(bool b){if(!b)throw std::runtime_error("seed rejection regression failed");}
int main(int argc,char**argv){try{
 if(argc!=3)return 2;
 for(int side=0;side<2;++side){std::ifstream file(argv[side+1]);auto rig=HandRig::load(file);
 RigPose previous,fresh;previous.side=fresh.side=side;
 previous.wrist.t=fresh.wrist.t={0,0,450};
 for(int i=0;i<20;++i){previous.angles[i]=rig.joints[i].lower;fresh.angles[i]=rig.joints[i].upper;}
 auto points=rig.landmarks(fresh.angles,fresh.wrist,side);
 auto oldpoints=rig.landmarks(previous.angles,previous.wrist,side);
 std::array<RigObservation,21> obs{},oldobs{};std::array<PixelObservation,21> pixels{};
 for(int i=0;i<21;++i){obs[i].present=oldobs[i].present=true;obs[i].point=points[i];oldobs[i].point=oldpoints[i];pixels[i].present={true,true};}
 Rigid head;RigPose alternate;require(whole_pose_seed(rig,previous,head,obs,alternate));
 auto seedpoints=rig.landmarks(alternate.angles,alternate.wrist,side);
 std::array<RigCamera,2> safe{};for(auto& c:safe){c.fx=c.fy=100;c.translation.z=1000;}
 std::array<RigCamera,2> edge{};bool found=false;
 for(int axis=0;axis<3&&!found;++axis)for(int sign:{-1,1}){
 V z{0,0,0};if(axis==0)z.x=sign;if(axis==1)z.y=sign;if(axis==2)z.z=sign;
 double a=1e9,b=1e9;for(int i=0;i<21;++i){a=std::min(a,z.dot(seedpoints[i]));b=std::min(b,z.dot(points[i]));}
 if(a>=b-1e-3)continue;
 V helper=axis==0?V{0,1,0}:V{1,0,0};V x=helper.cross(z),y=z.cross(x);
 for(auto& c:edge){c.rotation={x,y,z};c.translation.z=-(a+b)*.5;c.fx=c.fy=100;c.validate();}
 found=true;break;}
 require(found);
 for(int i=0;i<21;++i)for(int v=0;v<2;++v)require(edge[v].project(points[i],pixels[i].point[v]));
 bool throws=false;try{fit_rig_pixels(rig,alternate,edge,pixels,{},0,&fresh.angles);}catch(const std::invalid_argument& e){throws=std::string(e.what())=="Seed behind camera";}require(throws);
 CausalRigFitter causal;
 auto rejected=causal.fit(rig,alternate,edge,pixels,1000000000,7,{},true);
 require(!rejected.fit.sufficient);
 auto recovered=causal.fit(rig,fresh,edge,pixels,1033333333,7,{},true);
 require(recovered.fit.sufficient&&!recovered.history_eligible);
 // Valid warm angles can rescue an invalid fresh seed; do not refit that invalid control.
 RigPose bad=fresh;bad.angles=alternate.angles;
 auto warm=causal.fit(rig,bad,edge,pixels,1066666666,7,{},true);
 require(warm.fit.sufficient&&warm.used_history&&!std::isfinite(warm.fresh_seed_cost));
 WholePoseFitter fitter;PixelFitResult baseline;baseline.pose=previous;baseline.sufficient=true;
 fitter.refine(rig,previous,baseline,safe,oldobs,pixels,1000000000,7,&head);
 baseline.pose=fresh;auto result=fitter.refine(rig,fresh,baseline,edge,obs,pixels,1033333333,7,&head);
 require(!result.eligible&&!result.used&&result.fit.pose.angles==fresh.angles);
 auto resumed=fitter.refine(rig,fresh,baseline,edge,obs,pixels,1066666666,7,&head);
 require(resumed.eligible&&!resumed.used);
 }
 std::cout<<"Both sides: unsafe history rejected, baseline preserved, next frame recovers\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
