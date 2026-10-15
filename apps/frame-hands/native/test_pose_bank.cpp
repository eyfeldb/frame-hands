#include "pose_bank.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <iomanip>
int main(int argc,char**argv){try{
 if(argc!=4)return 2;std::ifstream f(argv[1]),c(argv[2]),b(argv[3]);auto rig=fhn::HandRig::load(f);auto cams=fhn::load_rig_cameras(c);auto bank=fhn::PoseBank::load(b);bank.validate(rig);int checks=0;
 auto check=[&](bool ok){if(!ok)throw std::runtime_error("Pose bank control failed");++checks;};
 auto rejects=[&](auto fn){bool threw=false;try{fn();}catch(const std::invalid_argument&){threw=true;}check(threw);};
 auto serialize=[&](int version,int count,int joints){std::ostringstream s;s<<std::setprecision(17)<<"FHPOSEBANK "<<version<<' '<<count<<' '<<joints<<' '<<bank.model_sha<<' '<<bank.training_sha<<'\n';for(const auto&a:bank.angles){for(double x:a)s<<x<<' ';s<<'\n';}return s.str();};
 auto parse=[&](const std::string&s){std::istringstream in(s);return fhn::PoseBank::load(in);};
 check(parse(serialize(1,int(bank.angles.size()),20)).angles==bank.angles);
 rejects([&]{parse(serialize(2,8,20));});rejects([&]{parse(serialize(1,0,20));});rejects([&]{parse(serialize(1,33,20));});rejects([&]{parse(serialize(1,8,19));});rejects([&]{parse(serialize(1,8,20)+"extra");});
 auto truncated=serialize(1,8,20);truncated.resize(truncated.find('\n')+1);rejects([&]{parse(truncated);});rejects([&]{parse(truncated+"nan");});
 auto bad=bank;bad.model_sha=std::string(64,'0');rejects([&]{bad.validate(rig);});bad=bank;bad.angles[0][0]=rig.joints[0].upper+1;rejects([&]{bad.validate(rig);});bad=bank;bad.angles[0][0]=std::numeric_limits<double>::quiet_NaN();rejects([&]{bad.validate(rig);});
 fhn::RigPose pose;pose.side=rig.subject_side;pose.wrist.t={0,0,-450};for(int j=0;j<20;++j)pose.angles[j]=.5*(rig.joints[j].lower+rig.joints[j].upper);
 std::array<fhn::PixelObservation,21> obs{};auto missing=fhn::fit_pose_bank(bank,rig,pose,cams,obs);check(!missing.fit.sufficient&&missing.fits==0&&missing.chosen==-1);
 // Distinct clean shapes must remain observation-led, even when a bank template differs.
 for(int trial=0;trial<12;++trial){for(int j=0;j<20;++j)pose.angles[j]=rig.joints[j].lower+(rig.joints[j].upper-rig.joints[j].lower)*(.15+.7*((trial*7+j*3)%13)/12.);
  auto points=rig.landmarks(pose.angles,pose.wrist,pose.side);for(int i=0;i<21;++i)for(int v=0;v<2;++v){obs[i].present[v]=true;check(cams[v].project(points[i],obs[i].point[v]));}
  auto result=fhn::fit_pose_bank(bank,rig,pose,cams,obs);check(result.fit.sufficient&&result.fits==2&&result.chosen==-1&&result.fit.cost==0);check(result.fit.pose.angles==pose.angles);check(fhn::pose_bank_cost(rig,result.fit.pose,pose,cams,obs)==result.fit.cost);
 }
 for(auto&o:obs)o.present[1]=false;missing=fhn::fit_pose_bank(bank,rig,pose,cams,obs);check(!missing.fit.sufficient&&missing.fits==0);
 obs[0].sigma[0]=0;rejects([&]{fhn::fit_pose_bank(bank,rig,pose,cams,obs);});
 std::cout<<"controls="<<checks<<" failures=0\n";return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
