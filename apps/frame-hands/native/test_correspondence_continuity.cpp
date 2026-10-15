#include "correspondence_continuity.h"
#include <iostream>
int main(){try{
 int checks=0;auto check=[&](bool ok){if(!ok)throw std::runtime_error("Continuity lifecycle control failed");++checks;};
 fhn::CorrespondenceResult strong;strong.chosen=0;strong.eligible[0]=true;strong.original_disagreement=10;strong.alternative_disagreement={2,2};strong.fit.sufficient=strong.control.sufficient=true;
 auto weak=strong;weak.chosen=-1;weak.eligible={false,false};weak.alternative_disagreement={5,5};auto chosen=weak;chosen.chosen=0;
 auto record=[&](fhn::CorrespondenceMemory& m,std::int64_t t){m.commit(t,1,0,true,strong,strong);};
 fhn::CorrespondenceMemory m;check(m.permit(1,1,0,true,weak)==-1);record(m,1);
 check(m.permit(33333334,1,0,true,weak)==0);m.commit(33333334,1,0,true,weak,chosen);check(m.strong_stamp()==1);
 check(m.permit(66666667,1,0,true,weak)==0);m.commit(66666667,1,0,true,weak,chosen);check(m.strong_stamp()==1);
 check(m.permit(100000001,1,0,true,weak)==0);m.commit(100000001,1,0,true,weak,chosen);check(m.strong_stamp()==1);
 check(m.permit(100000002,1,0,true,weak)==-1);m.commit(100000002,1,0,true,weak,weak);check(m.strong_stamp()==0);
 record(m,110000000);check(m.permit(110000001,2,0,true,weak)==-1);check(m.permit(110000001,1,1,true,weak)==-1);check(m.permit(110000001,1,0,false,weak)==-1);
 auto clean=weak;clean.original_disagreement=1;check(m.permit(110000001,1,0,true,clean)==-1);m.commit(110000001,1,0,true,clean,clean);check(m.strong_stamp()==0);
 record(m,120000000);auto unsupported=weak;unsupported.control.sufficient=false;check(m.permit(120000001,1,0,true,unsupported)==-1);m.commit(120000001,1,0,true,unsupported,unsupported);check(m.strong_stamp()==0);
 record(m,130000000);auto worse=weak;worse.alternative_disagreement[0]=11;check(m.permit(130000001,1,0,true,worse)==-1);
 auto invalid=weak;invalid.original_disagreement=std::numeric_limits<double>::infinity();check(m.permit(130000001,1,0,true,invalid)==-1);
 auto rejected=weak;rejected.alternative_disagreement[0]=2;check(m.permit(130000001,1,0,true,rejected)==-1);
 m.clear();check(m.permit(130000001,1,0,true,weak)==-1);
 bool failed=false;try{m.permit(130000000,1,0,true,weak);}catch(const std::invalid_argument&){failed=true;}check(failed);
 failed=false;try{m.commit(130000001,1,0,true,weak,chosen);}catch(const std::invalid_argument&){failed=true;}check(failed);
 m.commit(130000001,1,0,false,strong,strong);check(m.permit(130000002,1,0,true,weak)==-1);
 record(m,140000000);check(m.permit(240000001,1,0,true,weak)==-1);
 // Renewal requires both an earlier strict entry and fresh relative geometry.
 fhn::CorrespondenceMemory renewing(true);check(renewing.permit(1,1,0,true,weak)==-1);record(renewing,1);
 for(int i=1;i<=20;++i){auto t=1+33333333LL*i;check(renewing.permit(t,1,0,true,weak)==0);renewing.commit(t,1,0,true,weak,chosen);check(renewing.strong_stamp()==t);}
 auto t=700000000LL;auto marginal=weak;marginal.original_disagreement=7;check(renewing.permit(t,1,0,true,marginal)==0);auto old=renewing.strong_stamp();renewing.commit(t,1,0,true,marginal,chosen);check(renewing.strong_stamp()==old);
 check(renewing.permit(old+100000001,1,0,true,marginal)==-1);
 renewing.commit(old+100000001,1,0,true,marginal,marginal);check(renewing.strong_stamp()==0);
 record(renewing,800000000);renewing.commit(833333333,1,0,true,clean,clean);check(renewing.strong_stamp()==0);check(renewing.permit(833333334,1,0,true,weak)==-1);
 record(renewing,900000000);check(renewing.permit(933333333,2,0,true,weak)==-1);check(renewing.permit(933333333,1,1,true,weak)==-1);check(renewing.permit(933333333,1,0,false,weak)==-1);
 renewing.set_relative_renewal(false);check(renewing.strong_stamp()==0);check(renewing.permit(933333333,1,0,true,weak)==-1);
 std::cout<<"controls="<<checks<<" failures=0\n";return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
