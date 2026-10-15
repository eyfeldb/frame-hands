#include "stability.h"
#include "skeleton.h"
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>

std::array<fhs::V,21> reference(){
    std::array<fhs::Transform,31> local{};
    for(int i=0;i<31;++i)local[i]={fhs::rest[0][i].p,fhs::rest[0][i].q};
    auto bones=fhs::forward(local);std::array<fhs::V,21> p{};
    for(int i=0;i<21;++i)p[i]=bones[fhs::landmark_bone[i]].p;
    return p;
}
int main(int argc,char** argv){
    auto p=reference();assert(fhs::Stability::geometry(p));
    fhs::Stability s;assert(s.update(p,1,1000,1000));
    auto spike=p;for(auto& v:spike)v.x+=.7;
    assert(!s.update(spike,1,1033,1033));assert(s.rejected==1);
    assert(s.visible(1080)&&!s.visible(1081));
    assert(!s.update(p,1,1000,1050));assert(s.stamp==1000);
    assert(s.update(p,1,1066,1066));assert(s.distance(s.value,p)<1e-9);
    assert(!s.update(spike,1,1099,1099));assert(s.update(spike,1,1132,1132));
    assert(s.update(p,2,1165,1165));assert(s.distance(s.value,p)<1e-9);
    auto invalid=p;invalid[8]=invalid[7];assert(!s.update(invalid,2,1198,1198));
    assert(s.stamp==1165);s.reset();assert(!s.visible(1200));
    assert(!s.update(p,1,1500,1400));assert(!s.update(p,1,1000,1400));
    // Stationary +/-3 mm noise and a 2 m/s sweep, using known synthetic truth.
    double input_energy=0,output_energy=0,motion_error=0;
    for(int i=0;i<180;++i){auto noisy=p;double e=i%2?.003:-.003;for(auto& v:noisy)v.x+=e;
        assert(s.update(noisy,1,2000+33*i,2000+33*i));
        if(i>20){input_energy+=e*e;output_energy+=std::pow(s.value[0].x-p[0].x,2);}}
    s.reset();for(int i=0;i<180;++i){auto moving=p;for(auto& v:moving)v.x+=2.*i*.033;
        assert(s.update(moving,1,10000+33*i,10000+33*i));
        if(i>20)motion_error=std::max(motion_error,(s.value[0]-moving[0]).len());}
    assert(output_energy<input_energy*.5);assert(motion_error<.02);
    // Wrist rotation and fixed-wrist finger reversal preserve nonzero segments.
    s.reset();for(int i=0;i<360;++i){auto turned=p;double a=i*.04;
        fhs::Q q{std::cos(a/2),0,std::sin(a/2),0};for(auto& v:turned)v=q.rotate(v-p[0])+p[0];
        s.update(turned,1,20000+33*i,20000+33*i);assert(s.geometry(s.value));}
    s.reset();s.update(p,1,40000,40000);auto bent=p;
    for(int j=6;j<=8;++j)bent[j]=p[5]-(p[j]-p[5]);
    s.update(bent,1,40033,40033);assert(s.geometry(s.value));
    for(int j=1;j<21;++j)assert((s.value[j]-s.value[s.parent[j]]).len()>.003);
    // Both hands are independent; a held hand remains fixed in world space while
    // the viewer's head moves. The hold does not invent motion or extend freshness.
    fhs::Stability left,right;auto world=p;for(auto& v:world)v=v+fhs::V{2,1.6,-3};
    assert(left.update(world,7,50000,50040));assert(right.update(p,8,50000,50040));
    assert(!right.update(invalid,8,50033,50050));
    assert(left.visible(50080)&&right.visible(50080));
    assert(left.distance(left.value,world)<1e-9);
    fhs::Q head_turn{std::cos(.3),0,std::sin(.3),0};fhs::V head_pos{.2,1.7,.1};
    for(int i=0;i<21;++i){auto in_head=head_turn.inv().rotate(left.value[i]-head_pos);
        assert((head_turn.rotate(in_head)+head_pos-world[i]).len()<1e-9);}
    assert(!left.visible(50081)&&!right.visible(50081));
    assert(left.update(p,9,50100,50120));assert(left.distance(left.value,p)<1e-9);
    left.reset();assert(!left.visible(50121));assert(!right.visible(50121));
    std::printf("synthetic_pass noise_rms_ratio=%.6f constant_2mps_max_lag_mm=%.6f\n",std::sqrt(output_energy/input_energy),motion_error*1000);
    // Replay published source poses: no camera, inference, or private coordinates printed.
    if(argc==2){
        std::ifstream file(argv[1]);if(!file)return 2;
        fhs::Stability tracks[2];unsigned valid=0,bad=0,kept=0,new_bad=0,ambiguous=0,duplicate_ids=0,groups=0;std::vector<double> steps,rawsteps;
        std::array<fhs::V,21> last[2]{},lastraw[2]{};uint32_t lastid[2]{};uint64_t lasttime[2]{};
        // Pose files contain every tracker hypothesis. Publisher first chooses
        // the most-established hypothesis for each side (frames, then lower ID),
        // before the relay/mailbox see it. Every pose row is one successful hand
        // update, matching Hand::frames, including malformed published geometry.
        struct Hypothesis{fhs::Hand hand;unsigned frames;};
        std::map<uint32_t,unsigned> track_frames;
        std::vector<Hypothesis> slots[2];uint64_t group_time=0;
        auto flush=[&]{
            if(!group_time)return;++groups;
            fhs::Packet packet;packet.sequence=groups;packet.capture_ms=packet.sent_ms=group_time;
            for(int side=0;side<2;++side){
                if(!slots[side].empty()){
                    std::sort(slots[side].begin(),slots[side].end(),[](const Hypothesis& a,const Hypothesis& b){return a.frames!=b.frames?a.frames>b.frames:a.hand.id<b.hand.id;});
                    packet.hands[side]=slots[side][0].hand;
                    ambiguous+=unsigned(slots[side].size()-1);
                }
                slots[side].clear();
            }
            if(packet.hands[0].id&&packet.hands[0].id==packet.hands[1].id)++duplicate_ids;
            fhs::Mailbox mailbox;assert(mailbox.accept(packet,group_time));
            for(int side=0;side<2;++side){
                const auto& h=mailbox.packet.hands[side];
                if(!(h.flags&fhs::Present)){bad+=bool(packet.hands[side].flags&fhs::Present);continue;}
                std::array<fhs::V,21> points{};for(int i=0;i<21;++i)points[i]={h.points[i][0],h.points[i][1],h.points[i][2]};
                ++valid;auto& t=tracks[side];if(!t.update(points,h.id,group_time,group_time))continue;++kept;
                if(!t.geometry(t.value))++new_bad;
                if(lastid[side]==h.id&&group_time>lasttime[side]&&group_time-lasttime[side]<=70){steps.push_back(t.distance(last[side],t.value)*1000);rawsteps.push_back(t.distance(lastraw[side],points)*1000);}
                last[side]=t.value;lastraw[side]=points;lastid[side]=h.id;lasttime[side]=group_time;
            }
        };
        std::string line;
        while(std::getline(file,line)){
            std::istringstream row(line);double time,scale;uint32_t id;char side;int views;
            if(!(row>>time>>id>>side>>views>>scale))return 3;
            double ignored;for(int i=0;i<63;++i)if(!(row>>ignored))return 4;
            std::array<fhs::V,21> points{};for(auto& v:points)if(!(row>>v.x>>v.y>>v.z))return 5;
            if(side!='L'&&side!='R')return 6;
            uint64_t stamp=1000+uint64_t(std::llround(time*1000));if(stamp<group_time)return 7;
            if(group_time!=stamp){flush();group_time=stamp;}
            fhs::Hand h;h.id=id;h.flags=fhs::Present;
            for(int i=0;i<21;++i){h.points[i][0]=float(points[i].x);h.points[i][1]=float(points[i].y);h.points[i][2]=float(points[i].z);}
            slots[side=='L'?0:1].push_back({h,++track_frames[id]});
        }
        flush();
        auto p95=[](std::vector<double> v){std::sort(v.begin(),v.end());return v.empty()?0:v[size_t((v.size()-1)*.95)];};
        std::printf("valid=%u invalid=%u accepted=%u quarantined=%u filter_fallbacks=%u newly_invalid=%u matched_raw_step_p95_mm=%.6f candidate_step_p95_mm=%.6f\n",valid,bad,kept,tracks[0].rejected+tracks[1].rejected,tracks[0].fallbacks+tracks[1].fallbacks,new_bad,p95(rawsteps),p95(steps));
        std::printf("observed_timestamp_groups=%u duplicate_side_hypotheses_discarded=%u duplicate_identity_groups=%u matched_steps=%zu\n",groups,ambiguous,duplicate_ids,steps.size());
        assert(new_bad==0);
    }
}
