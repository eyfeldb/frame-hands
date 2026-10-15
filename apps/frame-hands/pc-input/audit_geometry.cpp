// Inspect earlier replay poses using the actual native bridge validation code.
// No runtime, pixels, network, pose publication or smoothing changes.
#include "protocol.h"
#include <cstdio>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>
int main(int argc,char** argv){
    if(argc!=2&&argc!=3){std::fprintf(stderr,"usage: audit-geometry replay-poses.txt [per-hand-status.txt]\n");return 2;}
    std::ifstream in(argv[1]);if(!in)return 2;
    std::ofstream status;if(argc==3){status.open(argv[2]);if(!status)return 2;status.precision(10);}
    std::map<std::string,int> counts;int rows=0,rotated=0,rotation_changed=0;
    std::string line;
    while(std::getline(in,line)){
        if(line.empty())continue;
        std::istringstream row(line);double time,scale;int id,views;std::string side;
        if(!(row>>time>>id>>side>>views>>scale)||id<=0||(side!="L"&&side!="R"))return 3;
        std::vector<double> values;double x;while(row>>x)values.push_back(x);
        if(!row.eof()||(values.size()!=126&&values.size()!=189))return 3;
        fhs::Hand h;h.id=uint32_t(id);h.flags=fhs::Present|(views>=2?fhs::Stereo:0);
        for(int k=0;k<63;++k)h.points[k/3][k%3]=float(values[63+k]);
        auto issue=fhs::hand_issue(h);++counts[fhs::hand_issue_name(issue)];++rows;
        if(status)status<<time<<" "<<id<<" "<<fhs::hand_issue_name(issue)<<"\n";
        // Rotate head-relative points around every principal axis, including edge-on.
        // A rigid rotation must not change segment lengths or palm area.
        for(int axis=0;axis<3;++axis)for(int deg=0;deg<360;deg+=15){
            double a=deg*3.14159265358979323846/360;
            fhs::Q q{std::cos(a),axis==0?std::sin(a):0,axis==1?std::sin(a):0,axis==2?std::sin(a):0};
            auto turned=h;
            for(int k=0;k<21;++k){auto p=h.points[k];auto v=q.rotate({p[0],p[1],p[2]});
                turned.points[k][0]=float(v.x);turned.points[k][1]=float(v.y);turned.points[k][2]=float(v.z);}
            ++rotated;rotation_changed+=fhs::hand_issue(turned)!=issue;
        }
    }
    std::printf("{\"rows\":%d,\"rotation_checks\":%d,\"rotation_changed_verdict\":%d,\"reasons\":{",rows,rotated,rotation_changed);
    bool first=true;for(auto& [name,n]:counts){std::printf("%s\"%s\":%d",first?"":",",name.c_str(),n);first=false;}
    std::puts("},\"scope\":\"Earlier replay geometry, not latest gestures or physical accuracy\"}");
    return rotation_changed?1:0;
}
