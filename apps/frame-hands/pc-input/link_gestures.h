#pragma once
#include "model_protocol.h"
#include <algorithm>
#include <array>
#include <cmath>
namespace fhs {
// Gesture values for Steam Link's hand-tracking controller type ("svl_hand_interaction_augmented"), whose
// PC-side input profile exposes /input/<finger>_pinch/value, /input/grip/value and /input/index_point/touch
// next to the skeleton. Computed from the 21 world-space joints the driver already retargets. Pinch is 1 at
// thumb-fingertip contact and 0 once the gap passes 40% of the hand width (index MCP to pinky MCP); SteamVR's
// own dashboard binding for this type clicks at 0.65, i.e. about 14 mm for a 63 mm wide hand. Grip is the
// four fingers' mean curl (fingertip distance from the wrist over the knuckle's: ~1.9 straight, ~0.9 fist).
struct LinkGestures {float pinch[4]{};float grip=0;bool point=false;};
inline LinkGestures link_gestures(const std::array<V,21>& p){
    LinkGestures g;
    auto dist=[&](int a,int b){V d{p[a].x-p[b].x,p[a].y-p[b].y,p[a].z-p[b].z};return d.len();};
    const double width=std::clamp(dist(5,17),.045,.11);
    const double close=std::max(.13*width,.008),open=std::max(.40*width,.022);
    const int tips[4]={8,12,16,20},mcps[4]={5,9,13,17};
    double curl[4];
    for(int n=0;n<4;++n){
        const double gap=dist(4,tips[n]);
        g.pinch[n]=float(std::clamp((open-gap)/(open-close),0.,1.));
        const double knuckle=dist(0,mcps[n]);
        curl[n]=knuckle>1e-6?dist(0,tips[n])/knuckle:2.;
    }
    const double others=(curl[1]+curl[2]+curl[3])/3,mean=(curl[0]+others*3)/4;
    g.grip=float(std::clamp((1.55-mean)/.45,0.,1.));
    g.point=curl[0]>1.55&&others<1.25;
    return g;
}
}
