// Link gesture values from synthetic hands: open (nothing), thumb on index tip (index pinch), fist (grip), point.
#include "link_gestures.h"
#include <cassert>
#include <cstdio>
using fhs::V;
static std::array<V,21> open_hand(){
    // Flat hand in the x-y plane, wrist at the origin, fingers along +y, 63 mm across the knuckles.
    std::array<V,21> p{};
    const double mx[5]={-.030,-.022,0,.022,.040};   // 22 mm between fingers, 62 mm across the knuckles
    for(int f=0;f<5;++f)for(int j=0;j<4;++j){const int i=1+f*4+j;p[i]={mx[f],(f==0?.03:.09)+j*.025,0};}
    p[4]={-.055,.085,0};                      // thumb tip out to the side
    return p;
}
int main(){
    auto p=open_hand();
    auto g=fhs::link_gestures(p);
    for(int n=0;n<4;++n)assert(g.pinch[n]==0.f);
    assert(g.grip==0.f&&!g.point);
    // Thumb tip onto the index tip: full index pinch; the neighbouring middle finger reads a partial value
    // (its tip is 18 mm away in this synthetic hand), ring and pinky nothing. SteamVR's binding clicks at 0.65.
    auto q=p;q[4]=q[8];q[4].x+=.004;
    g=fhs::link_gestures(q);
    assert(g.pinch[0]==1.f&&g.pinch[1]<.6f&&g.pinch[2]==0.f&&g.pinch[3]==0.f);
    // 15 mm gap: partial, still above the dashboard click level for a 63 mm hand? (open 25 mm, close 8 mm -> 0.59)
    auto r=p;r[4]=r[8];r[4].x+=.015;
    g=fhs::link_gestures(r);assert(g.pinch[0]>.5f&&g.pinch[0]<.7f);
    // Fist: all fingertips pulled to the palm.
    auto f=p;for(int finger=1;finger<5;++finger){const int tip=4+finger*4;f[tip]={p[1+finger*4].x,.05,.0};}
    g=fhs::link_gestures(f);assert(g.grip>.6f&&!g.point);
    // Point: index straight, others curled (a pointing hand is also a partial grip; games read index_point separately).
    auto pt=f;pt[8]=p[8];g=fhs::link_gestures(pt);assert(g.point&&g.pinch[0]==0.f);
    std::puts("PASS link gestures: open, index pinch, partial pinch, fist, point");
}
