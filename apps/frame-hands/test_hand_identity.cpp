// Identity evidence test; does not open camera/runtime files.
#include "hand_identity.h"
#include <cassert>
#include <cstdio>
int main() {
    HandIdentity h;
    int64_t t=1000000000;
    auto feed=[&](int vote,int n){for(int i=0;i<n;++i)h.observe(vote,t+=33333333);};
    feed(1,5); assert(!h.latched);
    h.observe(1,t); assert(!h.latched); // same observation twice
    feed(1,1); assert(h.latched && h.right);
    feed(-1,90); assert(h.right); // weak/curled/monocular data cannot rename it
    feed(0,12); assert(h.right); // short strong conflict
    feed(-1,1);feed(0,29);assert(h.right);
    feed(0,1); assert(!h.right && h.corrections==1); // correct wrong initialization
    HandIdentity fresh;
    for(int i=0;i<20;++i)fresh.observe(i%2,t+=33333333);
    assert(!fresh.latched);
    fresh=HandIdentity{};
    for(int i=0;i<5;++i)fresh.observe(1,t+=33333333);
    fresh.observe(1,t+=200000000);assert(!fresh.latched); // gap restarts evidence
    for(int i=0;i<5;++i)fresh.observe(1,t+=33333333);
    assert(fresh.latched && fresh.right);
    HandIdentity mono;
    for(int i=0;i<11;++i)mono.observe(0,t+=33333333,false);
    assert(!mono.latched);
    mono.observe(0,t+=33333333,false);assert(mono.latched && !mono.right);
    for(int i=0;i<59;++i)mono.observe(1,t+=33333333,false);
    assert(!mono.right);
    mono.observe(1,t+=33333333,false);assert(mono.right && mono.corrections==1);
    std::puts("PASS: acquisition, independent observations, weak evidence, transient conflict, correction, gap reset");
}
