#include "controller_takeover.h"
#include <cassert>
#include <cstdio>
int main(){
    fhnav::ButtonGate g;
    assert(!g.sample(true,1));assert(!g.sample(true,1)); // held launch button
    assert(!g.sample(true,0));assert(g.sample(true,1));assert(!g.sample(true,1));
    assert(g.sample(true,3)); // a second button while one is held
    assert(!g.sample(true,0));assert(g.sample(true,1ull<<32)); // trigger
    assert(!g.sample(false,0));assert(!g.sample(true,1)); // reconnect held
    assert(!g.sample(true,0));assert(g.sample(true,2));
    g.reset();assert(!g.sample(true,2));assert(!g.sample(true,0));assert(g.sample(true,2));
    std::puts("PASS controller takeover: release/repress, held selection, disconnect, second button, trigger");
}
