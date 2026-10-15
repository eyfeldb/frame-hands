#include "controller_takeover.h"
#include <cassert>
#include <cstdio>
#include <limits>
int main(){
    fhnav::ButtonGate g;
    constexpr uint64_t t=1000000000;
    fhnav::TakeoverWindow window;
    assert(window.update(10,true,t));assert(!window.ready(t));
    assert(window.update(10,true,t+100000000));
    // Click away, close the dashboard: dismissal is still inside settling time.
    assert(!window.update(10,false,t+120000000));
    assert(!window.ready(t+120000000));assert(!window.ready(t+399000000));
    assert(window.ready(t+400000000));
    assert(window.update(10,true,t+500000000));assert(!window.ready(t+500000000));
    assert(window.update(11,false,t+600000000));assert(!window.ready(t+600000000));
    // Activation stays held past the initial settling interval.
    assert(!g.sample(true,1,t));assert(!g.sample(true,1,t+300000000));
    assert(!g.armed);
    // A transient neutral sample must not convert the same held click to takeover.
    assert(!g.sample(true,0,t+320000000));
    assert(!g.sample(true,1,t+340000000));assert(!g.armed);
    assert(!g.sample(true,0,t+400000000));
    assert(!g.sample(true,0,t+599000000));assert(!g.armed);
    assert(!g.sample(true,0,t+600000000));assert(g.armed);
    // Delayed event from the activation click, drained after release, is stale.
    assert(!g.fresh_event(t+620000000,.32f,t+300000000));
    assert(!g.fresh_event(t+620000000,.04f,t+300000000));
    assert(g.fresh_event(t+640000000,.01f,t+300000000));
    assert(g.sample(true,1,t+640000000));
    assert(!g.sample(true,1,t+660000000));
    assert(g.sample(true,3,t+680000000)); // another actual button
    assert(!g.fresh_event(t+700000000,-1,t));
    assert(!g.fresh_event(t+700000000,std::numeric_limits<float>::quiet_NaN(),t));
    assert(!g.sample(false,0,t+720000000));assert(!g.armed);
    assert(!g.sample(true,1,t+740000000));assert(!g.armed);
    g.reset();assert(!g.fresh_event(t+800000000,0,t));
    puts("PASS: dashboard navigation/dismissal/reopen, selection change, activation hold, transient/stable release, stale/fresh events, component edges, disconnect, reset");
}
