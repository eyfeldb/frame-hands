#include "action_watchdog.h"
#undef NDEBUG
#include <cassert>
#include <cstdio>
int main(){
    fhs::ActionWatchdog w;
    assert(!w.update(0,false,false));assert(!w.update(100,true,false));
    assert(!w.update(8099,true,false));assert(w.update(8100,true,false));
    assert(!w.update(8110,true,true));assert(!w.update(8120,true,false));
    assert(!w.update(9000,false,false));assert(!w.update(10000,true,false));
    assert(!w.update(17999,true,false));assert(w.update(18000,true,false));
    std::puts("PASS action watchdog: timeout boundary, action recovery, acquisition/loss reset");
}
