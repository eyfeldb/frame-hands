#include "protocol.h"
#include <cassert>
#include <cstdio>
#include <limits>
int main(){
    const uint64_t t=1000000000;fh::Packet p;p.sequence=1;p.sent_ns=t;p.flags=fh::Visible;
    fh::Mailbox box;assert(box.accept(p,t));assert(!(box.read(t,true).flags&fh::Click));
    p.sequence=2;p.flags|=fh::Click;assert(box.accept(p,t));assert(box.read(t+100000000,true).flags&fh::Click);
    assert(!box.read(t+fh::expiry_ns+1,true).flags);assert(!box.read(t,false).flags);
    assert(!(box.read(t,true).flags&fh::Click));
    assert(!box.accept(p,t));p.sequence=3;p.sent_ns=t+1;assert(!box.accept(p,t));
    p.sent_ns=t;p.quaternion[0]=0;assert(!box.accept(p,t));p.quaternion[0]=1;
    p.position[0]=std::numeric_limits<double>::quiet_NaN();assert(!box.accept(p,t));p.position[0]=0;
    p.flags=fh::Click;assert(!box.accept(p,t));p.flags=0;assert(box.accept(p,t));assert(!box.read(t,true).flags);
    fh::PinchGate pinch;assert(!pinch.update(true,7,true));assert(!pinch.update(true,7,false));assert(pinch.update(true,7,true));
    assert(!pinch.update(false,7,true));assert(!pinch.update(true,7,true));assert(!pinch.update(true,7,false));assert(pinch.update(true,7,true));
    assert(!pinch.update(true,8,true));assert(!pinch.update(true,8,false));assert(pinch.update(true,8,true));assert(!pinch.update(true,8,false));
    fh::TouchGate touch;assert(!touch.update(true,7,-.001));assert(!touch.update(true,7,.025));assert(touch.update(true,7,.002));
    assert(touch.update(true,7,.010));assert(!touch.update(true,7,.020));assert(touch.update(true,7,.001));
    assert(!touch.update(false,7,.001));assert(!touch.update(true,7,.001));assert(!touch.update(true,7,.020));assert(!touch.update(true,8,.001));
    assert(!touch.update(true,8,.03));assert(!touch.update(true,8,-.1));assert(!touch.update(true,8,0));
    fh::ClaimPulse claim;assert(!claim.update(true,t));assert(!claim.update(true,t+299999999));assert(claim.update(true,t+300000000));
    assert(claim.update(true,t+370000000));assert(!claim.update(true,t+380000000));assert(!claim.update(false,t+400000000));
    assert(!claim.update(true,t+500000000));assert(claim.update(true,t+800000000));
    std::puts("PASS packet validation, replay/future rejection, dead bridge/head release, pinch rearm/identity change, contact approach/retraction/loss, delayed multi-frame pointer claim");
}
