// Pure publication contract test. No runtime files, camera or XR session opened.
#include "io.h"
#include <cassert>
#include <cstdio>

int main() {
    Hand established_right{},conflicting_right{},left{};
    established_right.id=8;established_right.frames=100;established_right.right_score=.99;
    conflicting_right.id=16;conflicting_right.frames=70;conflicting_right.right_score=.6;
    left.id=17;left.frames=3;left.right_score=.1;
    auto chosen=select_publication_hands({&conflicting_right,&left,&established_right});
    assert(chosen.size()==2 && chosen[0]->id==8 && chosen[1]->id==17);
    chosen=select_publication_hands({&conflicting_right,&established_right});
    assert(chosen.size()==1 && chosen[0]->id==8);
    chosen=select_publication_hands({&left,&established_right});
    assert(chosen.size()==2 && chosen[0]->id==8 && chosen[1]->id==17);
    assert(select_publication_hands({}).empty());
    conflicting_right.frames=100;
    chosen=select_publication_hands({&conflicting_right,&established_right});
    assert(chosen.size()==1 && chosen[0]->id==8);
    // Selection preserves measured coordinates and the original side calls.
    assert(established_right.right_score==.99 && conflicting_right.right_score==.6 && left.right_score==.1);
    std::puts("publication uniqueness, opposite-side preservation, empty input and deterministic ties PASS");
}
