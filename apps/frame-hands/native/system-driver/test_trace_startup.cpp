// Exercises the real accepted hook with the production service environment.
// No camera acquisition, input publication, or saved hand data.
#include "track/calib.h"
#include "native/tracker_rig_hook.h"
#include <iostream>
int main(){
    std::map<std::string,Camera> cameras;std::string error;
    if(!load_calibration(cameras,error)){std::cerr<<error;return 2;}
    try{fhn::TrackerRigHook hook;hook.configure(cameras);
        if(!hook.enabled())return 3;
        std::cout<<"Real rig hook initialized with /dev/null trace and whole-pose processing\n";
    }catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}
    return 0;
}
