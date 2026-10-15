// Reproduce diagnostic stream visibility with saved audit rows; no tracker/camera.
#include <fstream>
#include <iostream>
#include <string>
#include <filesystem>
#include <unistd.h>
int main(int argc,char** argv) {
    if(argc!=2)return 2;
    for(bool flush:{false,true}) {
        std::ifstream input(argv[1]);if(!input)return 3;
        auto path=std::filesystem::path("/tmp")/("ume-audit-probe-"+std::to_string(getpid())+(flush?"-flush":"-buffered"));
        if(std::filesystem::exists(path))return 4;
        std::ofstream output(path);std::string line;unsigned sets=0,updates=0;uintmax_t previous=0;
        while(sets<300&&std::getline(input,line)) {
            for(char c:line)output.put(c);
            output.put('\n');
            if(line=="end") {
                ++sets;if(flush)output.flush();
                auto bytes=std::filesystem::file_size(path);
                updates+=bytes!=previous;previous=bytes;
            }
        }
        output.close();std::filesystem::remove(path);
        std::cout<<"flush="<<flush<<" sets="<<sets<<" file_size_updates="<<updates<<std::endl;
        if(flush&&updates!=sets)return 5;
    }
}
