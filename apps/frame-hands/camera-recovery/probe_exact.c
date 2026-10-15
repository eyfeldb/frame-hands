/* Metadata only: no camera ring, mmap, tracepoint or pixel reads. */
#include "camd/xrcams.h"
int main(void){static xr_state_t s;char err[256];if(!xr_discover(&s,"XRService",err,sizeof(err))){fprintf(stderr,"%s\n",err);return 1;}xr_print(&s,stdout);return 0;}
