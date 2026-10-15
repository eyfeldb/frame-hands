#include "xrcams.h"
// Query descriptors and formats only. Never changes camera controls or reads images.
int main(void) {
  xr_state_t st;char error[512];
  if(!xr_discover(&st,"XRService",error,sizeof(error))){fprintf(stderr,"%s\n",error);return 1;}
  printf("cameras=%d groups=%d\n",st.ncameras,st.ngroups);
  for(int n=0;n<st.ncameras;++n){xr_camera_t* c=&st.cameras[n];xr_layout_t l;xr_camera_layout(c,&l);
    printf("video%d sensor=%s format=%c%c%c%c width=%u height=%u pitch=%u layout=%s\n",c->node,c->sensor,
      c->pixfmt&255,(c->pixfmt>>8)&255,(c->pixfmt>>16)&255,(c->pixfmt>>24)&255,c->width,c->height,l.pitch,xr_fmt_name(l.fmt));
  }
  return 0;
}
