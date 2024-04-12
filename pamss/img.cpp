//
// Created by Gabriele Facciolo on 08/03/16.
//

#include "img.hpp"
#include <vector>
#include <stdio.h>


Img::Img(int nx, int ny, int nch=1)
{
	this->nx = nx;
	this->ny = ny;
   this->nch = nch;
   this->data = std::vector<float >(nx*ny*nch,0);
}






#ifdef TEST_IMG
int main(){
   struct Img c,d;
   d.data.push_back(-1.1);
   struct Img a = Img(1002,1003,3);
   a[0]=10;
   struct Img b(1000,1000,3);
   b[0]=20;
   c = b;
   c[0]++;
   c[1]=-10;
   printf("%f %f %f %f %f %f %d\n", a[0],b[0],c[0],c[1],a.val(10,10,1),d[0],d.nx);
   printf("%d %d %d %d %d\n", a.nx,a.ny,a.sz[0],a.ncol, c.sz[0]);
}
#endif
