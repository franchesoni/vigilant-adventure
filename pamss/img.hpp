//
// Created by Gabriele Facciolo on 08/03/16.
//

#ifndef IMG_H_
#define IMG_H_
#include <vector>
#include <assert.h>

struct Img
{
   std::vector<float > data;
   union{
      int sz[3];
      struct{
         union { int ncol; int nx; };
         union { int nrow; int ny; };
         int nch;
      };
   };
//   int nx;
//   int ny;
//   int nch;


	Img(int nx, int ny, int nch);

	inline Img() {nx=0;ny=0;nch=0;}

   inline float operator[](int i) const { assert(i>=0 && i < nch*nx*ny); return data[i];}
   inline float& operator[](int i) { assert(i>=0 && i < nch*nx*ny); return data[i];}


   inline float val(int x, int y, int c) { return data[x+y*nx+c*nx*ny];}


//   private:
//   Img(const Img&);      // disable copy constructor
//   void operator=(const Img&);
//	  Img& operator= (const Img&p);

};

#endif /* IMG_H_ */
