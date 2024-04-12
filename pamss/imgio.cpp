//
// Created by Gabriele Facciolo on 08/03/16.
//

#include "img.hpp"
extern "C"
{
#include "iio.h"
#include <stdlib.h>
}
#include "cnpy.h"

struct Img read_npy(char *nm)
{
   printf("loading npy...");
   cnpy::NpyArray arr = cnpy::npy_load(nm);
   float *loaded_data = arr.data<float>();
   int nx = arr.shape[2];
   int ny = arr.shape[1];
   int nch = arr.shape[0];
   printf("size is %d %d %d\n", nx, ny, nch);
   struct Img img(nx, ny, nch);
   img.data.assign(loaded_data, loaded_data + nx * ny * nch);
   return img;
}

struct Img iio_read_vector_split(char *nm)
{
   struct Img out;
   float *tmpout = iio_read_image_float_split(nm, &out.nx, &out.ny, &out.nch);
   out.data.assign(tmpout, tmpout + out.nx * out.ny * out.nch);
   free(tmpout);
   return out;
}

void iio_write_vector_split(char *nm, struct Img &out)
{
   // .front() -> .data() in C++11
   iio_save_image_float_split(nm, &(out.data.front()), out.nx, out.ny, out.nch);
}
