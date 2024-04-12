//
// Created by Gabriele Facciolo on 08/03/16.
//

#include "img.hpp"
extern "C" {
#include "iio.h"
#include <stdlib.h>
}

struct Img iio_read_vector_split(char *nm)
{
   struct Img out;
   float *tmpout = iio_read_image_float_split(nm, &out.nx, &out.ny, &out.nch);
   out.data.assign(tmpout,tmpout + out.nx * out.ny * out.nch);
   free (tmpout);
   return out;
}


void iio_write_vector_split(char *nm, struct Img &out)
{
   // .front() -> .data() in C++11
   iio_save_image_float_split(nm, &(out.data.front()), out.nx, out.ny, out.nch);
}

