//
// Created by Gabriele Facciolo on 08/04/16.
//

#include "img.hpp"
#include "imgio.hpp"
#include <vector>
#include <list>
#include <algorithm>
#include <string.h>
#include <stdio.h>

#include "model.hpp"
#include "mergetreelog.h"

using namespace std;




Img apply_model_to_segmentation(Img &segment, Img &in, int order, double *error2=NULL, double *error2_real=NULL) {
    int nx=in.nx, ny=in.ny, nch=in.nch;
    Model model = Model(order,nch,nx,ny);
//    int msz = model.get_model_size();
    int msz = 1000;
    long double err2=0, err2_real=0;

    vector<list<Pixel> > tmp(nx*ny*2);
    for(int y=0, i=0; y<ny; y++)
        for(int x=0; x<nx; x++, i++)
            tmp[(int) segment[i]].push_back(Pixel(x,y,i));

    Img out(nx,ny,nch);
    for(vector<list<Pixel> >::iterator n = tmp.begin(); n != tmp.end(); n++) {
        double mm[msz];
        for(int i=0; i<msz;i++) mm[i] = 0;

        if (n->size()==0) continue;
        model.compute(mm, in, *n, n->size());

        for(list<Pixel>::iterator t = n->begin(); t!=n->end(); t++) {
            for(int c=0;c<nch;c++) {
                float v = model.evaluate(mm, t->x_pos, t->y_pos, c);
                out[t->x_pos + t->y_pos*nx + c*nx*ny] = v;
                v -= in[t->x_pos + t->y_pos*nx + c*nx*ny];
                err2_real += v*v;
            }
        }
        err2 += model.get_parameters_err2(mm)[0];
    }
    if(error2)      *error2 = err2;
    if(error2_real) *error2_real = err2_real;
    return out;
}














Img draw_labels_colors(const Img &lab) {
    // HACK to show the labels
    // NO warranty that adjacent regions get different colors
    int ncol=lab.nx, nrow=lab.ny;
    Img clab (ncol,nrow,3);
    for(int i=0;i<ncol*nrow;i++){
        int id = lab[i];
        unsigned char r = (unsigned char) ((id+7)>>4);
        unsigned char g = (unsigned char) ((id+1)>>2);
        unsigned char b = (unsigned char) id;
        clab[i + ncol*nrow*0] = r;
        clab[i + ncol*nrow*1] = g;
        clab[i + ncol*nrow*2] = b;
    }
    return clab;
}


// c: pointer to original argc
// v: pointer to original argv
// o: option name after hyphen
// d: default value (if NULL, the option takes no argument)
static char *pick_option(int *c, char ***v, char *o, char *d)
{
    int argc = *c;
    char **argv = *v;
    int id = d ? 1 : 0;
    for (int i = 0; i < argc - id; i++)
        if (argv[i][0] == '-' && 0 == strcmp(argv[i] + 1, o)) {
            char *r = argv[i + id] + 1 - id;
            *c -= id + 1;
            for (int j = i; j < argc - id; j++)
                (*v)[j] = (*v)[j + id + 1];
            return r;
        }
    return d;
}



// main program
int main(int argc, char** argv){

    char *nregions = pick_option(&argc, &argv, (char*) "n", (char*) "2");
    int order      = atoi(pick_option(&argc, &argv, (char*) "o", (char*) "1"));
    float lambda   = atof(pick_option(&argc, &argv, (char*) "l", (char*) "inf"));
    char *print_lambdas = pick_option(&argc, &argv, (char*) "L", (char*) "");
    if (argc < 4) {
        fprintf(stderr, "too few parameters\n"
                "   %s [-o model_order(1)] [-l lambda(inf)] [-n \"3 16 42\" list of regions(2)] "
                "      [-L lambdalog.txt] treelog in out [out_labels]\n"
                "   The treelog file is produced with pamss\n", argv[0]);
        return 1;
    }


    // load the image
    // load the tree
    //    - leafs are associated to pixels
    //    - internal nodes contain lambda and relative errors

    // * cut at a certain lambda: DFS: compute the model of a region (list of pixels)

    // *

    char *log_file = argv[1];
    char *in_file  = argv[2];
    char *out_file = argv[3];
    char *out_labels_file = argc > 4 ? argv[4] : NULL;

    Img in = iio_read_vector_split(in_file);
    int nx = in.nx;
    int ny = in.ny;
    int nch = in.nch;

    vector<Merge_log> t = readmlogfileb(log_file);

    // just print the lambdas
    if(strcmp(print_lambdas,"") != 0) {
        FILE *fp = fopen(print_lambdas, "wt");
        int per = 0;
        for (int i=0; i<t.size(); i++)
            fprintf(fp, "%ld %f %d\n", t.size()-i, t[i].lambda, per+=t[i].plenght);
        fclose(fp);
    }

   if(isfinite(lambda)) {
//    // Write
    Img segment = get_segmentation_from_merge_tree_log(t, nx, ny, atoi(nregions), lambda);
    if(out_labels_file!=NULL) {
        Img tmp =  draw_labels_colors(segment);
        iio_write_vector_split(out_labels_file, tmp);
    }

    double err2,err2real;
    Img out = apply_model_to_segmentation(segment, in, order, &err2, &err2real);
    iio_write_vector_split(out_file, out);
    printf("err2: %lf err2real %lf\n", err2, err2real);

   } else {


    const char *sep = " \t";
    char *word, *phrase, *brkt, *brkb;
    for (word = strtok_r(nregions, sep, &brkt);
         word;
         word = strtok_r(NULL, sep, &brkt))
    {
        int nreg = atoi(word);
        char fname[2048];
        snprintf(fname, 2047, "%06d_%s", nreg, out_labels_file);

//    // Write
        Img segment = get_segmentation_from_merge_tree_log(t, nx, ny, nreg, lambda);
        if(out_labels_file!=NULL) {
            Img tmp =  draw_labels_colors(segment);
            iio_write_vector_split(fname, tmp);
        }

        snprintf(fname, 2047, "%06d_%s", nreg, out_file);
        double err2,err2real;
        Img out = apply_model_to_segmentation(segment, in, order, &err2, &err2real);
        iio_write_vector_split(fname, out);

        printf("err2: %lf err2real %lf\n", err2, err2real);
    }

   }


    return 0;
}

