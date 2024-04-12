//
// Created by Gabriele Facciolo on 15/04/16.
//

#ifndef PROJECT_MERGETREELOG_H
#define PROJECT_MERGETREELOG_H

#include <vector>
#include "img.hpp"
#include <cmath>

using namespace std;

struct Merge_log {
    int a,b;
    double lambda;
    double err;
    double area;
    double plenght;
};

void showmlog(vector<Merge_log> &L);
vector<Merge_log> readmlogfileb(char* fname);
void writemlogfileb(char* fname, vector<Merge_log> &L);
Img get_segmentation_from_merge_tree_log(vector<Merge_log> &L, int nx, int ny, int nregions, double lambda = INFINITY);

#endif //PROJECT_MERGETREELOG_H
