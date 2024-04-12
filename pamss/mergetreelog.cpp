//
// Created by Gabriele Facciolo on 14/04/16.
//

#include "mergetreelog.h"
#include <stdio.h>

/*    Disjoint Set Forest     */
static int dsf_find(vector<int> &t, int a) {
    if (a != t[a])
        t[a] = dsf_find(t, t[a]);
    return t[a];
}

inline int dsf_make_link(vector<int> &t, int a, int b) {
    if (a < b) { // arbitrary choice
        t[b] = a;
        return a;
    } else {
        t[a] = b;
        return b;
    }
}

inline int dsf_join(vector<int> &t, int a, int b) {
    a = dsf_find(t, a);
    b = dsf_find(t, b);
    if (a != b)
        b = dsf_make_link(t, a, b);
    return b;
}
/*     End: Disjoint Set Forest     */


void showmlog(vector<Merge_log> &L) {
    for(vector<Merge_log>::iterator x = L.begin(); x != L.end(); x++)
        printf("%d %d %g %.20g %g %g\n", x->a, x->b, x->lambda, x->err, x->area, x->plenght);
}

vector<Merge_log> readmlogfileb(char* fname) {
    vector<Merge_log> L;
    Merge_log tmp;
    FILE *f = fopen(fname, "rb");
    while ( fread(&tmp, sizeof(Merge_log), 1, f))
        L.push_back(tmp);
    fclose(f);
    printf("read %ld log entries\n", L.size());
    return L;
}

void writemlogfileb(char* fname, vector<Merge_log> &L) {
    FILE *f = fopen(fname, "wb");
    for(vector<Merge_log>::iterator x = L.begin(); x != L.end(); x++)
        fwrite(&(*x), sizeof(Merge_log), 1, f);
    fclose(f);
    printf("written %ld log entries\n", L.size());
}

Img get_segmentation_from_merge_tree_log(vector<Merge_log> &L, int nx, int ny, int nregions, double lambda) {
    // replay log to form the segmentation
    vector<int> t(nx*ny);
    for(int i=0; i<nx*ny; i++) t[i] = i;

    for(int i=0; i+nregions-1< (int)L.size(); i++) {
        if (L[i].lambda > lambda) break;
        dsf_join(t, L[i].a, L[i].b);
    }

    Img out(nx,ny,1);
    for(int i=0; i<nx*ny; i++)
        out[i] = dsf_find(t, i);

    return out;
}

