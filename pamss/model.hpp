//
// Created by Gabriele Facciolo on 04/03/16.
//

#ifndef _MODEL_H
#define _MODEL_H

#include <vector>
#include <list>
#include <cmath>
#include "rag.hpp"    // definition of node
#include "img.hpp"

using namespace std;



//#define USE_CHEBYSHEV
#define USE_SVD


#include "svd.c"
#include "eigen_ccmath.c"
//extern "C" {
////void svd(double *out_V, double *out_S2, double *in_A, int n);
//
//// d: output size n (singular values)
//// a: input size n*m (m>=n, will get trashed)
//// u: output size m*m
//// v: output size n*n
//int svd(double *d, double *a, double *u, int m, double *v, int n);
//};




// computes x  =  V  ( S^+  U^t  b)
// where S^+ is the diagonal pseudo inverse matrix
// whose diagonal elements are:
//      1/s_i   for all s_i \in S, with |s_i| > 0.0001 and
//       0      oherwise
inline void svd_solve(const double *U, const double *S, const double *V, const double *b, double *const x, const int n) {
    // reset output
    for(int i=0; i<n; i++) x[i] = 0;

    for(int i=0; i<n; i++) {
        if(fabs(S[i]) > 0.0001) {// TODO:  decide weather fixed or relative ratio
        //    if(S[i]/S[0] > 0.01) {// TODO:  decide weather fixed or relative ratio
            double ub = 0;
            for(int j=0; j<n; j++)
                ub += U[i+j*n] * b[j];
            ub/=S[i];
            for(int j=0; j<n; j++)
                x[j] += ub * V[i+j*n];
        }
    }
}

inline void prod_mat_vec(double *const y, const double *A, const double *x, const int n, const int m){
    for (int i=0; i<n;i++){
        y[i] = 0;
        for (int j=0;j<m;j++) y[i] += A[i*m +j] * x[j];
    }
}




struct Model {
    //x(d)   := (A'A)^{-1}  A'd
    //err(x) := x A'A x +  d'd  - 2 A'd x
    //double * parameters_x;   // M order of the model   (one per channel)
    //double * parameters_AA;  // symmetric M^2 matrices (common for all the channels)
    //double * parameters_Ad;  // M vector               (one per channel)
    //double * parameters_dd;  // scalar                 (one per channel)

    int degree;
    int nch;
    int nx, ny;   // scaling parameters for the x and y coordinates (size of the image)
    int parameters_x_stride;
    int parameters_AA_stride;
    int model_size; //sum of all params


    Model(int degree, int nch=1, int nx=500, int ny=500) : degree(degree), nch(nch), nx(nx), ny(ny) {

        switch (degree){
            case 0: // F (constant value)
                parameters_x_stride = 1;
                break;
            case 2: // Ax^2 + By^2 + Cxy + Dx +Ey + F
                parameters_x_stride = 6;
                break;
            case 3: // x^3 + x^2y + xy^2 + y^3 + Ax^2 + By^2 + Cxy + Dx +Ey + F
                parameters_x_stride = 10;
                break;
            case 1: // Dx +Ey + F
            default:
                parameters_x_stride = 3;
                break;
        }
        parameters_AA_stride=(parameters_x_stride+1)*parameters_x_stride/2;

        model_size = parameters_x_stride * nch +      // nch model parameter vectors
                     parameters_AA_stride +           // covariance parameters
                     parameters_x_stride * nch +      // nch vectors Ad
                     nch +                            // nch scalars dd
                     1 +                              // scalar error2
                     1;                               // scalar epoch

    }

    inline int get_model_size() {return model_size;}

//#define get_parameters_x(b,c) (b+c*parameters_x_stride)
    inline double* get_parameters_x(double *const base, int channel = 0) {
        return base + channel * parameters_x_stride;
    }

    inline double* get_parameters_AA(double *const base) {
        return base + parameters_x_stride * nch;
    }

    inline double* get_parameters_Ad(double *const base, int channel = 0) {
        return base + parameters_x_stride * nch + parameters_AA_stride + channel * parameters_x_stride;
    }

    inline double* get_parameters_dd(double *const base, int channel = 0) {
        return base + parameters_x_stride * nch + parameters_AA_stride + parameters_x_stride * nch + channel;
    }

    inline double* get_parameters_err2(double *const base) {
        return base + parameters_x_stride * nch + parameters_AA_stride + parameters_x_stride * nch + nch;
    }

    inline double* get_parameters_epoch(double *const base) {
        return base + parameters_x_stride * nch + parameters_AA_stride + parameters_x_stride * nch + nch + 1;
    }




    inline void normalize_inplace(float &x, float &y, bool undo = false) {
        if(undo) {
            x = x * nx/2.f + nx/2.f;
            y = y * ny/2.f + ny/2.f;
        } else {
            x = (x - nx/2.f) / (nx/2.f);
            y = (y - ny/2.f) / (ny/2.f);
        }
    }
    inline void normalize_inplace(float &x, float &y, float &v, bool undo = false) {
        if(undo) {
            x = x * nx/2.f + nx/ 2.f;
            y = y * ny/2.f + ny/2.f;
//            v = v * 128.f  + 128.f;
        } else {
            x = (x - nx/2.f) / (nx/2.f);
            y = (y - ny/2.f) / (ny/2.f);
//            v = (v - 128.f)  / 128.f;
        }
    }

    double evaluate(double *m, float x, float y, int ch=0) {
        double *px = get_parameters_x(m, ch);

        normalize_inplace(x, y);

#ifdef USE_CHEBYSHEV
        const double vv[10] = {1., y, x, 2*y*y-1, x*y, 2*x*x-1, 4*y*y*y-3*y, 2*y*y*x-x, 2*y*x*x-y, 4*x*x*x-3*x};
#else
        const double vv[10] = {1., y, x, y*y, x*y, x*x, y*y*y, y*y*x, y*x*x, x*x*x};
#endif
        int T = 1;
        if(degree == 0) T = 1;  //case 0 // F
        if(degree == 1) T = 3;  //case 1 // F + Ey + Dx
        if(degree == 2) T = 6;  //case 2 // F + Ey + Dx + Cy^2 + Bxy + Ax^2
        if(degree == 3) T = 10;

        long double r = 0;
        for (int i=0; i < T; i++) r += vv[i] * px[i];

        float rr = (float) r;
        normalize_inplace(x, y, rr, true);
        return rr;
    }


    inline void accumulate_covariance(const int n, const double *const v, double *const w) {
        // fills the row-major upper triangle of the outer product of v with itself
        // and accumulates it to the vector w
        // the size of w must be: (n+1)(n)/2
        // n = 1 -> 1
        // n = 2 -> 3
        // n = 3 -> 6

        for(int i=0, t=0; i<n; i++)
            for(int j=i; j<n; j++, t++)
                w[t] += v[i] * v[j];
    }


    inline void load_covariance_matrix_AA(const int n, const double *v, double *const A) {
        // v must have:  (n+1)(n)/2 elements
        for(int i=0, t=0; i<n; i++)
            for(int j=i; j<n; j++, t++)
                A[i + j*n] = A[j + i*n] = v[t];
    }


    /* loads onlky the first k variables of the covariance matrix */
    inline void load_covariance_matrix_AA_sub(const int n, const int k, const double *v, double *const A) {
        // v must have:  (n+1)(n)/2 elements
        for(int i=0, t=0; i<k; t+=n-i, i++)
            for (int j=i; j<k; j++)
                A[i + j * k] = A[j + i * k] = v[t-i+j];
    }


    void precompute_params(const list<Pixel> &p, double *const m, const Img &u) {
        for (int c = 0; c < nch; c++) {
            int sAA = parameters_AA_stride;
            int sAd = parameters_x_stride;
            double *pAA = get_parameters_AA(m);
            double *pAd = get_parameters_Ad(m,c);
            double *pdd = get_parameters_dd(m,c);

            // zero-out all the parameters
            pdd[0]=0;
            for (int j = 0; j < sAd; ++j) pAd[j] = 0;
            if (c == 0)
                for (int j = 0; j < sAA; ++j) pAA[j] = 0;

            // traverse all the pixels and compute the parameters
            const int Npix = u.nx * u.ny;
            for (list<Pixel>::const_iterator i = p.begin(); i != p.end(); ++i) {
                float x = i->x_pos;
                float y = i->y_pos;
                float v = u[i->idx + Npix * c];
                normalize_inplace(x, y, v);

#ifdef USE_CHEBYSHEV
                const double vv[10] = {1., y, x, 2*y*y-1, x*y, 2*x*x-1, 4*y*y*y-3*y, 2*y*y*x-x, 2*y*x*x-y, 4*x*x*x-3*x};
#else
                const double vv[10] = {1., y, x, y*y, x*y, x*x, y*y*y, y*y*x, y*x*x, x*x*x};
#endif

                const int T = sAd; // degree={0,1,2,3} -> T={1,3,6,10}

                pdd[0] += v * v;
                for (int j = 0; j < T; ++j) pAd[j] += vv[j]*v;
                if (c == 0) accumulate_covariance(T, vv, pAA);
                //   e.g.
                //   1[0]   y[1]     x[2]
                //          yy[3]   xy[4]
                //                  xx[5]
            }
        }

    }

    inline void estimate_model(double *const m, int npix) {
        double *epoch = get_parameters_epoch(m);
        double *model_err2 = get_parameters_err2(m);
        double *pAA = get_parameters_AA(m);
        int L = parameters_x_stride;

        epoch[0] = 1;

        // static arrays needed for SVD and model evaluation
        const int   MAX_MODEL_SIZE = 10;
        double   AA[MAX_MODEL_SIZE*MAX_MODEL_SIZE];
        double  AA2[MAX_MODEL_SIZE*MAX_MODEL_SIZE];
        double svdV[MAX_MODEL_SIZE*MAX_MODEL_SIZE];
        double svdU[MAX_MODEL_SIZE*MAX_MODEL_SIZE];
        double svdS[MAX_MODEL_SIZE];
        double tmpX[MAX_MODEL_SIZE];

        // zero the error
        model_err2[0] = 0;

        // zero the model
        for(int c=0; c<nch; c++) {
            double *x = get_parameters_x(m, c);
            for (int i = 0; i < L; i++) x[0] = 0;
        }

        // subL is the restricted size of the covariance matrix
        // the restriction is determined only based on the number of
        // pixels in the region.
        int subL = L;
        int extra = 5;   // add some extra points

        // FORCE QUADRATIC MODEL
        if (npix < 10+extra) { subL = min(6, L); }
        // FORCE LINEAR MODEL
        if (npix < 6+extra) { subL = min(3, L); }
        // FORCE CONSTANT MODEL
        if (npix < 3+extra) { subL = min(1, L); }


        // SHORTCUT FOR THE AFFINE MODEL:compute model and error
        if (subL == 3) {
            load_covariance_matrix_AA_sub(L, subL, pAA, AA);
            const double a11=AA[0], a12=AA[1], a13=AA[2];
            const double a21=AA[3], a22=AA[4], a23=AA[5];
            const double a31=AA[6], a32=AA[7], a33=AA[8];

            const double detA = a11*( a33* a22- a32* a23)- a21*( a33* a12- a32* a13)+ a31*( a23* a12- a22* a13);

            if (fabs(detA)>1e-9) {
                const double invdetA = 1./detA;

                const double i11 =  ( a33* a22- a32* a23)*invdetA;
                const double i12 = -( a33* a12- a32* a13)*invdetA;
                const double i13 =  ( a23* a12- a22* a13)*invdetA;

                const double i21 = -( a33* a21- a31* a23)*invdetA;
                const double i22 =  ( a33* a11- a31* a13)*invdetA;
                const double i23 = -( a23* a11- a21* a13)*invdetA;

                const double i31 =  ( a32* a21- a31* a22)*invdetA;
                const double i32 = -( a32* a11- a31* a12)*invdetA;
                const double i33 =  ( a22* a11- a21* a12)*invdetA;

                for(int c=0; c<nch; c++) {
                    double *Ad = get_parameters_Ad(m, c);
                    double *dd = get_parameters_dd(m, c);
                    double *x  = get_parameters_x(m, c);

                    // inv(A' * A)  * A' d = v
                    x[0] =  i11* Ad[0]  + i12* Ad[1]  + i13* Ad[2] ;
                    x[1] =  i21* Ad[0]  + i22* Ad[1]  + i23* Ad[2] ;
                    x[2] =  i31* Ad[0]  + i32* Ad[1]  + i33* Ad[2] ;

                    //accumulate the err(x) := x A'A x +  d'd  - 2 A'd x
                    const double t1= a11 * x[0] +  a12 * x[1] +  a13* x[2] ;
                    const double t2= a21 * x[0] +  a22 * x[1] +  a23* x[2] ;
                    const double t3= a31 * x[0] +  a32 * x[1] +  a33* x[2] ;
                    model_err2[0] += (x[0] * t1 + x[1] * t2 + x[2] * t3)
                                     + dd[0] - 2 * (Ad[0] * x[0] + Ad[1] * x[1] + Ad[2] * x[2]);
                }
                return;
            }
            else subL=1;
        }


        // SHORTCUT FOR THE CONSTANT MODEL:compute model and error
        if (subL == 1) {
            for(int c=0; c<nch; c++) {
                double *Ad = get_parameters_Ad(m, c);
                double *dd = get_parameters_dd(m, c);
                double *x  = get_parameters_x(m, c);

                x[0] = Ad[0] / pAA[0];    //x(d)   := (A'A)^{-1}  A'd
                //accumulate the err(x) := x A'A x +  d'd  - 2 A'd x
                model_err2[0] += x[0] * pAA[0] * x[0] + dd[0] - 2 * Ad[0] * x[0];
            }
            return;
        }

        //load A'A
        load_covariance_matrix_AA_sub(L, subL, pAA, AA);
        load_covariance_matrix_AA_sub(L, subL, pAA, AA2);

        //inverse of A'A // destroys AA
#ifdef USE_SVD
        svd(svdS, AA2, svdU, subL, svdV, subL);
#else
        eigen(AA2, svdS, subL);
#endif

        // zero the model error
        model_err2[0] = 0;

        //compute model and error
        for(int c=0; c<nch; c++) {
            double *pAd = get_parameters_Ad(m,c);
            double *pdd = get_parameters_dd(m,c);
            double *px  = get_parameters_x(m,c);

#ifdef USE_SVD
            svd_solve(svdU, svdS, svdV, pAd, px, subL);
#else
            svd_solve(AA2, svdS, AA2, pAd, px, subL);
#endif

            //compute/accumulate the err
            //err(x) := x A'A x +  d'd  - 2 A'd x
            double tmp, tmp2;
            prod_mat_vec(tmpX,  AA, px,   subL, subL);
            prod_mat_vec(&tmp,  px,  tmpX, 1,    subL);
            prod_mat_vec(&tmp2, pAd, px,   1,    subL);
            model_err2[0] += tmp + pdd[0] -2 * tmp2;
        }
    }

    void compute(double *const m, const Img &u, const list<Pixel> &p, const int psize) {
        precompute_params(p, m, u);
        estimate_model(m, psize);
        //estimate_model(m, p.size());
    }

    void compute_by_merging(double *const m1, double *const m2, double *const mm, const Img &u, const list<Pixel> &p1,
                                const list<Pixel> &p2, const int p1size, const int p2size) {
        double *epoch1 = get_parameters_epoch(m1);
        double *epoch2 = get_parameters_epoch(m2);

        // check if the models m1 and m2 are initialized, otherwise initialize them using p1 and p2
        if (epoch1[0] < 0) compute(m1, u, p1, p1size);
        if (epoch2[0] < 0) compute(m2, u, p2, p2size);
  //      if (epoch1[0] < 0 || epoch1[0] > 100 ) compute(p1,m1);
  //      if (epoch2[0] < 0 || epoch2[0] > 100 ) compute(p2,m2);

        // compute the merged params: all the parameters are just added!
        // data contains parameters_AA,Ad,x,dd, error, and epoch
        for(int i=0; i<model_size; i++) //
            mm[i] = m1[i] + m2[i];
//        epoch = epoch1 + epoch2;
        // TODO: if epoch is too big, just force the calls to m1.compute(p1). But make sure that afterwards the priority queue correctly reflects what happend...

        estimate_model(mm, p1size + p2size);
        //estimate_model(mm, p1.size() + p2.size());
    }

};


#endif //_MODEL_H
