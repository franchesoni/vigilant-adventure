//
// Created by Gabriele Facciolo on 08/03/16.
//
#include <vector>
#include <list>
#include "img.hpp"
#include "imgio.hpp"

#define _DISABLE_TEST
#include "heap.cpp"
#include "rag.hpp"  // RAG, Edge, and Node
#include "model.hpp"
#include "mergetreelog.h"

using namespace std;

// RAG
// * nodes: vector node
//    - edge_id  list<int>
//    - list px
//
// * edges: vector edge
//    - node1_id, node2_id   int
//    - lenght



// the RAG stores the graph structure: nodes and edges.
//     nodes contain lists of edge_ids and lists of pixels
//     edges contain references to the two connected nodes, and also the index of its position in the HEAP
//             the heap position allows to accelerate the merge procedure,
//             because several edges are updated and removed after each merge
//
// the HEAP stores and maintains the priority of the edges as the graph evolves
//     the position of an edge (in the heap) varies as the heap changes
//     since the RAG.edges store the position of the edge in the heap
//     a special procedure is implemented to notify these changes to the as
//     the heap is maintained lambda is only stored in the priority queue


// The Model is just an object instance with a collection of functions that operate on chunks of memory
//     From the pixels contained in a node one can always compute:
//         - a polynomial model fitting the pixels
//         - the residual error between the model and the pixels
//     The heap priority of an edge is actually computed using these properties as:
//         priority = (-residual(n1) - residual(n2) + residual(n1 \cup n2)/ lenght(edge_between_n1_n2)
//     implementing and using these functions yields an exponential complexity
//     instead the models and errors of new nodes are computed by combining the models of the previous nodes
//     this has a constant complexity
//     In the end we need to efficiently maintain all the model data this is done by allocating a large vector
//     and using it as a cache for the models



/// priority queue stuff
struct pqITEM {
    int edge_id;
    double lambda;
    inline pqITEM(int edge_id, double lambda): edge_id(edge_id), lambda(lambda) {};
};

inline bool operator<(pqITEM &a, pqITEM &b){
    if (a.lambda < b.lambda) return true;
    else return false;
}

// this function maitains the vector of references from RAG edges to their corresponding position in the HEAP
inline void HEAP_position_change_callback(void *r, pqITEM &x, int newpos) {
    vector<int>* v =  (vector<int>*) r;
    (*v)[x.edge_id] = newpos;
}



// All the models (nodes and edges) are stored in a single data structure
// with methods to provide simple access to it
struct Models_data {
    int model_size;
    vector<double > _models_data;
    Model model;

    Models_data(Model model, int num_models) : model(model) {
        model_size = model.model_size;
        _models_data = vector<double > (num_models * model_size);
    }

    inline double * get(const int n) { return & _models_data.front() + n * model_size; }

    inline void compute_by_merging(int m1, int m2, int mm, const Img &u, const Node &n1, const Node &n2) {
#if __cplusplus <= 199711L
        model.compute_by_merging(get(m1), get(m2), get(mm), u, n1.pixels, n2.pixels, n1.area, n2.area);
#else
        model.compute_by_merging(get(m1), get(m2), get(mm), u, n1.pixels, n2.pixels, n1.pixels.size(), n2.pixels.size());
#endif
    }
    inline void compute(int m1, const Img &u, const Node &n1) {
#if __cplusplus <= 199711L
        model.compute(get(m1), u, n1.pixels, n1.area);
#else
        model.compute(get(m1), u, n1.pixels, n1.pixels.size());
#endif
    }
    inline double get_err2(int m1) {
        return model.get_parameters_err2(get(m1))[0];
    }
    inline double evaluate(int m1, float x, float y, int ch=0) {
        return model.evaluate(get(m1), x, y, ch);
    }

};



/*
 * Computes the minimum lambda for removing a boundary,
 * which is equivalent to merging two regions
 * Computes: dE = E(r1) + E(r2) - E(r1 U r2)
 *              = err(r1) + err(r2) - err(r1 U r2) + lenght(boundary)
 * where:
 *  E(r) = lenght(r->boundaryes) + err(r)
 *  E(r1 U r2 ) = lenght(r1->boundaryes) + lenght(r2->boundaryes)
 *               -lenght(boundary) + err(r1 U r2) */
inline double merge_gain(Edge &e, RAG &r, Models_data &models) {
    int m1 = r.nodes[e.n1].model_id,
        m2 = r.nodes[e.n2].model_id,
        mm = e.model_id;
    // update the merged model
    models.compute_by_merging(m1, m2, mm, r.imagepixels, r.nodes[e.n1], r.nodes[e.n2]);

    double err_r1     = models.get_err2(m1);
    double err_r2     = models.get_err2(m2);
    double err_merged = models.get_err2(mm);

    // Compute the lambda or gain
    return (-err_r1 - err_r2 + err_merged)/ (double) e.lenght;
}


int debug=1;

//struct Merge_log {
//    int a,b;
//    double lambda;
//    double err;
//    double area;
//    double plenght;
//};



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




inline int partition_merge_regions(struct RAG &r, int eid, myHEAP<pqITEM> &H, vector<int> &edge_position_in_heap, Models_data &models){
    const Edge &te  = r.edges[eid];
    const Node &tn1 = r.nodes[te.n1];
    const Node &tn2 = r.nodes[te.n2];
    float current_lambda = H.array[0].lambda;

    int _debug_heap_update_priority=0;

    // merge the nodes linked to eid
    list<int> removed_edges;
    int merged_node_id = merge_nodes(r, eid, removed_edges);

    // update the model for merged_node_id: swapping its model_id with eid
    std::swap(r.nodes[merged_node_id].model_id, r.edges[eid].model_id);

    // remove duplicated edges from the heap
    for( list<int>::iterator e_it = removed_edges.begin(); e_it != removed_edges.end(); e_it++) {
        r.edges[*e_it].n1 = -1;    // make the edge invalid
        if (*e_it != eid) {        // the eid edge was removed by the pop!
            H.remove(edge_position_in_heap[*e_it]);
            _debug_heap_update_priority++;
        }
    }

//    if (max(tn1.area, tn2.area) > 100 *min(tn1.area , tn2.area))
//                    return _debug_heap_update_priority;

    // update the properties and priorities (lambda) of the new region and its neighbors
    list<int> *el = & r.nodes[merged_node_id].edge_list;
    for(list<int>::iterator e_it = el->begin(); e_it != el->end(); e_it++) {
        Edge &e = r.edges[*e_it];
        int epos = edge_position_in_heap[*e_it];
        // compute the merged model and the gain (lambda) associated to the edge
        double gain = merge_gain(e, r, models);

        if (0) {   // HACK FOR IMAGES WITH VERY LARGE FLAT REGIONS
//                if(r.nodes[e.n1].area > 10 || r.nodes[e.n2].area > 10) gain += 1e-12;
            if(r.nodes[e.n1].area > 100 || r.nodes[e.n2].area > 100) gain += 1e-1;
//            gain += (r.nodes[e.n1].edge_list.size() + r.nodes[e.n1].edge_list.size() ) * 1;
//            if(r.nodes[e.n1].area > 100 || r.nodes[e.n2].area > 100) gain += 1e-1;
//    if (max(r.nodes[e.n1].area , r.nodes[e.n2].area ) > 100 *min(r.nodes[e.n1].area , r.nodes[e.n2].area )) gain += 1e-1;
//                if(r.nodes[e.n1].area > 1000 || r.nodes[e.n2].area > 1000) gain += 1e-6;
//                if(r.nodes[e.n1].area > 10000 || r.nodes[e.n2].area > 10000) gain += 1e-3;
        }

        H.array[epos].lambda = gain;   // update lambda in the heap
        H.fix(epos);                   // notify HEAP that a priority may have changed
        _debug_heap_update_priority++;
        //if(! H._verify_heap()) printf("HEAP CORRUPTED!\n");
    }
    return _debug_heap_update_priority;

}


vector<Merge_log> mumford_shah_segmentation(Img &in, int order, int target_regions, double target_lambda, double gradient_weight = 0, double maxrmse = INFINITY)
{
    // store all the merges
    vector<Merge_log> DSFtreelog;

    // Initialize RAG
    struct RAG r = initialize_RAG_from_image(in, gradient_weight);

    // Setup model
    Model model = Model(order,in.nch,in.nx,in.ny);

    if (debug)  printf("initializing node and edge models, building heap\n");

    // The Model is just a collection of functions that operate on chunks of memory
    // the correspondence between and edge or node with its model is computed using the corresponding indices
    // All the models (nodes and edges) are stored in a single data structure: Models_data
    Models_data models(model, r.nodes.size() + r.edges.size());
    int current_model_idx = 0;

    // compute all the node models
    for(vector<Node>::iterator n = r.nodes.begin(); n != r.nodes.end(); n++) {
        if(n->node_id < 0 ) continue; // invalid nodes are not updated
        n->model_id = current_model_idx++;
        models.compute(n->model_id, r.imagepixels, *n);
    }

    // build HEAP and the index edge_position_in_heap, which maintains the position of every edge in the heap
    vector<int> edge_position_in_heap(r.edges.size());
    myHEAP<pqITEM> H(r.edges.size(), HEAP_position_change_callback, &edge_position_in_heap);

    // compute all the edge model and gains, use it as priority in the heap
    for(vector<Edge>::iterator e = r.edges.begin(); e != r.edges.end(); e++) {
        if(e->n1 >= 0 && e->n2 >=0) { // a valid edge between two valid nodes
            e->model_id = current_model_idx++;           // set the reference to its unique model
            // compute the merged model and the gain (lambda) associated to the edge
            double gain = merge_gain(*e, r, models);
            // make a heap with the edges using the gain (lambda) as priority
            H.push(pqITEM(e - r.edges.begin(), gain));  // {index in the vector, gain}
        }
    }


    if (debug) printf("start merging\n");
    int _debug_heap_update_priority = 0, _debug_merges = 0;

    // start processing the heap
    while (r.nregions > max(target_regions,1) ) {
    //while (!H.isEmpty()) {
        // top of the heap
        int eid = H.top().edge_id;
        const Edge &te  = r.edges[eid];
        const Node &tn1 = r.nodes[te.n1];
        const Node &tn2 = r.nodes[te.n2];
        float current_lambda = H.array[0].lambda;

        // stop conditions for lambda, regions, and error
        if (current_lambda > target_lambda) break;
        if (current_lambda == INFINITY) break;
        if (maxrmse != INFINITY) {      // TODO this should be a function of Model...
//            const float th = maxrmse/256.0;     // this is because the values are scaled inside the model
            const float th = maxrmse;     // this is because the values are scaled inside the model
            // if average error for this region is above the threshold
            if ( models.get_err2(te.model_id) > model.nch * th * th * (tn1.area + tn2.area) ) {
                //printf("%f %f %d %f\n", current_lambda, models.get_err2(te.model_id) , (tn1.area + tn2.area) );
                H.array[0].lambda = INFINITY;   // update lambda in the heap
                H.fix(0);                       // notify HEAP that a priority may have changed
                _debug_heap_update_priority++;
                continue;
            }
        }

        // pop eid from the heap
        H.pop();

        // log in the DSFtreelog the region merge that is about to happen
        assert(tn1.node_id>=0 && tn2.node_id>=0);
        Merge_log mtmp = {tn1.node_id, tn2.node_id, current_lambda, models.get_err2(te.model_id),
                  (double) tn1.area + tn2.area, te.lenght} ;
        DSFtreelog.push_back( mtmp );

        // merge
        _debug_heap_update_priority += partition_merge_regions(r, eid, H, edge_position_in_heap, models);
        _debug_merges ++;

        if(debug && (_debug_merges % 1000) == 0 )
            printf("last lambda + %f   merges: %d heap updates:%d\n", H.array[0].lambda, _debug_merges,
                       _debug_heap_update_priority);

    }

    printf("last lambda + %f   merges: %d heap updates:%d\n", H.array[0].lambda, _debug_merges,
           _debug_heap_update_priority);
    return DSFtreelog;
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



Img relabel_segments_consecutive(const Img &lab) {
   int nx = lab.nx, ny = lab.ny;
   vector<int> t(nx*ny);
   for(int i=0; i<nx*ny; i++) t[i] = -1;

   Img res(nx,ny,1);
   int lcount = 0;

   for(int i=0;i<nx*ny;i++){
      int oldlab = lab[i];
      if( t[oldlab] < 0 ) {
         t[oldlab] = lcount;
         lcount++;
      }
      if( t[oldlab] >=0 ) res[i] = t[oldlab];
      else printf("I give up programming!\n");
   }
   return res;
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

    int nregions = atoi(pick_option(&argc, &argv, (char*) "n", (char*) "2"));
    int order    = atoi(pick_option(&argc, &argv, (char*) "o", (char*) "1"));
    float lambda = atof(pick_option(&argc, &argv, (char*) "l", (char*) "inf"));
    float maxrmse = atof(pick_option(&argc, &argv, (char*) "e", (char*) "inf"));
    char *initial_segmentation_file = pick_option(&argc, &argv, (char*) "i", (char*) "");
    char *mergelogfile = pick_option(&argc, &argv, (char*) "t", (char*) "");
    float gradient_weight = atof(pick_option(&argc, &argv, (char*) "g", (char*) "0"));

    if (argc < 3) {
        fprintf(stderr, "too few parameters\n"
                "   %s [-o model_order(1)] [-l lambda(inf)] [-n regions(2)]"
                " [-e maxrmse(inf)][-i initial_segm] [-t mergelogfile] [-g grad_weight(0)] in out [out_labels]\n", argv[0]);
        return 1;
    }

    char *in_file = argv[1];
    char *out_file = argv[2];
    char *out_labels_file = argc > 3 ? argv[3] : NULL;

    // if in file ends with .npy use cnpy to open
    Img in = strstr(in_file, ".npy") != NULL ? read_npy(in_file) : iio_read_vector_split(in_file);
    int nx = in.nx;
    int ny = in.ny;
    int nch = in.nch;
    // sanitize input
    for(int i=0;i<nx*ny*nch;i++) in[i] = isfinite(in[i]) ? in[i] : 0;

    Img initial_segmentation;
    if(strcmp(initial_segmentation_file,"") != 0)
        initial_segmentation = iio_read_vector_split(initial_segmentation_file);

//    struct RAG r0 = initialize_RAG_from_image(initial_segmentation);
//    printf("%d\n", r0.nregions);
//    Model model00(0,initial_segmentation.nch,nx,ny);
//    mumford_shah_segmentation(r0, model00, 1, 0);
//    printf("%d\n", r0.nregions);
    // initialize_RAG_from_initial_segmentation

    // Call the main function
    vector<Merge_log> t;
    Img segment;
    if(strcmp(mergelogfile,"") != 0) {
        t = mumford_shah_segmentation(in, order, 1, INFINITY, gradient_weight);
        segment = get_segmentation_from_merge_tree_log(t, nx, ny, nregions, lambda);
    } else {
        t = mumford_shah_segmentation(in, order, nregions, lambda, gradient_weight, maxrmse);
        segment = get_segmentation_from_merge_tree_log(t, nx, ny, 1, lambda);
    }


    // Write outputs
    double err2,err2real;
    Img out = apply_model_to_segmentation(segment, in, order, &err2, &err2real);
    iio_write_vector_split(out_file, out);
    printf("err2: %lf err2real %lf\n", err2, err2real);

    if(out_labels_file!=NULL) {
    //    Img tmp = relabel_segments_consecutive(segment);
        Img tmp =  draw_labels_colors(segment);
        iio_write_vector_split(out_labels_file, tmp);
    }

    if(strcmp(mergelogfile,"") != 0)
        writemlogfileb(mergelogfile, t);

    return 0;
}

