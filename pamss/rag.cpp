//
// Created by Gabriele Facciolo on 02/03/16.
//

#include "rag.hpp"
#include <cmath>


// RAG
// * nodes: vector node
//    - edge_id  list<int>
//    - list px
//
// * edges: vector edge
//    - node1_id, node2_id   int
//    - lenght


using namespace std;

struct RAG initialize_RAG_from_image(Img &u, float gradient_weight) {
    int nx = u.nx, ny = u.ny, nch = u.nch;
    int N = nx*ny;

    struct RAG r;
    r.nodes = vector<Node >(N); // all the nodes must be ready
    r.edges = vector<Edge >();  // edges can be initialized with push_back
    r.nregions = N;

    // store a copy of the image to query the pixel values
    r.imagepixels = u;

    // initialize graph
    for (int y=0; y<ny; y++)
        for (int x=0; x<nx; x++) {
            int idx = x + y*nx;

            // * node contain pixel referenes: the actual value of the pixel is stored in imagepixels
            // initialize node
            r.nodes[idx].pixels.push_front( Pixel(x,y,idx) );
            r.nodes[idx].area                 = 1;
            r.nodes[idx].node_id              = idx;
            r.nodes[idx]._first_seen_by       = -1;   // needed by the merge operation

            // * edge reference to nodes
            // * node reference to edges
            if (x<nx-1) {
                int idx_right = (x+1) + y*nx;
                int cur_edge_id = r.edges.size();
                float edge_lenght = 0;
                for (int q=0; q<nch; q++) edge_lenght += fabs(u[idx+nx*ny*q] - u[idx_right+nx*ny*q]);
                edge_lenght = 1.0 / (1.0 + gradient_weight*(edge_lenght/nch));
                r.edges.push_back(Edge(idx, idx_right, edge_lenght, cur_edge_id));// TODO edge strenght may depend on the pixel values

                r.nodes[idx      ].edge_list.push_front(cur_edge_id );
                r.nodes[idx_right].edge_list.push_front(cur_edge_id );
            }

            if (y<ny-1) {
                int idx_down  = x + (y+1)*nx;
                int cur_edge_id = r.edges.size();
                float edge_lenght = 0;
                for (int q=0; q<nch; q++) edge_lenght += fabs(u[idx+nx*ny*q] - u[idx_down+nx*ny*q]);
                edge_lenght = 1.0 / (1.0 + gradient_weight*(edge_lenght/nch));
                r.edges.push_back(Edge(idx, idx_down, edge_lenght, cur_edge_id));

                r.nodes[idx      ].edge_list.push_front(cur_edge_id );
                r.nodes[idx_down ].edge_list.push_front(cur_edge_id );
            }
        }

    return r;
}

inline int edge_get_other_node_id(Edge &e, int n_id) {
    if( e.n1 == n_id ) return e.n2;
    else if( e.n2 == n_id ) return e.n1;
    printf("ERR: was waiting for %d or %d to be %d, didn't happend\n", e.n1, e.n2, n_id) ;
    return -10000000; // this should never happend
}

// merge n2 into n1 or viceversa
// removes the edge_id, and puts all duplicate edges in removed_edges
// TODO: maybe edges.n1&n2 should be ordered to facilitate the comparison
// returns the id of the merged node
int merge_nodes(struct RAG &r, int edge_id, list< int > &removed_edges) {
    int n1_id       =   r.edges[edge_id].n1;
    int n2_id       =   r.edges[edge_id].n2;
    struct Node *n1 = & r.nodes[n1_id];
    struct Node *n2 = & r.nodes[n2_id];
    list<int> *e1   = & r.nodes[n1_id].edge_list;
    list<int> *e2   = & r.nodes[n2_id].edge_list;

    r.nregions--;

    // TODO: the size property can be expensive on lists!
    // reverse n1 & n2  so that n2 has less edges to process
//    if(n1->area < n2->area) {
//        swap(n1_id,n2_id); swap(n1,n2); swap(e1,e2);
//    }

    // move pixels
    n1->pixels.splice(n1->pixels.begin(), n2->pixels);

    // merge area
    n1->area += n2->area;


    // PROCEDURE:
    // ------------
    // traverse n2.edges (e2)
    //    touch all the neighbors of n2
    // set n2 as touched
    // traverse n1.edges (e1) // i.e. the neighbors of n1
    //    if a neighbor of n1 is touched:
    //       remove edge storing it in the edges to remove list
    //       remove the reciprocal edge (not storing it)
    //
    // traverse e2
    //    reset touched neighbors
    //    update n2 id (it should be n1)
    // splice e1 and e2 -> e1


    // traverse n2.edges (e2)
    //    touch all the neighbors of n2, store the id of the edge that allowed to find nX first: _first_seen_by (property)
    list<int>::iterator e_it=e2->begin();
    while ( e_it != e2->end() ) {
        int nX = edge_get_other_node_id(r.edges[*e_it], n2_id);
        r.nodes[nX]._first_seen_by = *e_it;

//        if(nX==n1_id) {
//            list<int>::iterator x = e_it++;
//            removed_edges.splice(removed_edges.begin(), *e2, x);
//            continue;
//        }

        ++e_it;
    }
    // set n2 as touched the edge is the same
    n2->_first_seen_by = n1->_first_seen_by;

    // traverse n1.edges (e1) to remove all the duplicate edges
    // there are two types of duplicates:
    //    1) the link n1-n2 and n2-n1
    //    2) the links n1-nX (and its reciprocal) where nX has
    //       already been reached from n2 in the previous step
    // hence all the edges e_it detected as duplicates can be treated as follows:
    //    - accumulate the lenght of e_it on the eX edge (which will stay in the case 2)
    //    - remove e_it from the edge list of nX (not n1)
    //    - move e_it in n1.edge_list to the list of removed_edges
    e_it=e1->begin();
    while ( e_it != e1->end() ) {
        int nX = edge_get_other_node_id(r.edges[*e_it], n1_id);
        //    if neighbors of n1 are touched:
        //       remove edge storing it in the edges to remove list
        //       remove the reciprocal edge (not storing it)

//        if(nX==n2_id) {
//            list<int>::iterator x = e_it++;
//            removed_edges.splice(removed_edges.begin(), *e1, x);
//            continue;
//        }

        if (r.nodes[nX]._first_seen_by >= 0){
            int eX = r.nodes[nX]._first_seen_by;
            r.edges[eX].lenght += r.edges[*e_it].lenght;
            r.nodes[nX].edge_list.remove(*e_it);       // remove the reciprocal edge from the other node's list
            list<int>::iterator x = e_it++;
            removed_edges.splice(removed_edges.begin(), *e1, x);

        } else {
            ++e_it;
        }
    }

    // reset the _first_seen_by label in e2
    // visit edges of n2 to update the neighbor information: n2_id should disappear
    n1->_first_seen_by = -1;
    for(list<int>::iterator e_it=e2->begin(); e_it != e2->end(); e_it++ ) {
        int nX = edge_get_other_node_id(r.edges[*e_it], n2_id);
        r.nodes[nX]._first_seen_by = -1;
        if(      r.edges[*e_it].n1 == n2_id ) r.edges[*e_it].n1 = n1_id;
        else if( r.edges[*e_it].n2 == n2_id ) r.edges[*e_it].n2 = n1_id;
        else printf("this should not happend\n");
    }

    e1->splice(e1->end(), *e2);

    n2->node_id=-1; // make the node n2 invalid

    return n1_id;
}
















