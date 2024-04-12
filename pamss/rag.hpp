//
// Created by Gabriele Facciolo on 02/03/16.
//

#ifndef PROJECT_RAG_HPP
#define PROJECT_RAG_HPP

// RAG
// * nodes: vector node
//    - list<int> edge_ids
//    - list<pixels> px
//
// * edges: vector edge
//    - int node1_id, node2_id
//    - float lenght

#include <list>
#include <vector>
#include <stdio.h>
#include "img.hpp"


using namespace std;


// edge element
struct Edge
{
    float lenght;
    int n1, n2;

    int edge_id;

    int model_id;

    Edge()
    {
        n1=n2=-1;
        edge_id=-1;
        model_id=-1;
    }

    Edge(int n1, int n2, float lenght, int edge_id) : n1(n1), n2(n2), lenght(lenght), edge_id(edge_id) // member init list
    {
        model_id=-1;
    }

};


struct Pixel {
    int x_pos;
    int y_pos;
    int idx;    // pixel index
    Pixel(int x, int y, int idx) : x_pos(x), y_pos(y), idx(idx) {} // member init list

};


struct Node
{
    int area;
    list< int > edge_list;
    list< struct Pixel > pixels;
    int _first_seen_by;   // this property is used to accelerate the removal of duplicated edges

    int node_id;

    int model_id;

    Node() {
        node_id=-1;
        area=0;
        _first_seen_by = -1;
        model_id=-1;
    }

};


struct RAG {
    vector<Node > nodes;
    vector<Edge > edges;
    Img imagepixels;

    int nregions;
    int nedges;
};

struct RAG initialize_RAG_from_image(Img &u, float gradient_weight);

int merge_nodes(struct RAG &r, int edge_id, list< int > &removed_edges );


#endif //PROJECT_RAG_HPP


