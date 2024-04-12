#ifndef BINARY_TREE_HEAP
#define BINARY_TREE_HEAP
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
/* Templated priority queue, using a binary heap 
 * by Gabriele Facciolo, gfacciol@gmail.com
 * based on the C code by Rene Wiermer, rwiermer@googlemail.com
 *
 * The implemementation is based on Heapsort as described in
 * "Introduction to Algorithms" (Cormen, Leiserson, Rivest, Stein)
 */

using namespace std;



// this is a special binary heap.
// whenever a node changes position in the heap,
// the method position_change_callback is called.
// This allows to link other structures that 
// need quick access to the heap structure. 

// the type T MUST have the comparison operator:  bool operator<(T a, T b)
//


template <typename T>
struct myHEAP{
    vector<T> array;
    // these members allow to implement a callback function for
    // maintaining externally the position of each item in the priority queue
    void  *_position_change_callback_data;
    void (*_position_change_callback_func)(void *, T&, int);

    // when the position of an item has changed in the queue the new_index
    // is passed to the callback function
    void position_change_callback(T &item, int new_index) {
        if(_position_change_callback_func)
            _position_change_callback_func(_position_change_callback_data, item, new_index);
    }

    int size(){
        return array.size();
    }

    bool isEmpty(){
        return array.size() == 0;
    }

    void clear() {
        array.clear();
    }

    myHEAP(long reservedsize,
           void (*pos_change_callback_func)(void *, T&, int) = NULL,
           void  *pos_change_callback_data                   = NULL) {
        array.reserve(reservedsize); // the array of items of type T
        _position_change_callback_data = pos_change_callback_data;
        _position_change_callback_func = pos_change_callback_func;
    }


    /* Reestablish heap condition by moving up the element i*/
    void moveUP( int i ) {
        T tmp = array[i];
        /*find the correct place to insert*/
        int parent=(i-1)/2;
        while ((i > 0) && (tmp < array[parent])) {
            array[i]=array[parent];
            position_change_callback(array[i], i);
            i = parent;
            parent=(i-1)/2;
        }
        array[i]=tmp;
        position_change_callback(array[i], i);
    }


    void moveDOWN_shorter_works_but_slower(int i) {
        const int sz = array.size();
        int smallest;
        while (2*i+1 < sz){
            int l = 2 * i + 1; /*left child*/
            int r = 2 * i + 2; /*right child*/

            if (l < sz && array[l] < array[i]) smallest = l;
            else smallest = i;
            if (r < sz && array[r] < array[smallest]) smallest = r;

            position_change_callback(array[smallest], i);
            if (i == smallest) break;
            std::swap(array[smallest],array[i]);
            i=smallest;
        }

    }

    /* Reestablish heap condition by moving down the element i until to a good place*/
    void moveDOWN(int i) {
        const int sz = array.size();
        int smallest = i;
        while (true) {
            /* prepare for the next loop*/
            i = smallest;
            int l = 2*i+1; /*left child*/
            int r = 2*i+2; /*right child*/

            // find the smallest
            if (l < sz && array[l] < array[i]) smallest = l;
            else smallest = i;
            if (r < sz && array[r] < array[smallest]) smallest = r;

            if (smallest == i) break;

            std::swap(array[smallest],array[i]);
            position_change_callback(array[smallest], smallest);
            position_change_callback(array[i], i);
        }
    }

    /* Reestablish heap condition by moving down the element i until to a good place*/
    void moveDOWNx(int i) {
        int l=2*i+1; /*left child*/
        int r=2*i+2; /*right child*/
        int smallest;

        const int sz = array.size();

        if ((l < sz)&&(array[l] < array[i])) smallest = l;
        else smallest = i;
        if ((r < sz)&&(array[r] < array[smallest])) smallest = r;

        while (smallest != i) {
            std::swap(array[smallest],array[i]);
            position_change_callback(array[smallest], smallest);
            position_change_callback(array[i], i);

            /* prepare for the next loop*/
            i = smallest;
            l = 2*i+1; /*left child*/
            r = 2*i+2; /*right child*/

            if ((l < sz) && (array[l] < array[i])) smallest = l;
            else smallest=i;
            if ((r < sz) && (array[r] < array[smallest])) smallest = r;
        }
    }



    /* Add FIELD p to the HEAP h this operation takes O(lg(n))
     * returns 1 if the operation was successfull or 0 if the heap is full*/
    int push(T p) {
        /* the new availeble position is at the end of the heap */
        array.push_back(p);
        int last = array.size()-1;
        position_change_callback(array[last],last);

        /* move the element to its position */
        moveUP(last);
        return 1;
    }

    /* Extract Minimum
     * returns the pointer to ITEM or NULL if the heap is empty*/
    T pop() {
        //if (isEmpty()) return 0; // TODO FIX !!!

        /* extract the min*/
        T min = array[0];
        position_change_callback(min, -1);

        int sz = array.size();
        if(sz > 1) {
            /* overwrite it with the last element */
            array[0] = array[sz - 1];
            position_change_callback(array[0],0);
        }

        /* reduce the size of the heap */
        array.pop_back();

        /* reestabilish propertyes */
        moveDOWN(0);

        return min;
    }

    /* Query Minimum (do not remove it)
     * returns the pointer to ITEM or NULL if the heap is empty*/
    T top() {
        // if (isEmpty()) return 0; // TODO FIX !!!
        /* the min*/
        return array[0];
    }


    /* Refresh element i*/
    // re-establishes the heap ordering after the element at index i has changed its value.
    // Changing the value of the element at index i and then calling Fix is equivalent to,
    // but less expensive than, calling Remove(h, i) followed by a Push of the new value.
    // The complexity is O(log(n))
    void fix(int i) {
        int sz = array.size();
        if (i >= sz) {
            printf("HEAP (fix): there is no element %d the heap is shorter (%d)\n",i,sz);
            return;
        }

        if (i<0) {
            printf("HEAP (fix): sorry the element is not here\n");
            return;
        }

        /* determine if the element goes up or down */
        int parent = (i-1)/2;
        if ((i>0) && (array[i] < array[parent]))
            moveUP(i);
        else
            moveDOWN(i);
    }


    /* Delete element i*/
    void remove(int i) {
        int sz = array.size();
        if (i >= sz) {
            printf("HEAP (remove): there is no element %d the heap is shorter (%d)\n",i,sz);
            return;
        }

        if (i<0) {
            printf("HEAP (remove): sorry the element is not here\n");
            return;
        }

        /* tag the previous element as outside the heap */
        position_change_callback(array[i],-1);

        /* move the last element to this position i to overwrite it */
        if (i != sz-1) {                 // ATTENTION maybe:   i==sz-1
            array[i] = array[sz-1];
            array.pop_back();             // decrement the size
            position_change_callback(array[i],i);

            /* determine if the element goes up or down */
            fix(i);
        } else {
            array.pop_back();
        }
    }



    // check the that the heap condition is verified .. for testing
    bool _verify_heap() {
        int sz = array.size();
        for(int i=0; i<array.size(); i++){
            int l=2*i+1; /*left child*/
            int r=2*i+2; /*right child*/
            if(l<sz && array[l] < array[i]) return false;
            if(r<sz && array[r] < array[i]) return false;
        }
        return true;
    }


    // O(N) construction of the heap (instead of the O(N logN) of using heap.push()
    void heapify() {
        for (int i=array.size()/2; i>=0; i--) {
            moveDOWN(i);
        }
    }

    int push_without_order(T p) {
        array.push_back(p);
        int last = array.size()-1;
        position_change_callback(array[last],last);
        return 1;
    }


};






#ifndef _DISABLE_TEST



/*
 *
 *
 *
 *
 *
 * Usage example - Test code
 *
 *
 *
 *
 *
 * */

struct ITEM {
    float value;
    int label;
    int external_ref;
};
typedef struct ITEM ITEM;

bool operator<(ITEM a, ITEM b){
    if (a.value < b.value) return true;
    else return false;
}


// FANCIER STUFF 
void position_change_callback(void *r, ITEM &x, int newpos) {
    vector<int>* v =  (vector<int>*) r;
    (*v)[x.external_ref] = newpos;
    printf(".");
}




#include <stdio.h>
int main() {
    int number_of_nodes = 20;
    int i;

    struct myHEAP<ITEM> H(number_of_nodes);

    printf("\nAdding nodes to the heap");
    printf("\nLabel\tCost\n");
    for (i=0;i<number_of_nodes;i++) {
        ITEM x = {rand(), i, i};
        printf("%d\t%.2f\n",x.label,x.value);
        H.push(x);
    }
    printf("%lu\n", H.array.size());

    printf("\nupdating nodes\n");
    for (i=0;i<number_of_nodes/2;i++) {
        H.array[i].value = rand();
        H.fix(i);
    }
    printf("%lu\n", H.array.size());

    printf("\nremoving nodes\n");
    for (i=number_of_nodes-10;i>=0;i--) {
        H.remove(i);
    }
    printf("%lu\n", H.array.size());

    printf("\n\nExtracting the nodes from the heap\nLabel\tCost\n");
    while (!H.isEmpty()) {
        ITEM res = H.pop();
        printf("%d\t%.2f\n",res.label,res.value);
    }




    printf("\nTest the callback function\n");
    printf("--------------------------\n");



    vector<int> ext_data(number_of_nodes);
    H.clear();
    H = myHEAP<ITEM>(number_of_nodes, position_change_callback, &ext_data);

    printf("\nAdding nodes to the heap");
    //printf("\nLabel\tCost\n");
    for (i=0;i<number_of_nodes;i++) {
        ITEM x = {rand(), i, i};
        //		printf("%d\t%.2f\n",x.label,x.value);
        H.push(x);
    }
    printf("%lu\n", H.array.size());

    //	printf("\nupdating nodes\n");
    //	for (i=0;i<number_of_nodes/2;i++) {
    //      H.array[i].value = rand();
    //		H.fix(i);
    //	}
    //   printf("%lu\n", H.array.size());

    printf("\nremoving nodes\n");
    for (i=number_of_nodes-10;i>=0;i--) {
        H.remove(i);
    }
    printf("%lu\n", H.array.size());


    printf("\npositions in the heap of all the nodes\n");
    for (i=0;i<number_of_nodes;i++)
        printf("%d\t", ext_data[i]);
    printf("\n");


    printf("\n\nExtracting the nodes from the heap\nLabel\tCost\n");
    while (!H.isEmpty()) {
        ITEM res = H.pop();
        printf("%d\t%.2f\n",res.label,res.value);
    }

    printf("\npositions in the heap of all the nodes\n");
    for (i=0;i<number_of_nodes;i++)
        printf("%d\t", ext_data[i]);
    printf("\n");
}

#endif

#endif
