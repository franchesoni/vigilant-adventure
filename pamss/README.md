# Piecewise Polynomial Mumford-Shah Segmentation

# Usage

    ./pamss [-o model_order(1)] [-l lambda(inf)] [-n regions(2)] [-e maxrmse(inf)] [-i initial_segm] [-t mergelogfile] in out [out_labels]

    ./mstree [-o model_order(1)] [-l lambda(inf)] [-n regions(2)]      treelog in out [out_labels]

    ./mstree -o 3 -n 2500 -L  Merge_log   ~/Work/datasets/standard_test_images/lena_gray_256.tif /tmp/{a,b}.tif   2>&1 >/dev/null | gnuplot -p -e "set logscale y; plot '<cat' using 1:2 w l"


    order=0
    ./pamss data/kodim12.png -o $order -t llog  -n 100 o.png l.tif
    ./mstree -L log.txt -o $order -n "2 4 8 16 32 64 128 256 512 1024 2048 4096 9162"  llog  data/kodim12.png o.png l.tif 
    gnuplot -p -e "set logscale y; plot 'log.txt' using 1:2 w l"


# implementation NOTES

* lazy heap remove is slower than actually removing the item, pop has to do the work anyway
* 
