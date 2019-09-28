#include <tiramisu/tiramisu.h>

using namespace tiramisu;

int main(int argc, char **argv){
    tiramisu::init("function10243_schedule_42");

    constant n_filters("n_filters", 3), im_height("im_height", 1024), im_width("im_width", 1024), filter_height("filter_height", 3), filter_width("filter_width", 3);

    var c("c", 0, n_filters), im_h("im_h", 0, im_height), im_w("im_w", 0, im_width), f_h("f_h", 0, filter_height), f_w("f_w", 0, filter_width), h("h", 0, im_height - 3 + 1), w("w", 0, im_width - 3 + 1);

    input image("image", {im_h, im_w}, p_int32);

    input filter("filter", {c, f_h, f_w}, p_int32);

    computation convolved("convolved", {c, h, w}, p_int32);
    convolved.set_expression(image(h, w) * filter(c, h, w) + image(h + 0, w + 1) * filter(c, h + 0, w + 1) + image(h + 0, w + 2) * filter(c, h + 0, w + 2) + image(h + 1, w + 0) * filter(c, h + 1, w + 0) + image(h + 1, w + 1) * filter(c, h + 1, w + 1) + image(h + 1, w + 2) * filter(c, h + 1, w + 2) + image(h + 2, w + 0) * filter(c, h + 2, w + 0) + image(h + 2, w + 1) * filter(c, h + 2, w + 1) + image(h + 2, w + 2) * filter(c, h + 2, w + 2));
    
    convolved.interchange(c, h);
    convolved.unroll(w, 16);
    convolved.parallelize(h);
    
    buffer image_buff("image_buff", {1024, 1024}, p_int32, a_input);
    buffer filter_buff("filter_buff", {3, 3, 3}, p_int32, a_input);
    buffer convolved_buff("convolved_buff", {3, 1022, 1022}, p_int32, a_output);
    
    image.store_in(&image_buff);
    filter.store_in(&filter_buff);
    convolved.store_in(&convolved_buff);

    tiramisu::codegen({&image_buff, &filter_buff, &convolved_buff}, "../data/programs/function10243/function10243_schedule_42/function10243_schedule_42.o");

    return 0;
}