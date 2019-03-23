#include "tiramisu/tiramisu.h"
#include "wrapper_warp_affine_dist.h"

using namespace tiramisu;


#define tfloor(val) expr(o_floor, val)
#define icast(val) expr(o_cast, p_int32, val)
#define fcast(val) expr(o_cast, p_float32, val)
#define mixf(x, y, a) (fcast(x) * (expr((float) 1) - fcast(a)) + fcast(y) * fcast(a))
#define S(s) std::to_string(s)

int main(int argc, char* argv[]) {
  
  // Set default tiramisu options.
  global::set_default_tiramisu_options();
  
  tiramisu::function affine_dist("warp_affine_dist");
  int ROWS_PER_NODE = NROWS / NNODES;
  
  computation input("{input[y,x]: 0<=y<" + S(NROWS) + " and 0<=x<" + S(NCOLS) + "}", expr(), false, 
		    tiramisu::p_uint8, &affine_dist);

  constant nodes("nodes", expr(NNODES), p_int32, true, NULL, 0, &affine_dist);
  
  var x("x"), y("y");
  expr a00 = expr((float) 0.1);
  expr a01 = expr((float) 0.1);
  expr a10 = expr((float) 0.1);
  expr a11 = expr((float) 0.1);
  expr b00 = expr((float) 0.1);
  expr b10 = expr((float) 0.1);

  expr o_r = a11 * fcast(y) + a10 * fcast(x) + b00;
  expr o_c = a01 * fcast(y) + a00 * fcast(x) + b10;
  expr r = o_r - tfloor(o_r);
  expr c = o_c - tfloor(o_c);
  
  expr coord_00_r = icast(tfloor(o_r));
  expr coord_00_c = icast(tfloor(o_c));
  expr coord_01_r = coord_00_r;
  expr coord_01_c = coord_00_c + 1;
  expr coord_10_r = coord_00_r + 1;
  expr coord_10_c = coord_00_c;
  expr coord_11_r = coord_00_r + 1;
  expr coord_11_c = coord_00_c + 1;

  coord_00_r = clamp(coord_00_r, 0, NROWS-1);
  coord_00_c = clamp(coord_00_c, 0, NCOLS-1);
  coord_01_r = clamp(coord_01_r, 0, NROWS-1);
  coord_01_c = clamp(coord_01_c, 0, NCOLS-1);
  coord_10_r = clamp(coord_10_r, 0, NROWS-1);
  coord_10_c = clamp(coord_10_c, 0, NCOLS-1);
  coord_11_r = clamp(coord_11_r, 0, NROWS-1);
  coord_11_c = clamp(coord_11_c, 0, NCOLS-1);

  expr A00 = input(coord_00_r, coord_00_c);
  expr A10 = input(coord_10_r, coord_10_c);
  expr A01 = input(coord_01_r, coord_01_c);
  expr A11 = input(coord_11_r, coord_11_c);

  expr e = fcast(mixf(mixf(A00, A10, r), mixf(A01, A11, r), c));

  computation warp("{warp[y,x]: 0<=y<" + S(NROWS) + " and 0<=x<" + S(NCOLS) + "}", e, true, 
		   tiramisu::p_float32, &affine_dist);

  var y1("y1"), y2("y2"), q("q"), z("z");

  warp.split(y, ROWS_PER_NODE, y1, y2);

  // b/c I don't know the data to send, just send all the data to all the nodes
  /*  xfer exchange = 
    computation::create_xfer("[nodes,rank]->{exchange_s[z,y,x]: 0<=z<nodes and z!=rank and (rank*" + S(ROWS_PER_NODE) 
			     + ")<=y<(rank*" + S(ROWS_PER_NODE) + "+" + S(ROWS_PER_NODE) + ") and 0<=x<" + S(NCOLS) + "}", 
			     "[nodes,rank]->{exchange_r[z,y,x]: 0<=z<nodes and z!=rank and (z*" + 
			     S(ROWS_PER_NODE) + ")<=y<(z*" + S(ROWS_PER_NODE) + "+" + S(ROWS_PER_NODE) + ") and 0<=x<" + S(NCOLS) + "}",
			     z, z,
			     xfer_prop(p_uint8, {ASYNC, BLOCK, MPI}), 
			     xfer_prop(p_uint8, {SYNC, BLOCK, MPI}), input(y,x), &affine_dist);
  */
  // These send/recv pairs actually work together, i.e. some of the sends in exchange_back may be received by the recvs in exchange_fwd
  xfer exchange_back = 
    computation::create_xfer("[nodes,rank]->{exchange_back_s[q,z,y,x]: 0<=q<nodes and 0<=z<rank and 0<=y<" + S(ROWS_PER_NODE) + " and 0<=x<" + S(NCOLS) + "}", 
			     "[nodes,rank]->{exchange_back_r[q,z,y,x]: 0<=q<nodes and 0<=z<rank and 0<=y<" + S(ROWS_PER_NODE) + " and 0<=x<" + S(NCOLS) + "}",
			     z, z,
			     xfer_prop(p_uint8, {ASYNC, BLOCK, MPI}), 
			     xfer_prop(p_uint8, {SYNC, BLOCK, MPI}), input(y+q*ROWS_PER_NODE,x), &affine_dist);
  xfer exchange_fwd = 
    computation::create_xfer("[nodes,rank]->{exchange_fwd_s[q,z,y,x]: 0<=q<nodes and rank+1<=z<nodes and 0<=y<" + S(ROWS_PER_NODE) + " and 0<=x<" + S(NCOLS) + "}", 
			     "[nodes,rank]->{exchange_fwd_r[q,z,y,x]: 0<=q<nodes and rank+1<=z<nodes and 0<=y<" + S(ROWS_PER_NODE) + " and 0<=x<" + S(NCOLS) + "}",
			     z, z,
			     xfer_prop(p_uint8, {ASYNC, BLOCK, MPI}), 
			     xfer_prop(p_uint8, {SYNC, BLOCK, MPI}), input(y+q*ROWS_PER_NODE,x), &affine_dist);
			     
  
  warp.tag_distribute_level(y1);
  exchange_back.s->tag_distribute_level(q);
  exchange_back.r->tag_distribute_level(q);
  exchange_fwd.s->tag_distribute_level(q);
  exchange_fwd.r->tag_distribute_level(q);
  
  exchange_back.s->collapse_many({collapse_group(3, 0, -1, NCOLS)});//, collapse_group(2, 0, -1, ROWS_PER_NODE)});
  exchange_back.r->collapse_many({collapse_group(3, 0, -1, NCOLS)});//, collapse_group(2, 0, -1, ROWS_PER_NODE)});
  exchange_fwd.s->collapse_many({collapse_group(3, 0, -1, NCOLS)});//, collapse_group(2, 0, -1, ROWS_PER_NODE)});
  exchange_fwd.r->collapse_many({collapse_group(3, 0, -1, NCOLS)});//, collapse_group(2, 0, -1, ROWS_PER_NODE)});
  
  exchange_back.s->before(*exchange_fwd.s, computation::root);
  exchange_fwd.s->before(*exchange_back.r, computation::root);
  exchange_back.r->before(*exchange_fwd.r, computation::root);
  exchange_fwd.r->before(warp, computation::root);

  buffer b_input("b_input", {NROWS, NCOLS}, p_uint8, a_input, &affine_dist);
  buffer b_warp("b_warp", {NROWS, NCOLS}, p_float32, a_output, &affine_dist);
  
  input.set_access("{input[y,x]->b_input[y,x]}");
  warp.set_access("[rank]->{warp[y,x]->b_warp[t,x]: t=(y-(" + S(ROWS_PER_NODE) + "*rank))}");
  exchange_back.r->set_access("{exchange_back_r[q,z,y,x]->b_input[y+z*" + S(ROWS_PER_NODE) + ",x]}");
  exchange_fwd.r->set_access("{exchange_fwd_r[q,z,y,x]->b_input[y+z*" + S(ROWS_PER_NODE) + ",x]}");

  affine_dist.codegen({&b_input, &b_warp}, "build/generated_fct_warp_affine_dist.o");
  affine_dist.dump_halide_stmt();
  
  return 0;
}
