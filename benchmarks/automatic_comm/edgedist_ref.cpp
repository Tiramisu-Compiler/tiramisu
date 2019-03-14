#include <tiramisu/tiramisu.h>
#include "wrapper_edgedist.h"

using namespace tiramisu;

int main(int argc, char* argv[])
{
    tiramisu::init("edgedist_ref");

    constant COLS("COLS",_COLS);
    constant ROWS("ROWS",_ROWS);

    function* edge = global::get_implicit_function();
    edge->add_context_constraints("[ROWS]->{:ROWS = "+std::to_string(_ROWS)+"}");

    var ir("ir", 0, ROWS-2), jr("jr", 0, COLS-2), c("c", 0, 3), ii("ii"), jj("jj"), kk("kk"), p("p"),q("q"), i1("i1"), i0("i0"), jn("jn", 0, COLS);
    var in("in", 0, ROWS);
    var iout("iout", 0, ROWS-4), jout("jout", 0, COLS-4);
    var s("s"); var r("r");

    input Img("Img", {in, jn, c}, p_int32);

    computation R("R", {ir, jr, c}, (Img(ir, jr, c) + Img(ir, jr+1, c) + Img(ir, jr+2, c) +
				   Img(ir+1, jr, c) + Img(ir+1, jr+2, c)+ Img(ir+2, jr, c) + Img(ir+2, jr+1, c)
                   + Img(ir+2, jr+2, c))/((int32_t) 8));

    computation Out("Out", {iout, jout, c}, (R(iout+1, jout+1, c) - R(iout+2, jout, c))
        + (R(iout+2, jout+1, c) - R(iout+1, jout, c)));

    R.before(Out, computation::root);

    Img.split(in,_ROWS/NODES,i0,i1);
    Out.split(iout,_ROWS/NODES,i0,i1);
    R.split(ir,_ROWS/NODES,i0,i1);

    Img.tag_distribute_level(i0);
    Out.tag_distribute_level(i0);
    R.tag_distribute_level(i0);

    Img.drop_rank_iter(i0);
    Out.drop_rank_iter(i0);
    R.drop_rank_iter(i0);

    xfer border_comm_R = computation::create_xfer(
    "[COLS]->{border_comm_R_send[s,ii,jj,kk]: 1<=s<10 and 0<=ii<2 and 0<=jj<COLS and 0<=kk<3}",
    "[COLS]->{border_comm_R_recv[r,ii,jj,kk]: 0<=r<9 and 0<=ii<2 and 0<=jj<COLS and 0<=kk<3}",
    s-1,
    r+1,
    xfer_prop(p_int32, {MPI, BLOCK, ASYNC}), xfer_prop(p_int32, {MPI, BLOCK, ASYNC}),
    Img(ii, jj, kk), edge);

    border_comm_R.s->tag_distribute_level(s);
    border_comm_R.r->tag_distribute_level(r);

    border_comm_R.s->before(*border_comm_R.r, computation::root);
    border_comm_R.r->before(R, computation::root);
    R.before(Out, computation::root);

    buffer b_Img("b_Img", {_ROWS/NODES + 2, _COLS, 3}, p_int32, a_input);
    buffer   b_R("b_R",   {_ROWS/NODES + 2, _COLS, 3}, p_int32, a_output);

    Img.store_in(&b_Img);
    R.store_in(&b_R);
    Out.store_in(&b_Img);

    border_comm_R.r->set_access("{border_comm_R_recv[r,ii,jj,kk]->b_Img[" + std::to_string(_ROWS/10) + "+ii,jj,kk]}");

    tiramisu::codegen({&b_Img, &b_R}, "build/generated_fct_edgedist_ref.o");

  return 0;
}
