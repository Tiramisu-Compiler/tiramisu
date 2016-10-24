    function blurxy("blurxy_coli");
    buffer b_input("b_input", 2, {coli::expr(SIZE0),coli::expr(SIZE1)}, p_uint16, NULL, a_input, &blurxy);
    buffer b_blury("b_blury", 2, {coli::expr(SIZE0),coli::expr(SIZE1)}, p_uint16, NULL, a_output, &blurxy);
    expr e_p0 = expr((int32_t) SIZE0 - 1);
    expr e_p1 = expr((int32_t) SIZE1 - 1);
    constant p0("N", e_p0, p_int32, true, NULL, 0, &blurxy);
    constant p1("M", e_p1, p_int32, true, NULL, 0, &blurxy);
    expr e_p2 = (expr((int32_t) SIZE1 - 1)/expr((int32_t) 8))*expr((int32_t) 8);
    constant p2("NM", e_p2, p_int32, true, NULL, 0, &blurxy);


    // Declare the computations c_blurx and c_blury.
    computation c_input("[N]->{c_input[i,j]: 0<=i<N and 0<=j<N}", expr(), false, p_uint16, &blurxy);

    idx i = idx("i");
    idx j = idx("j");

    expr e1 = (c_input(i-1, j) + c_input(i, j) + c_input(i+1, j))/((uint16_t) 3);
    computation c_blurx("[N,M]->{c_blurx[i,j]: 0<i<N-1 and 0<j<M-1}", e1, true, p_uint16, &blurxy);

    expr e2 = (c_blurx(i, j-1) + c_blurx(i, j) + c_blurx(i, j+1))/((uint16_t) 3);
    computation c_blury("[N,M]->{c_blury[i,j]: 0<i<N-1 and 0<j<M-1}", e2, true, p_uint16, &blurxy);

    // Create a memory buffer (2 dimensional).
    buffer b_blurx("b_blurx", 2, {coli::expr(SIZE0),coli::expr(SIZE1)}, p_uint16, NULL, a_temporary, &blurxy);

    // Map the computations to a buffer.
    c_input.set_access("{c_input[i,j]->b_input[i,j]}");
    c_blurx.set_access("{c_blurx[i,j]->b_blurx[i,j]}");
    c_blury.set_access("{c_blury[i,j]->b_blury[i,j]}");

    // Set the arguments to blurxy
    blurxy.set_arguments({&b_input, &b_blury});
