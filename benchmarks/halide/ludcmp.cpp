    // Compute the inverse of mul1 using LU decomposition.
    // We use the following reference implementation (lines 95 to 126)
    // https://github.com/Meinersbur/polybench/blob/master/polybench-code/linear-algebra/solvers/ludcmp/ludcmp.c
    //	    1)- Compute the LU decomposition of mul1: LU = mul1
    //	    2)- Use LU to compute X, the inverse of mul1 by solving the following
    //	    system:
    //		    LU*X=I
    //	    where I is the identity matrix.
    var i1("i1", 0, 4*w);
    var j1("j1", 0, i1);
    var l2("l2", 0, j1);

    // LU decomposition of A
    computation        w1("w1",        {k, i1, j1},     mul1(k, i1, j1));
    computation w1_update("w1_update", {k, i1, j1, l2},   w1(k, i1, j1) - mul1(k, i1, l2)*mul1(k, l2, j1));
    computation      temp("temp",      {k, i1, j1},       w1(k, i1, j1)/mul1(k, j1, j1));

    var j2("j2", i1, 4*w);
    var l3("l3",  0,  i1);

    computation        w2("w2",        {k, i1, j2},     temp(k, i1, j2));
    computation w2_update("w2_update", {k, i1, j2, l3}, w2(k, i1, j2) - temp(k, i1, l3)*temp(k, l3, j2));
    computation        LU("LU",        {k, i1, j2},     w2(k, i1, j2));

    // Finding the inverse of A.
    // The inverse will be stored in X.
    var r("r", 0, 4*w);
    var r2("r2", r, r+1);
    computation     Y("Y", {k, r, i1}, p_uint8);
    computation     bp("bp", {k, r, i1}, expr((uint8_t) 0));
    computation     bp_update("bp_update", {k, r, r2}, expr((uint8_t) 1));
    computation     w3("w3", {k, r, i1}, bp(k, r, i1));
    computation     w3_update("w3_update", {k, r, i1, j1}, w3(k, r, i1) - LU(k, i1, j1)*Y(k, r, j1));
    Y.set_expression(w3(k, r, i1));

    // TODO: support reverse order loops.
    var j3("j3", i1+1, 4*w);
    computation     X("X", {k, r, i1}, p_uint8);
    computation     w4("w4", {k, r, i1}, Y(k, r, 4-w-1-i1));
    computation     w4_update("w4_update", {k, r, i1, j3}, w4(k, r, i1) - LU(k, 4-w-1-i1, j3)*X(k, r, j3));
    X.set_expression(w4(k, r, i1)/LU(k, 4-w-1-i1, 4-w-1-i1));

