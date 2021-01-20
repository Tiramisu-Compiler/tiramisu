#include <tiramisu/expr.h>
#include <tiramisu/core.h>

namespace tiramisu
{

tiramisu::expr& tiramisu::expr::operator=(tiramisu::expr const & e)
{
    this->_operator = e._operator;
    this->op = e.op;
    this->access_vector = e.access_vector;
    this->argument_vector = e.argument_vector;
    this->defined = e.defined;
    this->name = e.name;
    this->dtype = e.dtype;
    this->etype = e.etype;

    // Copy the integer value
    if (e.get_expr_type() == tiramisu::e_val)
    {
        if (e.get_data_type() == tiramisu::p_uint8)
        {
            this->uint8_value = e.get_uint8_value();
        }
        else if (e.get_data_type() == tiramisu::p_int8)
        {
            this->int8_value = e.get_int8_value();
        }
        else if (e.get_data_type() == tiramisu::p_uint16)
        {
            this->uint16_value = e.get_uint16_value();
        }
        else if (e.get_data_type() == tiramisu::p_int16)
        {
            this->int16_value = e.get_int16_value();
        }
        else if (e.get_data_type() == tiramisu::p_uint32)
        {
            this->uint32_value = e.get_uint32_value();
        }
        else if (e.get_data_type() == tiramisu::p_int32)
        {
            this->int32_value = e.get_int32_value();
        }
        else if (e.get_data_type() == tiramisu::p_uint64)
        {
            this->uint64_value = e.get_uint64_value();
        }
        else if (e.get_data_type() == tiramisu::p_int64)
        {
            this->int64_value = e.get_int64_value();
        }
        else if (e.get_data_type() == tiramisu::p_float32)
        {
            this->float32_value = e.get_float32_value();
        }
        else if (e.get_data_type() == tiramisu::p_float64)
        {
            this->float64_value = e.get_float64_value();
        }
    }
    return *this;
}

tiramisu::expr tiramisu::expr::substitute(std::vector<std::pair<var, expr>> substitutions) const
{
    for (auto &substitution: substitutions)
        if (this->is_equal(substitution.first))
            return substitution.second;


    return apply_to_operands([&substitutions](const expr& e){
        return e.substitute(substitutions);
    });
}

tiramisu::expr tiramisu::expr::substitute_access(std::string original, std::string substitute) const {
    expr && result = this->apply_to_operands([&original, &substitute](const expr& e){
        return e.substitute_access(original, substitute);
    });
    if (result.get_op_type() == o_access && result.name == original)
    {
        result.name = substitute;
    }
    return result;
}

tiramisu::var::var(std::string name, bool save)
{
    assert(!name.empty());

    auto declared = var::declared_vars.find(name);

    if (declared != var::declared_vars.end())
    {
        *this = declared->second;
    }
    else
    {
        this->name = name;
        this->etype = tiramisu::e_var;
        this->dtype = global::get_loop_iterator_data_type();
        this->defined = true;
        if (save)
        {
            var::declared_vars.insert(std::make_pair(name, *this));
            DEBUG(3, std::cout << "Saved variable " << this->name << " of type " << str_from_tiramisu_type_primitive(this->dtype));
        }
    }
}

tiramisu::var::var(tiramisu::primitive_t type, std::string name, bool save)
{
    assert(!name.empty());

    auto declared = var::declared_vars.find(name);

    if (declared != var::declared_vars.end())
    {
        assert(declared->second.dtype == type);
        *this = declared->second;
    }
    else
    {
        this->name = name;
        this->etype = tiramisu::e_var;
        this->dtype = type;
        this->defined = true;
        if (save)
        {
            var::declared_vars.insert(std::make_pair(name, *this));
            DEBUG(10, std::cout << "Saved variable " << this->name << " of type " << str_from_tiramisu_type_primitive(this->dtype));
        }
    }
}

tiramisu::expr tiramisu::expr::copy() const
{
    return (*this);
}

//std::unordered_map<std::string, var> tiramisu::var::declared_vars;

expr cast(primitive_t tT, const expr & e) {
    if (e.get_data_type() == tT)
        return e;
    return expr{o_cast, tT, e};
}

expr tiramisu::expr::operator+(tiramisu::expr other) const {
    return tiramisu::expr{o_add, *this, other};
}

expr tiramisu::expr::operator-(tiramisu::expr other) const {
    return tiramisu::expr{o_sub, *this, other};
}

expr tiramisu::expr::operator*(tiramisu::expr other) const {
    return tiramisu::expr{o_mul, *this, other};
}

expr tiramisu::expr::operator/(tiramisu::expr other) const {
    return tiramisu::expr{o_div, *this, other};
}

expr tiramisu::expr::operator%(tiramisu::expr other) const {
    return tiramisu::expr{o_mod, *this, other};
}

expr tiramisu::expr::operator>>(tiramisu::expr other) const {
    return tiramisu::expr{o_right_shift, *this, other};
}

expr tiramisu::expr::operator<<(tiramisu::expr other) const {
    return tiramisu::expr{o_left_shift, *this, other};
}

expr memcpy(const buffer &from, const buffer &to) {
    return expr(o_memcpy, var(p_void_ptr, from.get_name()), var(p_void_ptr, to.get_name()));
}

expr allocate(const buffer &b)
{
    return expr{o_allocate, b.get_name()};
}

expr cublas_gemm(const buffer &A, const buffer &B, buffer &C,
                 expr M, expr N, expr K,
                 expr alpha, expr beta,
                 expr ldA, expr ldB, expr ldC,
                 expr offsetA, expr offsetB, expr offsetC,
                 expr transposeA, expr transposeB)
{
    if (A.get_location() != cuda_ast::memory_location::global ||
        B.get_location() != cuda_ast::memory_location::global ||
        C.get_location() != cuda_ast::memory_location::global) {
        ERROR("Buffers must be on GPU global memory", true);
    }
    std::string fname;
    expr alpha_expr;
    expr beta_expr;
    if (A.get_elements_type() == p_float32 &&
        B.get_elements_type() == p_float32 &&
        C.get_elements_type() == p_float32) {
        fname = "tiramisu_cublas_sgemm";
        alpha_expr = cast(p_float32, alpha);
        beta_expr = cast(p_float32, beta);
    } else if (A.get_elements_type() == p_float64 &&
               B.get_elements_type() == p_float64 &&
               C.get_elements_type() == p_float64) {
        fname = "tiramisu_cublas_dgemm";
        alpha_expr = cast(p_float64, alpha);
        beta_expr = cast(p_float64, beta);
    } else {
        ERROR("All input buffers should be of same type and either p_float32 or p_float64", true);
    }
    return expr(o_call, fname,
            {
                var(p_void_ptr, A.get_name()),
                var(p_void_ptr, B.get_name()),
                var(p_void_ptr, C.get_name()),
                cast(p_uint64, M), cast(p_uint64, N), cast(p_uint64, K),
                alpha_expr, beta_expr,
                cast(p_uint64, ldA), cast(p_uint64, ldB), cast(p_uint64, ldC),
                cast(p_uint64, offsetA), cast(p_uint64, offsetB), cast(p_uint64, offsetC),
                cast(p_boolean, transposeA), cast(p_boolean, transposeB)
            },
            tiramisu::p_uint8);
}

expr cuda_stream_synchronize()
{
    return expr(o_call, "tiramisu_cuda_stream_synchronize", {int32_t(0)}, tiramisu::p_int32);
}

expr cblas_gemm(const buffer &A, const buffer &B, buffer &C,
                expr M, expr N, expr K,
                expr alpha, expr beta,
                expr ldA, expr ldB, expr ldC,
                expr offsetA, expr offsetB, expr offsetC,
                expr transposeA, expr transposeB)
{
    std::string fname;
    expr alpha_expr;
    expr beta_expr;
    if (A.get_elements_type() == p_float32 &&
        B.get_elements_type() == p_float32 &&
        C.get_elements_type() == p_float32) {
        fname = "tiramisu_cblas_sgemm";
        alpha_expr = cast(p_float32, alpha);
        beta_expr = cast(p_float32, beta);
    } else if (A.get_elements_type() == p_float64 &&
               B.get_elements_type() == p_float64 &&
               C.get_elements_type() == p_float64) {
        fname = "tiramisu_cblas_dgemm";
        alpha_expr = cast(p_float64, alpha);
        beta_expr = cast(p_float64, beta);
    } else {
        ERROR("All input buffers should be of same type and either p_float32 or p_float64", true);
    }
    return expr(o_call, fname,
            {
                var(p_void_ptr, A.get_name()),
                var(p_void_ptr, B.get_name()),
                var(p_void_ptr, C.get_name()),
                cast(p_int32, M), cast(p_int32, N), cast(p_int32, K),
                alpha_expr, beta_expr,
                cast(p_int32, ldA), cast(p_int32, ldB), cast(p_int32, ldC),
                cast(p_int32, offsetA), cast(p_int32, offsetB), cast(p_int32, offsetC),
                cast(p_boolean, transposeA), cast(p_boolean, transposeB)
            },
            tiramisu::p_uint8);
}

expr spmv(expr transposeA,
          expr alpha,
          const buffer &csrA,
          const buffer &descrA,
          expr layer_num,
          expr weight_type,
          const buffer &B,
          expr beta,
          buffer &C,
          expr offsetB, expr offsetC
)
{
    return expr(o_call, "tiramisu_spmv",
            {
              cast(p_boolean, transposeA),
              cast(p_float32, alpha),
              var(p_void_ptr, csrA.get_name()),
              var(p_void_ptr, descrA.get_name()),
              cast(p_int32, layer_num),
              cast(p_int32, weight_type),
              var(p_void_ptr, B.get_name()),
              cast(p_float32, beta),
              var(p_void_ptr, C.get_name()),
              cast(p_int32, offsetB), cast(p_int32, offsetC)
            },
            tiramisu::p_uint8);
}

/**
  FlexNLP-Tiramisu API
*/

expr flexnlp_load_weights(const buffer &host_data, expr offset_host, expr num_elem, expr device_id){
  std::string fname;
  fname = "tiramisu_flexnlp_load_weights";

  return expr(o_call, fname,
          {
              var(p_void_ptr, host_data.get_name()),
              cast(p_int32, offset_host),
              cast(p_int32, num_elem),
              cast(p_int32, device_id)
          },
          tiramisu::p_int32);
}

expr flexnlp_load_input(const buffer &host_data, expr offset_host, expr num_elem, expr device_id){
  std::string fname;
  fname = "tiramisu_flexnlp_load_input";

  return expr(o_call, fname,
          {
              var(p_void_ptr, host_data.get_name()),
              cast(p_int32, offset_host),
              cast(p_int32, num_elem),
              cast(p_int32, device_id)
          },
          tiramisu::p_int32);
}

expr flexnlp_store_output(const buffer &host_data, expr offset_host, expr num_elem, expr device_id){
  std::string fname;
  fname = "tiramisu_flexnlp_store_output";

  return expr(o_call, fname,
          {
              var(p_void_ptr, host_data.get_name()),
              cast(p_int32, offset_host),
              cast(p_int32, num_elem),
              cast(p_int32, device_id)
          },
          tiramisu::p_int32);
}

/**
  Run an LSTM Cell inference
*/
expr flexnlp_lstm_cell(const buffer &x_in, const buffer &W_in, const buffer &output, const buffer &h_out,
                       expr layer_number, expr device_id, expr load_weight)
{
    /*
      x_in :  [NTIMESTEPS][BATCH_SIZE][INPUT_SIZE]
      w_in : [NUM_LAYERS][4][OUTPUT_SIZE][INPUT_SIZE + HIDDEN_SIZE]
      output : [NTIMESTEPS][BATCH_SIZE][HIDDEN_SIZE]
      layer_number :
    */
    std::string fname;
    fname = "tiramisu_flexnlp_run_lstm";

    std::vector<expr> sizes_X_in = x_in.get_dim_sizes();
    std::vector<expr> sizes_W_in = W_in.get_dim_sizes();
    std::vector<expr> sizes_output = output.get_dim_sizes();

    expr batch_size = sizes_X_in[0];
    expr timesteps = sizes_X_in[1];
    expr input_size = sizes_X_in[2];

    expr hidden_size = sizes_output[2];

    // If weights can't fit, give an error message
    /*
    PETop* acc = flexnlp_context->get_accelerator_by_id(device_id.get_int_val());
    int weights_memory_limit = acc->GetMemSize(1);
    if (4*hidden_size.get_int_val()*(hidden_size.get_int_val() + input_size.get_int_val())>weights_memory_limit){
      // Say that you need to use
      ERROR("Can't run LSTM at once on a FlexNLP device, weights matrix is too larche to fit in the FlexNLP spad1 memory, please use the tiramisu_flexnlp_run_lstm_split version, and format the weights matrix as [NUM_LAYERS][HSIZE/O_SIZE][4][O_SIZE][HSIZE+ISIZE]", true)
    }
    */

    expr n_layers, output_size;
    if (sizes_W_in.size()<4){ // One single layer
      n_layers = expr(1);
      output_size = sizes_W_in[1];
      layer_number = expr(-1);
    }
    else{
      n_layers = sizes_W_in[0];
      output_size = sizes_W_in[2];
    }

    return expr(o_call, fname,
            {
                var(p_void_ptr, x_in.get_name()),
                var(p_void_ptr, W_in.get_name()),
                var(p_void_ptr, output.get_name()),
                var(p_void_ptr, h_out.get_name()),

                cast(p_int32, input_size),
                cast(p_int32, hidden_size),
                cast(p_int32, output_size),
                cast(p_int32, timesteps),
                cast(p_int32, batch_size),

                cast(p_int32, layer_number),

                cast(p_int32, load_weight),
                cast(p_int32, device_id)
            },
            tiramisu::p_int32);
}

expr flexnlp_lstm_cell_manual(const buffer &x_in, const buffer &W_in, const buffer &output, const buffer &h_out,
                       expr layer_number, expr device_id)
{
    /*
      x_in :  [NTIMESTEPS][BATCH_SIZE][INPUT_SIZE]
      w_in : [NUM_LAYERS][4][OUTPUT_SIZE][INPUT_SIZE + HIDDEN_SIZE]
      output : [NTIMESTEPS][BATCH_SIZE][HIDDEN_SIZE]
      layer_number :
    */
    std::string fname;
    fname = "tiramisu_flexnlp_run_lstm_manual";

    std::vector<expr> sizes_X_in = x_in.get_dim_sizes();
    std::vector<expr> sizes_W_in = W_in.get_dim_sizes();
    std::vector<expr> sizes_output = output.get_dim_sizes();

    expr batch_size = sizes_X_in[0];
    expr timesteps = sizes_X_in[1];
    expr input_size = sizes_X_in[2];

    expr hidden_size = sizes_output[2];

    expr n_layers, output_size;
    if (sizes_W_in.size()<4){ // One single layer
      n_layers = expr(1);
      output_size = sizes_W_in[1];
      layer_number = expr(0);
    }
    else{
      n_layers = sizes_W_in[0];
      output_size = sizes_W_in[2];
    }

    return expr(o_call, fname,
            {
                var(p_void_ptr, x_in.get_name()),
                var(p_void_ptr, W_in.get_name()),
                var(p_void_ptr, output.get_name()),
                var(p_void_ptr, h_out.get_name()),

                cast(p_int32, input_size),
                cast(p_int32, hidden_size),
                cast(p_int32, output_size),
                cast(p_int32, timesteps),
                cast(p_int32, batch_size),

                cast(p_int32, layer_number),

                cast(p_int32, device_id)
            },
            tiramisu::p_int32);
}

/**
  Run an LSTM Cell inference
*/
expr flexnlp_lstm_cell_partitioned(const buffer &x_in, const buffer &W_in, const buffer &output, const buffer &h_out,
                       expr layer_number, expr device_id, expr load_weight)
{
    /*
      x_in :  [NTIMESTEPS][BATCH_SIZE][INPUT_SIZE]
      w_in : [NUM_LAYERS][HIDDEN_SIZE/OUTPUT_SIZE][4][OUTPUT_SIZE][INPUT_SIZE + HIDDEN_SIZE]
      output : [NTIMESTEPS][BATCH_SIZE][HIDDEN_SIZE]
      layer_number :
    */
    std::string fname;
    fname = "tiramisu_flexnlp_run_partitioned_lstm";

    std::vector<expr> sizes_X_in = x_in.get_dim_sizes();
    std::vector<expr> sizes_W_in = W_in.get_dim_sizes();
    std::vector<expr> sizes_output = output.get_dim_sizes();

    expr batch_size = sizes_X_in[0];
    expr timesteps = sizes_X_in[1];
    expr input_size = sizes_X_in[2];

    expr hidden_size = sizes_output[2];

    expr n_layers, output_size;
    if (sizes_W_in.size()<4){ // One single layer
      n_layers = expr(1);
      output_size = sizes_W_in[2];
      layer_number = expr(-1);
    }
    else{
      n_layers = sizes_W_in[0];
      output_size = sizes_W_in[3];
    }

    return expr(o_call, fname,
            {
                var(p_void_ptr, x_in.get_name()),
                var(p_void_ptr, W_in.get_name()),
                var(p_void_ptr, output.get_name()),
                var(p_void_ptr, h_out.get_name()),

                cast(p_int32, input_size),
                cast(p_int32, hidden_size),
                cast(p_int32, output_size),
                cast(p_int32, timesteps),
                cast(p_int32, batch_size),

                cast(p_int32, layer_number),

                cast(p_int32, load_weight),
                cast(p_int32, device_id)
            },
            tiramisu::p_int32);
}

/**
  Run an LSTM Cell inference split among hidden_size/output_size accelerators
*/
expr flexnlp_lstm_cell_partitioned_multi_accelerator(const buffer &x_in, const buffer &W_in, const buffer &output, const buffer &h_out, expr layer_number)
{
    /*
      x_in :  [NTIMESTEPS][BATCH_SIZE][INPUT_SIZE]
      w_in : [NUM_LAYERS][HIDDEN_SIZE/OUTPUT_SIZE][4][OUTPUT_SIZE][INPUT_SIZE + HIDDEN_SIZE]
      output : [NTIMESTEPS][BATCH_SIZE][HIDDEN_SIZE]
      layer_number :
    */
    std::string fname;
    fname = "tiramisu_flexnlp_run_partitioned_lstm_multi";

    std::vector<expr> sizes_X_in = x_in.get_dim_sizes();
    std::vector<expr> sizes_W_in = W_in.get_dim_sizes();
    std::vector<expr> sizes_output = output.get_dim_sizes();

    expr batch_size = sizes_X_in[0];
    expr timesteps = sizes_X_in[1];
    expr input_size = sizes_X_in[2];

    expr hidden_size = sizes_output[2];

    expr n_layers, output_size;
    if (sizes_W_in.size()<4){ // One single layer
      n_layers = expr(1);
      output_size = sizes_W_in[2];
      layer_number = expr(-1);
    }
    else{
      n_layers = sizes_W_in[0];
      output_size = sizes_W_in[3];
    }

    return expr(o_call, fname,
            {
                var(p_void_ptr, x_in.get_name()),
                var(p_void_ptr, W_in.get_name()),
                var(p_void_ptr, output.get_name()),
                var(p_void_ptr, h_out.get_name()),

                cast(p_int32, input_size),
                cast(p_int32, hidden_size),
                cast(p_int32, output_size),
                cast(p_int32, timesteps),
                cast(p_int32, batch_size),

                cast(p_int32, layer_number)
            },
            tiramisu::p_int32);
}

expr flexnlp_initialize(expr number_of_devices)
{
    std::string fname;
    fname = "tiramisu_flexnlp_initialize";

    return expr(o_call, fname,
            {
                cast(p_int32, number_of_devices)
            },
            tiramisu::p_int32);
}

expr flexnlp_finalize()
{
    std::string fname;
    fname = "tiramisu_flexnlp_finalize";

    return expr(o_call, fname,
            {
              int32_t(0)
            },
            tiramisu::p_int32);
}

}
