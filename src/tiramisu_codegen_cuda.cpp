//
// Created by malek on 12/15/17.
//

#include <isl/printer.h>
#include <isl/aff.h>
#include <isl/set.h>
#include <isl/constraint.h>
#include <isl/space.h>
#include <isl/map.h>
#include <isl/union_map.h>
#include <isl/union_set.h>
#include <isl/ast_build.h>
#include <isl/schedule.h>
#include <isl/schedule_node.h>

#include <tiramisu/debug.h>
#include <tiramisu/core.h>
#include <tiramisu/type.h>
#include <tiramisu/expr.h>
#include <tiramisu/utils.h>

#include <string>
#include <array>
#include <tiramisu/cuda_ast.h>
#include <isl/ast_type.h>
#include <isl/ast.h>

#include <fstream>
#include <memory>

namespace tiramisu {
    tiramisu::expr replace_original_indices_with_transformed_indices(tiramisu::expr exp,
                                                                     std::map<std::string, isl_ast_expr *> iterators_map);
namespace cuda_ast {

const tiramisu::cuda_ast::op_data_t tiramisu_operation_description(tiramisu::op_t op) {
    switch (op) {
        UNARY2(tiramisu::o_minus, "-")
        FN_CALL2(tiramisu::o_floor, "floor", 1)
        FN_CALL2(tiramisu::o_sin, "sin", 1)
        FN_CALL2(tiramisu::o_cos, "cos", 1)
        FN_CALL2(tiramisu::o_tan, "tan", 1)
        FN_CALL2(tiramisu::o_asin, "asin", 1)
        FN_CALL2(tiramisu::o_acos, "acos", 1)
        FN_CALL2(tiramisu::o_atan, "atan", 1)
        FN_CALL2(tiramisu::o_sinh, "sinh", 1)
        FN_CALL2(tiramisu::o_cosh, "cosh", 1)
        FN_CALL2(tiramisu::o_tanh, "tanh", 1)
        FN_CALL2(tiramisu::o_asinh, "asinh", 1)
        FN_CALL2(tiramisu::o_acosh, "acosh", 1)
        FN_CALL2(tiramisu::o_atanh, "atanh", 1)
        FN_CALL2(tiramisu::o_abs, "abs", 1)
        FN_CALL2(tiramisu::o_sqrt, "sqrt", 1)
        FN_CALL2(tiramisu::o_expo, "exp", 1)
        FN_CALL2(tiramisu::o_log, "log", 1)
        FN_CALL2(tiramisu::o_ceil, "ceil", 1)
        FN_CALL2(tiramisu::o_round, "round", 1)
        FN_CALL2(tiramisu::o_trunc, "trunc", 1)
        BINARY2(tiramisu::o_add, "+")
        BINARY2(tiramisu::o_sub, "-")
        BINARY2(tiramisu::o_mul, "*")
        BINARY2(tiramisu::o_div, "/")
        BINARY2(tiramisu::o_mod, "%")
        BINARY2(tiramisu::o_logical_and, "&&")
        BINARY2(tiramisu::o_logical_or, "||")
        UNARY2(tiramisu::o_logical_not, "!")
        BINARY2(tiramisu::o_eq, "==")
        BINARY2(tiramisu::o_ne, "!=")
        BINARY2(tiramisu::o_le, "<=")
        BINARY2(tiramisu::o_lt, "<")
        BINARY2(tiramisu::o_ge, ">=")
        BINARY2(tiramisu::o_gt, ">")
        FN_CALL2(tiramisu::o_max, "max", 2)
        FN_CALL2(tiramisu::o_min, "min", 2)
        BINARY2(tiramisu::o_right_shift, ">>")
        BINARY2(tiramisu::o_left_shift, "<<")
        TERNARY2(tiramisu::o_select, "?", ":")
        FN_CALL2(tiramisu::o_lerp, "lerp", 3)
        default:
            assert(false);
            return tiramisu::cuda_ast::op_data_t();
    }
}

const tiramisu::cuda_ast::op_data_t isl_operation_description(isl_ast_op_type op) {
    switch (op) {
        BINARY_TYPED2(isl_ast_op_and, "&&", tiramisu::p_boolean)
        BINARY_TYPED2(isl_ast_op_and_then, "&&", tiramisu::p_boolean)
        BINARY_TYPED2(isl_ast_op_or, "||", tiramisu::p_boolean)
        BINARY_TYPED2(isl_ast_op_or_else, "||", tiramisu::p_boolean)
        FN_CALL2(isl_ast_op_max, "max", 2)
        FN_CALL2(isl_ast_op_min, "min", 2)
        UNARY2(isl_ast_op_minus, "-")
        BINARY2(isl_ast_op_add, "+")
        BINARY2(isl_ast_op_sub, "-")
        BINARY2(isl_ast_op_mul, "*")
        BINARY2(isl_ast_op_div, "/")
        BINARY2(isl_ast_op_fdiv_q, "/")
        BINARY2(isl_ast_op_pdiv_q, "/")
        BINARY2(isl_ast_op_pdiv_r, "%")
        BINARY2(isl_ast_op_zdiv_r, "%")
        TERNARY2(isl_ast_op_cond, "?", ":")
        FN_CALL2(isl_ast_op_select, "lerp", 3)
        BINARY_TYPED2(isl_ast_op_eq, "==", tiramisu::p_boolean)
        BINARY_TYPED2(isl_ast_op_le, "<=", tiramisu::p_boolean)
        BINARY_TYPED2(isl_ast_op_lt, "<", tiramisu::p_boolean)
        BINARY_TYPED2(isl_ast_op_ge, ">=", tiramisu::p_boolean)
        BINARY_TYPED2(isl_ast_op_gt, ">", tiramisu::p_boolean)
        default: {
            assert(false);
            return tiramisu::cuda_ast::op_data_t();
        }
    }
}

const std::string tiramisu_type_to_cuda_type(tiramisu::primitive_t t) {
    switch (t) {
        case tiramisu::p_none:
            return "void";
        case tiramisu::p_boolean:
            return "bool";
        case tiramisu::p_int8:
            return "int8_t";
        case tiramisu::p_uint8:
            return "uint8_t";
        case tiramisu::p_int16:
            return "int16_t";
        case tiramisu::p_uint16:
            return "uint16_t";
        case tiramisu::p_int32:
            return "int32_t";
        case tiramisu::p_uint32:
            return "uint32_t";
        case tiramisu::p_int64:
            return "int64_t";
        case tiramisu::p_uint64:
            return "uint64_t";
        case tiramisu::p_float32:
            return "float";
        case tiramisu::p_float64:
            return "double";
        default: {
            assert(false);
            return "";
        }
    }
}

}
}

namespace tiramisu {

cuda_ast::statement_ptr tiramisu::cuda_ast::generator::cuda_stmt_from_isl_node(isl_ast_node *node) {
        isl_ast_node_type type = isl_ast_node_get_type(node);

        switch (type) {
            case isl_ast_node_for:
                return cuda_stmt_handle_isl_for(node);
            case isl_ast_node_block:
                return cuda_stmt_handle_isl_block(node);
            case isl_ast_node_if:
                return cuda_stmt_handle_isl_if(node);
            case isl_ast_node_mark: DEBUG(3, tiramisu::str_dump("mark"));
                return nullptr;
            case isl_ast_node_user:
                return cuda_stmt_handle_isl_user(node);
            default: DEBUG(3, tiramisu::str_dump("default"));
                return nullptr;
        }
    }

    cuda_ast::statement_ptr tiramisu::cuda_ast::generator::cuda_stmt_handle_isl_if(isl_ast_node *node) {
        isl_ast_node *then_body = isl_ast_node_if_get_then(node);
        isl_ast_expr_ptr condition{isl_ast_node_if_get_cond(node)};

        statement_ptr then_body_stmt{cuda_stmt_from_isl_node(then_body)};
        statement_ptr else_body_stmt;
        if (isl_ast_node_if_has_else(node)) {
            isl_ast_node *else_body = isl_ast_node_if_get_else(node);
            else_body_stmt = cuda_stmt_from_isl_node(else_body);
        }

        statement_ptr condition_stmt{cuda_stmt_handle_isl_expr(condition, node)};

        if (else_body_stmt) {
            return statement_ptr{new cuda_ast::if_condition{condition_stmt, then_body_stmt, else_body_stmt}};
        } else {
            return statement_ptr{new cuda_ast::if_condition{condition_stmt, then_body_stmt}};
        }

    }

    cuda_ast::statement_ptr tiramisu::cuda_ast::generator::cuda_stmt_handle_isl_block(isl_ast_node *node) {
        isl_ast_node_list_ptr children_list{isl_ast_node_block_get_children(node)};
        const int block_length = isl_ast_node_list_n_ast_node(children_list.get());
        auto *b = new block;
        for (int i = 0; i < block_length; i++) {
            isl_ast_node *child_node = isl_ast_node_list_get_ast_node(children_list.get(), i);
            b->add_statement(cuda_stmt_from_isl_node(child_node));
        }
        return statement_ptr{b};
    }

    void
    tiramisu::cuda_ast::generator::cuda_stmt_foreach_isl_expr_list(isl_ast_expr *node,
                                                                   const std::function<void(int, isl_ast_expr *)> &fn,
                                                                   int start) {
        int n = isl_ast_expr_get_op_n_arg(node);
        for (int i = start; i < n; i++) {
            fn(i, isl_ast_expr_get_op_arg(node, i));
        }
    }

    cuda_ast::statement_ptr tiramisu::cuda_ast::generator::cuda_stmt_val_from_for_condition(isl_ast_expr_ptr &expr, isl_ast_node *node)
    {
        // TODO this is potentially a hack
        assert(isl_ast_expr_get_type(expr.get()) == isl_ast_expr_type::isl_ast_expr_op);
        auto expr_type = isl_ast_expr_get_op_type(expr.get());
        assert(expr_type == isl_ast_op_type::isl_ast_op_lt || expr_type == isl_ast_op_type::isl_ast_op_le);
        isl_ast_expr_ptr value{isl_ast_expr_get_op_arg(expr.get(), 1)};
        if (expr_type == isl_ast_op_lt) {
            DEBUG(3, tiramisu::str_dump("not supported"));
            // Even though this is "not supported", this case happens with block dimensions sometimes.
            // cuda_ast expects loop ranges to be inclusive so we adjust the value by subtracting 1.
            cuda_ast::statement_ptr stmt = cuda_stmt_handle_isl_expr(value, node);
            return statement_ptr{new binary{stmt->get_type(),
                stmt,
                statement_ptr{new cuda_ast::value{value_cast(stmt->get_type(), 1)}},
                "-"}};
        } else {
            return cuda_stmt_handle_isl_expr(value, node);
        }
    }

    cuda_ast::statement_ptr tiramisu::cuda_ast::generator::cuda_stmt_handle_isl_for(isl_ast_node *node) {
        isl_ast_expr_ptr iterator{isl_ast_node_for_get_iterator(node)};
        isl_id_ptr iterator_id{isl_ast_expr_get_id(iterator.get())};
        std::string iterator_name(isl_id_get_name(iterator_id.get()));
        DEBUG(3, tiramisu::str_dump("The iterator name is: ", iterator_name.c_str()));


        isl_ast_expr_ptr condition{isl_ast_node_for_get_cond(node)};
        isl_ast_expr_ptr incrementor{isl_ast_node_for_get_inc(node)};
        isl_ast_expr_ptr initializer{isl_ast_node_for_get_init(node)};
        isl_ast_node *body = isl_ast_node_for_get_body(node);


        // TODO check if degenerate

        statement_ptr result;

        m_scalar_data.insert(
                std::make_pair(iterator_name,
                               std::make_pair(tiramisu::global::get_loop_iterator_data_type(),
                                              cuda_ast::memory_location::reg)));
        iterator_stack.push_back(iterator_name);
        auto initializer_stmt = cuda_stmt_handle_isl_expr(initializer, node);
        iterator_upper_bound.push_back(cuda_stmt_val_from_for_condition(condition, node));
        iterator_lower_bound.push_back(initializer_stmt);

        this->loop_level ++;
        auto body_statement = cuda_stmt_from_isl_node(body);
        this->loop_level --;

        // Check if GPU
        auto gpu_it = gpu_iterators.find(iterator_name);
        if (gpu_it != gpu_iterators.end())
        {
            current_kernel->set_dimension(gpu_it->second);
            gpu_iterators.erase(gpu_it);
            if (gpu_iterators.empty())
            {
                auto * final_body = new cuda_ast::block;
                for (auto &dec : kernel_simplified_vars)
                    final_body->add_statement(dec);
                for (auto &cond : gpu_conditions)
                    body_statement = statement_ptr{new if_condition(cond, body_statement)};
                final_body->add_statement(body_statement);
                gpu_conditions.clear();
                gpu_local.clear();
                kernel_simplified_vars.clear();
                current_kernel->set_body(statement_ptr{final_body});
                // Add previous iterators as arguments to the kernel:
                for (auto it = iterator_stack.begin(); it != iterator_stack.end() - 1; it++) {
                    current_kernel->add_used_scalar(
                        scalar_ptr{new cuda_ast::scalar{
                            tiramisu::global::get_loop_iterator_data_type(),
                            *it,
                            cuda_ast::memory_location::reg}});
                }
                kernels.push_back(current_kernel);
                result = statement_ptr{new kernel_call{current_kernel}};
                iterator_to_kernel_map[node] = current_kernel;
                current_kernel.reset();
                in_kernel = false;
            } else {
                result = body_statement;
            }
        }
        else
        {
            auto it = std::static_pointer_cast<cuda_ast::scalar>(cuda_stmt_handle_isl_expr(iterator, node));
            auto initializer_statement = statement_ptr{new declaration{
                    assignment_ptr{new scalar_assignment{it, initializer_stmt}}}};
            auto condition_statement = cuda_stmt_handle_isl_expr(condition, node);
            auto incrementor_statement = statement_ptr{new binary{it->get_type(), it, cuda_stmt_handle_isl_expr(incrementor, node), "+="}};

            // TODO get loop bound in the core


            result = statement_ptr{new cuda_ast::for_loop{
                    initializer_statement,
                    condition_statement,
                    incrementor_statement,
                    body_statement}};
        }


        m_scalar_data.erase(iterator_name);

        iterator_lower_bound.pop_back();
        iterator_upper_bound.pop_back();
        iterator_stack.pop_back();

        return result;
    }

    cuda_ast::statement_ptr tiramisu::cuda_ast::generator::cuda_stmt_handle_isl_expr(isl_ast_expr_ptr &expr,
                                                                                     isl_ast_node *node) {
        isl_ast_expr_type type = isl_ast_expr_get_type(expr.get());
        switch (type) {
            case isl_ast_expr_op: DEBUG(3, tiramisu::str_dump("isl op"));
                return cuda_stmt_handle_isl_op_expr(expr, node);
            case isl_ast_expr_id: {
                isl_id *id = isl_ast_expr_get_id(expr.get());
                std::string id_string(isl_id_get_name(id));
                DEBUG(3, std::cout << '"' << id_string << '"');
                // TODO handle scheduled lets
                return get_scalar_from_name(id_string);
            }
            case isl_ast_expr_int: {
                isl_val_ptr val{isl_ast_expr_get_val(expr.get())};
                return cuda_stmt_handle_isl_val(val);
            }
            default: DEBUG(3, tiramisu::str_dump("expr default"));
                return nullptr;
                break;
        }
    }


    cuda_ast::value_ptr tiramisu::cuda_ast::generator::cuda_stmt_handle_isl_val(isl_val_ptr &node) {
        // TODO handle infinity
        long num = isl_val_get_num_si(node.get());
        long den = isl_val_get_den_si(node.get());
        assert(den == 1);
        return value_ptr{new cuda_ast::value{value_cast(global::get_loop_iterator_data_type(), num)}};
    }


    cuda_ast::statement_ptr tiramisu::cuda_ast::generator::parse_tiramisu(const tiramisu::expr &tiramisu_expr) {
        switch (tiramisu_expr.get_expr_type()) {
            case e_val:
                return statement_ptr{new cuda_ast::value{tiramisu_expr}};
            case e_var:
            {
                return get_scalar_from_name(tiramisu_expr.get_name());
            }

            case e_none:
                assert(false);
            case e_op: {
                switch (tiramisu_expr.get_op_type()) {
                    case o_access: {
                        buffer_ptr b = this->get_buffer(tiramisu_expr.get_name());
                        std::vector<statement_ptr> indices;
                        for (auto &access: tiramisu_expr.get_access()) {
                            indices.push_back(this->parse_tiramisu(access));
                        }
                        return statement_ptr{new buffer_access{b, indices}};
                    }
                    case o_call: {
                        std::vector<statement_ptr> operands{static_cast<size_t>(tiramisu_expr.get_n_arg())};
                        std::transform(tiramisu_expr.get_arguments().begin(), tiramisu_expr.get_arguments().end(),
                                       operands.begin(),
                                       std::bind(&generator::parse_tiramisu, this, std::placeholders::_1));
                        return statement_ptr{
                                new function_call{tiramisu_expr.get_data_type(), tiramisu_expr.get_name(), operands}};
                    }
                    case o_cast: {
                        // Add a cast statement only if necessary
                        if (tiramisu_expr.get_data_type() != tiramisu_expr.get_operand(0).get_data_type()) {
                            return statement_ptr{new cuda_ast::cast{tiramisu_expr.get_data_type(),
                                parse_tiramisu(tiramisu_expr.get_operand(0))}};
                        } else {
                            return parse_tiramisu(tiramisu_expr.get_operand(0));
                        }
                    }
                    case o_allocate:
                    {
                        auto buffer = get_buffer(tiramisu_expr.get_name());
                        if (buffer->get_location() == memory_location::shared
                                || buffer->get_location() == memory_location::local
                                || buffer->get_location() == memory_location::reg)
                            return statement_ptr {new cuda_ast::declaration{buffer}};
                        else
                            return statement_ptr {new cuda_ast::allocate{buffer}};
                    }
                    case o_free:
                        return statement_ptr {new cuda_ast::free{get_buffer(tiramisu_expr.get_name())}};

                    case o_memcpy:
                        assert(tiramisu_expr.get_operand(0).get_expr_type() == e_var && tiramisu_expr.get_operand(1).get_expr_type() == e_var && "Can only transfer from buffers to buffers");
                        return statement_ptr {new cuda_ast::memcpy{this->get_buffer(tiramisu_expr.get_operand(0).get_name()), this->get_buffer(tiramisu_expr.get_operand(1).get_name())}};
                    default: {
//                        auto it = cuda_ast::tiramisu_operation_description(tiramisu_expr.get_op_type());
//                        assert(it != cuda_ast::tiramisu_operation_description.cend());
                        const op_data_t &op_data = cuda_ast::tiramisu_operation_description(tiramisu_expr.get_op_type());//it->second;
                        std::vector<statement_ptr> operands;
                        for (int i = 0; i < op_data.arity; i++) {
                            operands.push_back(parse_tiramisu(tiramisu_expr.get_operand(i)));
                        }
                        if (op_data.infix) {
                            assert(op_data.arity > 0 && op_data.arity < 4 &&
                                   "Infix operators are either unary, binary, or tertiary.");
                            switch (op_data.arity) {
                                case 1:
                                    return statement_ptr {
                                            new cuda_ast::unary{tiramisu_expr.get_data_type(), operands[0],
                                                                std::string{op_data.symbol}}};
                                case 2:
                                    return statement_ptr {
                                            new cuda_ast::binary{tiramisu_expr.get_data_type(), operands[0],
                                                                 operands[1], std::string{op_data.symbol}}};
                                case 3:
                                    return statement_ptr {
                                            new cuda_ast::ternary{tiramisu_expr.get_data_type(), operands[0],
                                                                  operands[1], operands[2], std::string{op_data.symbol},
                                                                  std::string{op_data.next_symbol}}};
                                default:
                                    assert(false && "Infix operators are either unary, binary, or tertiary.");
                            }
                        } else {
                            return statement_ptr{
                                    new cuda_ast::function_call{tiramisu_expr.get_data_type(), op_data.symbol,
                                                                operands}};
                        }
                    }
                }
            }
            default:
                assert(false);
        }
    }

    cuda_ast::buffer_ptr tiramisu::cuda_ast::generator::get_buffer(const std::string &name) {
        auto it = m_buffers.find(name);
        cuda_ast::buffer_ptr buffer;
        if (it != m_buffers.end())
            buffer = it->second;
        else {
            auto tiramisu_buffer = this->m_fct.get_buffers().at(name);
            std::vector<cuda_ast::statement_ptr> sizes;
            for (auto &dim : tiramisu_buffer->get_dim_sizes()) {

                sizes.push_back(this->parse_tiramisu(dim));
            }
            buffer = buffer_ptr{new cuda_ast::buffer{tiramisu_buffer->get_elements_type(), tiramisu_buffer->get_name(),
                                                     tiramisu_buffer->location, sizes}};
            m_buffers[name] = buffer;
        }
        if (in_kernel && gpu_local.find(name) == gpu_local.end())
        {
            current_kernel->add_used_buffer(buffer);
        }
        return buffer;
    }

    cuda_ast::gpu_iterator cuda_ast::generator::get_gpu_condition(gpu_iterator::type_t type, gpu_iterator::dimension_t dim,
                                                                      cuda_ast::statement_ptr lower_bound,
                                                                      cuda_ast::statement_ptr upper_bound) {
        auto min_cap = upper_bound->extract_min_cap();
        statement_ptr actual_bound;
        if (min_cap.first)
        {
            actual_bound = min_cap.first;
        } else {
            actual_bound = upper_bound;
        }
        gpu_iterator result{type, dim, statement_ptr{new binary{actual_bound->get_type(),
                                                                actual_bound,
                                                                statement_ptr{new value{value_cast(actual_bound->get_type(), 1)}},
                                                                "+"}}};
        if (min_cap.first) {
            statement_ptr it_access{new gpu_iterator_read{result}};
            gpu_conditions.push_back(
                    statement_ptr{
                            new binary{p_boolean, it_access, min_cap.second->replace_iterators(gpu_iterators), "<="}}
            );
        }

        kernel_simplified_vars.push_back( statement_ptr{ new declaration{ assignment_ptr{
                new scalar_assignment{
                        scalar_ptr{new scalar{lower_bound->get_type(), result.simplified_name(), memory_location::reg, true}},
                        statement_ptr { new binary{
                                lower_bound->get_type(),
                                statement_ptr{new gpu_iterator_read{result, false}},
                                lower_bound->replace_iterators(gpu_iterators),
                                "+"
                        }}
                }}}}
        );
        return result;
    }

    cuda_ast::statement_ptr tiramisu::cuda_ast::generator::cuda_stmt_handle_isl_op_expr(isl_ast_expr_ptr &expr,
                                                                                        isl_ast_node *node) {
        isl_ast_op_type op_type = isl_ast_expr_get_op_type(expr.get());
        if (op_type == isl_ast_op_call) {
            auto *comp = get_computation_annotated_in_a_node(node);
            // TODO use lower bound
            if (!this->in_kernel) {
                for (auto &comp_gpu_pair : this->m_fct.gpu_block_dimensions) {
                    if (comp_gpu_pair.first == comp->get_name()) {
                        int level;
                        this->in_kernel = true;
                        if ((level = std::get<0>(comp_gpu_pair.second)) != -1) {
                            this->gpu_iterators[iterator_stack[level]] = get_gpu_condition(gpu_iterator::type_t::BLOCK,
                                                                                           gpu_iterator::dimension_t::x,
                                                                                           iterator_lower_bound[level],
                                                                                           iterator_upper_bound[level]);
                        }
                        if ((level = std::get<1>(comp_gpu_pair.second)) != -1) {
                            this->gpu_iterators[iterator_stack[level]] = get_gpu_condition(gpu_iterator::type_t::BLOCK,
                                                                                           gpu_iterator::dimension_t::y,
                                                                                           iterator_lower_bound[level],
                                                                                           iterator_upper_bound[level]);
                        }
                        if ((level = std::get<2>(comp_gpu_pair.second)) != -1) {
                            this->gpu_iterators[iterator_stack[level]] = get_gpu_condition(gpu_iterator::type_t::BLOCK,
                                                                                           gpu_iterator::dimension_t::z,
                                                                                           iterator_lower_bound[level],
                                                                                           iterator_upper_bound[level]);
                        }
                        break;
                    }
                }


                if (in_kernel) {
                    for (auto &comp_gpu_pair : this->m_fct.gpu_thread_dimensions) {
                        if (comp_gpu_pair.first == comp->get_name()) {
                            int level;
                            if ((level = std::get<0>(comp_gpu_pair.second)) != -1) {
                                this->gpu_iterators[iterator_stack[level]] = get_gpu_condition(
                                        gpu_iterator::type_t::THREAD,
                                        gpu_iterator::dimension_t::x,
                                        iterator_lower_bound[level],
                                        iterator_upper_bound[level]);
                            }
                            if ((level = std::get<1>(comp_gpu_pair.second)) != -1) {
                                this->gpu_iterators[iterator_stack[level]] = get_gpu_condition(
                                        gpu_iterator::type_t::THREAD,
                                        gpu_iterator::dimension_t::y,
                                        iterator_lower_bound[level],
                                        iterator_upper_bound[level]);
                            }
                            if ((level = std::get<2>(comp_gpu_pair.second)) != -1) {
                                this->gpu_iterators[iterator_stack[level]] = get_gpu_condition(
                                        gpu_iterator::type_t::THREAD,
                                        gpu_iterator::dimension_t::z,
                                        iterator_lower_bound[level],
                                        iterator_upper_bound[level]);
                            }
                            this->current_kernel = kernel_ptr{new kernel};
                            break;
                        }
                    }
                }
                for (auto &it : gpu_conditions) {
                    auto used_scalars = it->extract_scalars();
                    for (auto &scalar: used_scalars) {
                        if (gpu_iterators.find(scalar) == gpu_iterators.end()) {
                            auto data = m_scalar_data.at(scalar);
                            this->current_kernel->add_used_scalar(
                                    scalar_ptr{new cuda_ast::scalar{data.first, scalar, data.second}});
                        }
                    }
                }
            }
            if (comp->get_expr().get_expr_type() == e_sync) {
                return statement_ptr{new cuda_ast::sync};
            }
            else if (comp->get_expr().get_op_type() == o_memcpy)
            {
                return statement_ptr{parse_tiramisu(comp->get_expr())};
            }
            else if (comp->get_expr().get_op_type() == o_allocate)
            {
                this->gpu_local.insert(comp->get_expr().get_name());
                auto buffer = get_buffer(comp->get_expr().get_name());
                if (buffer->get_location() == memory_location::shared
                        || buffer->get_location() == memory_location::local
                        || buffer->get_location() == memory_location::reg)
                    return statement_ptr {new cuda_ast::declaration{buffer}};
                else
                    return statement_ptr {new cuda_ast::allocate{buffer}};
            } else if (comp->get_expr().get_op_type() == o_free)
            {
                auto buffer = get_buffer(comp->get_expr().get_name());
                return statement_ptr {new cuda_ast::free{buffer}};
            } else {
                auto &associated_lets = comp->get_associated_let_stmts();
                for (auto &statement: associated_lets)
                {
                    this->m_scalar_data.insert(std::make_pair(statement.first, std::make_pair(statement.second.get_data_type(), memory_location::reg)));
                }
                if (index_exprs.find(comp) == index_exprs.end())
                    index_exprs[comp] = comp->index_expr;
                auto result = comp->create_tiramisu_assignment(index_exprs[comp]);
                statement_ptr asgmnt;
                if (result.first.get_expr_type() == e_var)
                {
                    cuda_ast::scalar_ptr s = scalar_ptr{new scalar{result.first.get_data_type(), result.first.get_name(), memory_location::reg, true}};
                    // TODO associated let statement doesn't work well with declaration
                    asgmnt = statement_ptr{new declaration{assignment_ptr{new scalar_assignment{s, parse_tiramisu(result.second)}}}};
                    // TODO remove once done
                    m_scalar_data.insert(std::make_pair(result.first.get_name(), std::make_pair(result.first.get_data_type(), memory_location::reg)));
                    gpu_local.insert(result.first.get_name());
                }
                else
                {
                    cuda_ast::buffer_ptr b = this->get_buffer(result.first.get_name());
                    asgmnt = statement_ptr{new buffer_assignment{b, parse_tiramisu(result.first.get_access()[0]),
                                                                  parse_tiramisu(result.second)}};
                }
                if (comp->get_predicate().is_defined()) {

                    std::vector<isl_ast_expr *> ie = {}; // Dummy variable.
                    tiramisu::expr tiramisu_predicate = replace_original_indices_with_transformed_indices(
                            comp->get_predicate(),
                            comp->get_iterators_map());
                    asgmnt = statement_ptr{new if_condition{parse_tiramisu(tiramisu_predicate), asgmnt}};
                }
                if (associated_lets.empty()) {
                    return asgmnt;
                } else {
                    auto *block_result = new cuda_ast::block;
                    for (auto &stmt: associated_lets) {
                        auto associated_expr = replace_original_indices_with_transformed_indices(stmt.second, comp->get_iterators_map());
                        std::vector<isl_ast_expr *> ie;
                        associated_expr = tiramisu::generator::replace_accesses(&this->m_fct, ie, associated_expr);
                        block_result->add_statement(
                                statement_ptr{new declaration{
                                        assignment_ptr{new scalar_assignment{
                                                scalar_ptr{new scalar{stmt.second.get_data_type(), stmt.first,
                                                                      memory_location::reg, true}},
                                                parse_tiramisu(associated_expr)}}}});
                        m_scalar_data.erase(stmt.first);

                    }
                    block_result->add_statement(asgmnt);
                    return statement_ptr{block_result};
                }
            }
        } else {
//            auto it = isl_operation_description.find(op_type);
//            assert(it != isl_operation_description.end() && "Operation not supported");
            auto &description = isl_operation_description(op_type);//it->second;

            std::vector<statement_ptr> operands;
            for (int i = 0; i < description.arity; i++) {
                isl_ast_expr_ptr arg{isl_ast_expr_get_op_arg(expr.get(), i)};
                operands.push_back(cuda_stmt_handle_isl_expr(arg, node));
            }
            primitive_t type = (description.type_preserving) ? operands.back()->get_type()
                                                             : description.type; // Get the type of the last element because ternary condition

            if (description.infix) {
                switch (description.arity) {
                    case 1:
                        return statement_ptr{new cuda_ast::unary{type, operands[0], std::string(description.symbol)}};
                    case 2:
                        return statement_ptr{
                                new cuda_ast::binary{type, operands[0], operands[1], std::string(description.symbol)}};
                    case 3:
                        return statement_ptr{new cuda_ast::ternary{type, operands[0], operands[1], operands[2],
                                                                   std::string(description.symbol),
                                                                   std::string(description.next_symbol)}};
                    default:
                        assert(false && "Infix operators are either unary, binary, or tertiary.");
                }
            } else {
                return statement_ptr{new cuda_ast::function_call{type, description.symbol, operands}};
            }
        }

    }

    cuda_ast::statement_ptr tiramisu::cuda_ast::generator::cuda_stmt_handle_isl_user(isl_ast_node *node) {
        isl_ast_expr_ptr expr{isl_ast_node_user_get_expr(node)};
        return cuda_stmt_handle_isl_expr(expr, node);
    }

    void tiramisu::function::gen_cuda_stmt() {
        DEBUG_FCT_NAME(3);
        DEBUG_INDENT(4);

        DEBUG(3, this->gen_c_code());

        cuda_ast::generator generator{*this};

        auto *body = new cuda_ast::block;
        auto main_body = generator.cuda_stmt_from_isl_node(this->get_isl_ast());

        for (auto &kernel : generator.kernels)
        {
            body->add_statement(cuda_ast::statement_ptr{new cuda_ast::kernel_definition{kernel}});
        }
        auto function_body = new cuda_ast::block;

        for (auto &invariant: this->get_invariants()) {
            std::vector<isl_ast_expr*> ie{};
            auto rhs = generator.parse_tiramisu(generator::replace_accesses(this, ie, invariant.get_expr()));
            auto scalar = cuda_ast::scalar_ptr{
                    new cuda_ast::scalar{rhs->get_type(), invariant.get_name(), cuda_ast::memory_location::reg, true}};
            function_body->add_statement(
                    cuda_ast::statement_ptr{new cuda_ast::declaration{
                            cuda_ast::assignment_ptr{new cuda_ast::scalar_assignment{scalar, rhs}}}});
        }
        std::vector<cuda_ast::statement_ptr> allocations;
        std::vector<cuda_ast::statement_ptr> const_buffer_declaration;
        std::vector<cuda_ast::statement_ptr> frees;
        for (const auto &b : this->get_buffers()) {
            tiramisu::buffer *buf = b.second;
            // Allocate only arrays that are not passed to the function as arguments.
            if (buf->get_argument_type() == tiramisu::a_temporary && buf->get_auto_allocate()) {
                auto cuda_ast_buffer = generator.get_buffer(buf->get_name());

                cuda_ast::statement_ptr declaration{new cuda_ast::declaration{cuda_ast_buffer}};
                body->add_statement(declaration);

                if (buf->location != cuda_ast::memory_location::constant) {
                    allocations.push_back(cuda_ast::statement_ptr{new cuda_ast::allocate(cuda_ast_buffer)});
                    frees.push_back(cuda_ast::statement_ptr{new cuda_ast::free(cuda_ast_buffer)});
                }


                buf->mark_as_allocated();

                if (buf->location == cuda_ast::memory_location::constant)
                {
                    const_buffer_declaration.push_back(declaration);
                    auto hf = new cuda_ast::host_function{p_none, buf->get_name() + "_get_symbol", {},
                                            cuda_ast::statement_ptr{new cuda_ast::return_statement{std::static_pointer_cast<cuda_ast::statement>(cuda_ast_buffer)}}};
                    hf->set_pointer_return();
                    const_buffer_declaration.push_back(cuda_ast::statement_ptr{hf});
                }
            }
        }
        for (auto &a : allocations)
            function_body->add_statement(a);
        function_body->add_statement(main_body);
        for (auto &f : frees)
            function_body->add_statement(f);
        std::vector<cuda_ast::abstract_identifier_ptr> arguments;
        for (auto &b : this->get_arguments())
            arguments.push_back(generator.get_buffer(b->get_name()));
        body->add_statement(cuda_ast::statement_ptr {new cuda_ast::host_function{p_none, this->get_name(), arguments, cuda_ast::statement_ptr{function_body}}});

        std::cout << ((cuda_ast::statement *) body)->print() << std::endl;
        delete body;

        std::shared_ptr<cuda_ast::block> resulting_file{new cuda_ast::block};

        for (auto &s: const_buffer_declaration)
            resulting_file->add_statement(s);


        for (auto &kernel: generator.kernels)
        {
            resulting_file->add_statement(cuda_ast::statement_ptr{new cuda_ast::kernel_definition{kernel}});

            std::shared_ptr<cuda_ast::block> wrapper_block{new cuda_ast::block};
            wrapper_block->add_statement(cuda_ast::statement_ptr{new cuda_ast::kernel_call{kernel}});
            wrapper_block->add_statement(cuda_ast::statement_ptr{
                    new cuda_ast::return_statement{
                            cuda_ast::statement_ptr{new cuda_ast::value(value_cast(cuda_ast::kernel::wrapper_return_type, 0))}
                    }
            });
            resulting_file->add_statement(
                    cuda_ast::statement_ptr{new cuda_ast::host_function{cuda_ast::kernel::wrapper_return_type,
                                                                        kernel->get_wrapper_name(), kernel->get_arguments(),
                                                                        std::static_pointer_cast<cuda_ast::statement>(wrapper_block)}}
            );
        }


        std::string code = std::static_pointer_cast<cuda_ast::statement>(resulting_file)->print();
        std::cout << code << std::endl;

        nvcc_compiler = std::shared_ptr<cuda_ast::compiler>{new cuda_ast::compiler{code}};

        this->iterator_to_kernel_map = generator.iterator_to_kernel_map;


        DEBUG_INDENT(-4);
    }

    tiramisu::cuda_ast::generator::generator(tiramisu::function &fct) : m_fct(fct) {
        for (const tiramisu::constant &invariant : fct.get_invariants()) {
            m_scalar_data.insert(std::make_pair(invariant.get_name(),
                                                std::make_pair(invariant.get_data_type(),
                                                               cuda_ast::memory_location::reg)));
        }
    }


    cuda_ast::statement::statement(primitive_t type) : type(type) {}

    cuda_ast::cast::cast(primitive_t type, statement_ptr stmt) : statement(type), to_be_cast(stmt) {}

    cuda_ast::abstract_identifier::abstract_identifier(primitive_t type, const std::string &name,
                                                       cuda_ast::memory_location location) : statement(type),
                                                                                             name(name),
                                                                                             location(location) {}

    const std::string &cuda_ast::abstract_identifier::get_name() const {
        return name;
    }

    cuda_ast::memory_location cuda_ast::abstract_identifier::get_location() const {
        return location;
    }

    cuda_ast::buffer::buffer(primitive_t type, const std::string &name, cuda_ast::memory_location location,
                             const std::vector<cuda_ast::statement_ptr> &size) : abstract_identifier(type, name,
                                                                                                     location),
                                                                                 size(size) {}

    cuda_ast::scalar::scalar(primitive_t type, const std::string &name, cuda_ast::memory_location location)
            : abstract_identifier(type, name, location), is_const(false) {}

    cuda_ast::scalar::scalar(primitive_t type, const std::string &name, memory_location location, bool is_const) : abstract_identifier(type, name, location), is_const(is_const){}

    cuda_ast::value::value(uint8_t val) : statement(p_uint8), u8_val(val) {}
    cuda_ast::value::value(int8_t val) : statement(p_int8), i8_val(val) {}
    cuda_ast::value::value(uint16_t val) : statement(p_uint16), u16_val(val) {}
    cuda_ast::value::value(int16_t val) : statement(p_int16), i16_val(val) {}
    cuda_ast::value::value(uint32_t val) : statement(p_uint32), u32_val(val) {}
    cuda_ast::value::value(int32_t val) : statement(p_int32), i32_val(val) {}
    cuda_ast::value::value(uint64_t val) : statement(p_uint64), u64_val(val) {}
    cuda_ast::value::value(int64_t val) : statement(p_int64), i64_val(val) {}
    cuda_ast::value::value(float val) : statement(p_float32), f32_val(val) {}
    cuda_ast::value::value(double val) : statement(p_float64), f64_val(val) {}

    cuda_ast::function_call::function_call(primitive_t type, const std::string &name,
                                           const std::vector<cuda_ast::statement_ptr> &arguments) : statement(type),
                                                                                                    name(name),
                                                                                                    arguments(
                                                                                                            arguments) {}

    cuda_ast::for_loop::for_loop(statement_ptr initialization, cuda_ast::statement_ptr condition,
                                 cuda_ast::statement_ptr incrementer, statement_ptr body) : statement(p_none),
                                                                                            initial_value(
                                                                                                    initialization),
                                                                                            condition(condition),
                                                                                            incrementer(incrementer),
                                                                                            body(body) {}

    cuda_ast::block::block() : statement(p_none) {}

    cuda_ast::block::~block() = default;

    cuda_ast::if_condition::if_condition(cuda_ast::statement_ptr condition, statement_ptr then_body,
                                         statement_ptr else_body) : statement(p_none), condition(condition),
                                                                    then_body(then_body), has_else(true),
                                                                    else_body(else_body) {}
    cuda_ast::if_condition::if_condition(cuda_ast::statement_ptr condition, statement_ptr then_body)
                                                                  : statement(p_none), condition(condition),
                                                                    then_body(then_body), has_else(false) {}


    void cuda_ast::block::add_statement(statement_ptr stmt) {
        elements.push_back(stmt);
    }

    cuda_ast::buffer_access::buffer_access(cuda_ast::buffer_ptr accessed,
                                           const std::vector<cuda_ast::statement_ptr> &access) : statement(
            accessed->get_type()),
                                                                                                 accessed(accessed),
                                                                                                 access(access) {}

    cuda_ast::op::op(primitive_t type, const std::vector<statement_ptr> &operands) : statement(type),
                                                                                     m_operands(operands) {}

    cuda_ast::unary::unary(primitive_t type, statement_ptr operand, std::string &&op_symbol) : op(type, {operand}),
                                                                                               m_op_symbol(op_symbol) {}

    cuda_ast::binary::binary(primitive_t type, statement_ptr operand_1, statement_ptr operand_2,
                             std::string &&op_symbol)
            : op(type, {operand_1, operand_2}), m_op_symbol(op_symbol) {}

    cuda_ast::ternary::ternary(primitive_t type, statement_ptr operand_1, statement_ptr operand_2,
                               statement_ptr operand_3,
                               std::string &&op_symbol_1, std::string &&op_symbol_2) : op(type, {operand_1, operand_2,
                                                                                                 operand_3}),
                                                                                       m_op_symbol_1(op_symbol_1),
                                                                                       m_op_symbol_2(op_symbol_2) {}


    cuda_ast::declaration::declaration(assignment_ptr asgmnt) : statement(p_none), is_initialized(true),
                                                                asgmnt(asgmnt) {}

    cuda_ast::declaration::declaration(abstract_identifier_ptr identifier) : statement(p_none), is_initialized(false),
                                                                             id(identifier) {}

    primitive_t cuda_ast::statement::get_type() const {
        return type;
    }

    cuda_ast::assignment::assignment(primitive_t type) : cuda_ast::statement(type) {}

    cuda_ast::buffer_assignment::buffer_assignment(cuda_ast::buffer_ptr buffer, statement_ptr index_access,
                                                   statement_ptr rhs) : assignment(buffer->get_type()),
                                                                        m_buffer(buffer), m_index_access(index_access),
                                                                        m_rhs(rhs) {}

    cuda_ast::scalar_assignment::scalar_assignment(cuda_ast::scalar_ptr scalar, statement_ptr rhs) : assignment(
            scalar->get_type()), m_scalar(scalar), m_rhs(rhs) {}

    bool cuda_ast::op_data_t::operator==(const cuda_ast::op_data_t &rhs) const {
        return infix == rhs.infix &&
               arity == rhs.arity &&
               symbol == rhs.symbol &&
               next_symbol == rhs.next_symbol;
    }

    bool cuda_ast::op_data_t::operator!=(const cuda_ast::op_data_t &rhs) const {
        return !(rhs == *this);
    }

    std::string cuda_ast::statement::print() {
        std::stringstream ss;
        print(ss, "");
        return ss.str();
    }

    void cuda_ast::statement::print_body(std::stringstream &ss, const std::string &base) {
//        ss << base << "\t";
//        this->print(ss, base + "\t");
        ss << base << "{\n";
        ss << base << "\t";
        this->print(ss, base + "\t");
        ss << ";\n";
        ss << base << "}";
    }

    void cuda_ast::block::print_body(std::stringstream &ss, const std::string &base) {
        ss << base << "{\n";
        ss << base << "\t";
        this->print(ss, base + "\t");
        ss << ";";
        ss << "\n";
        ss << base << "}";
    }

    void cuda_ast::block::print(std::stringstream &ss, const std::string &base) {
        for (int i = 0; i < elements.size();) {
            if (i != 0)
                ss << base;
            elements[i]->print(ss, base);
            if (++i < elements.size())
                ss << ";\n";
        }
    }


    void cuda_ast::scalar::print(std::stringstream &ss, const std::string &base) {std::string name;
        statement_ptr body;
        ss << get_name();
    }

    void cuda_ast::value::print(std::stringstream &ss, const std::string &base) {
        switch (get_type())
        {
            case p_uint8:
            ss << +u8_val;
                break;
            case p_int8:
                ss << +i8_val;
                break;
            case p_uint16:
                ss << +u16_val;
                break;
            case p_int16:
                ss << +i16_val;
                break;
            case p_uint32:
                ss << +u32_val;
                break;
            case p_int32:
                ss << +i32_val;
                break;
            case p_uint64:
                ss << +u64_val;
                break;
            case p_int64:
                ss << +i64_val;
                break;
            case p_float32:
                ss << +f32_val;
                break;
            case p_float64:
                ss << +f64_val;
                break;
            default:
                assert(!"Value type not supported.");
        }
    }

    void cuda_ast::scalar_assignment::print(std::stringstream &ss, const std::string &base) {
        ss << m_scalar->get_name() << " = ";
        m_rhs->print(ss, base);
    }

    void cuda_ast::buffer_assignment::print(std::stringstream &ss, const std::string &base) {
        ss << m_buffer->get_name();
        if (m_buffer->get_location() != memory_location::reg) {
            ss << "[";
            m_index_access->print(ss, base);
            ss << "]";
        }
        ss << " = ";
        m_rhs->print(ss, base);
    }

    void cuda_ast::function_call::print(std::stringstream &ss, const std::string &base) {
        ss << name << "(";
        int i = 0;
        while (i < arguments.size()) {
            arguments[i]->print(ss, base);
            i++;
            if (i < arguments.size()) {
                ss << ", ";
            }
        }
        ss << ")";
    }

    void cuda_ast::for_loop::print(std::stringstream &ss, const std::string &base) {
        ss << "for (";
        initial_value->print(ss, base);
        ss << "; ";
        condition->print(ss, base);
        ss << "; ";
        incrementer->print(ss, base);
        ss << ")\n";
        body->print_body(ss, base);
    }

    void cuda_ast::if_condition::print(std::stringstream &ss, const std::string &base) {
        ss << "if (";
        condition->print(ss, base);
        ss << ")\n";
        then_body->print_body(ss, base);
        if (has_else) {
            ss << "\n" << base << "else\n";
            else_body->print_body(ss, base);
        }
    }

    void cuda_ast::buffer_access::print(std::stringstream &ss, const std::string &base) {
        ss << accessed->get_name();
        if (accessed->get_location() != memory_location::reg) {
            ss << "[";
            int i = 0;
            while (i < access.size()) {
                access[i]->print(ss, base);
                i++;
                if (i < access.size()) {
                    ss << ", ";
                }
            }
            ss << "]";
        }
    }

    void cuda_ast::unary::print(std::stringstream &ss, const std::string &base) {
        ss << m_op_symbol;
        m_operands[0]->print(ss, base);
    }

    void cuda_ast::binary::print(std::stringstream &ss, const std::string &base) {
        ss << "(";
        m_operands[0]->print(ss, base);
        ss << " " << m_op_symbol << " ";
        m_operands[1]->print(ss, base);
        ss << ")";

    }

    void cuda_ast::ternary::print(std::stringstream &ss, const std::string &base) {
        ss << "(";
        m_operands[0]->print(ss, base);
        ss << " " << m_op_symbol_1 << " ";
        m_operands[1]->print(ss, base);
        ss << " " << m_op_symbol_2 << " ";
        m_operands[2]->print(ss, base);
        ss << ")";

    }

    void cuda_ast::buffer::print(std::stringstream &ss, const std::string &base) {
        ss << get_name();
    }

    void cuda_ast::declaration::print(std::stringstream &ss, const std::string &base) {
        if (is_initialized) {
            asgmnt->print_declaration(ss, base);
        } else {
            switch (id->get_location()) {
                case cuda_ast::memory_location::shared:
                    ss << "__shared__ ";
                    break;
                case cuda_ast::memory_location::constant:
                    ss << "__constant__ ";
                    break;
                default:
                    break;
            }
            id->print_declaration(ss, base);
        }

    }

    void cuda_ast::cast::print(std::stringstream &ss, const std::string &base) {
        ss << "((" << tiramisu_type_to_cuda_type(get_type()) << ") ";
        to_be_cast->print(ss, base);
        ss << ")";
    }

    void cuda_ast::buffer::print_declaration(std::stringstream &ss, const std::string &base) {
        ss << tiramisu_type_to_cuda_type(get_type()) << " ";
        if (this->get_location() == memory_location::global || this->get_location() == memory_location::host)
        {
            ss << "*" << get_name();
        } else if (this->get_location() == memory_location::reg) {
            ss << get_name();
        } else {
            ss << get_name() << "[";
            print_size(ss, base, " * ");
            ss << "]";
        }
    }

    void cuda_ast::buffer::print_size(std::stringstream &ss, const std::string &base, const std::string &seperator) {
        for (int i = 0; i < size.size();) {
            size[i]->print(ss, base);
            if (++i < size.size())
                ss << seperator;
        }

    }

    void cuda_ast::scalar::print_declaration(std::stringstream &ss, const std::string &base) {
        if (is_const)
            ss << "const ";
        ss << tiramisu_type_to_cuda_type(get_type()) << " ";
        print(ss, base);
    }

    cuda_ast::sync::sync() : statement(p_none) {}

    void cuda_ast::sync::print(std::stringstream &ss, const std::string &base) {
        ss << "__syncthreads()";
    }

    cuda_ast::kernel::dim3d_t::dim3d_t() {
        x = statement_ptr{new cuda_ast::value{value_cast(global::get_loop_iterator_data_type(), 1)}};
        y = statement_ptr{new cuda_ast::value{value_cast(global::get_loop_iterator_data_type(), 1)}};
        z = statement_ptr{new cuda_ast::value{value_cast(global::get_loop_iterator_data_type(), 1)}};
    }


    void cuda_ast::kernel::dim3d_t::set(gpu_iterator::dimension_t dim, statement_ptr size) {
        switch (dim)
        {
            case gpu_iterator::dimension_t::x:
                this->x.swap(size);
                break;
            case gpu_iterator::dimension_t::y:
                this->y.swap(size);
                break;
            case gpu_iterator::dimension_t::z:
                this->z.swap(size);
                break;
        }
    }


    cuda_ast::kernel::kernel() : kernel_number(kernel_count++) {}

    void cuda_ast::kernel::set_dimension(gpu_iterator dimension){
        if (dimension.type == gpu_iterator::type_t::BLOCK)
            block_dimensions.set(dimension.dimension, dimension.size);
        else
            thread_dimensions.set(dimension.dimension, dimension.size);
    }

    void cuda_ast::kernel::set_body(statement_ptr body) {
        this->body = body;
    }

    std::string cuda_ast::kernel::get_name() const {
        return "_kernel_" + std::to_string(this->kernel_number);
    }

    std::string cuda_ast::kernel::get_wrapper_name() const {
        return this->get_name() + "_wrapper";
    }

    cuda_ast::kernel_call::kernel_call(kernel_ptr kernel) : statement(p_none), kernel(kernel){}
    void cuda_ast::kernel_call::print(std::stringstream &ss, const std::string &base) {
        ss << "{\n";
        auto new_base = base + "\t";
        ss << new_base << "dim3 blocks(";
        kernel->block_dimensions.x->print(ss, base);
        ss << ", ";
        kernel->block_dimensions.y->print(ss, base);
        ss << ", ";
        kernel->block_dimensions.z->print(ss, base);
        ss << ");\n";
        ss << new_base << "dim3 threads(";
        kernel->thread_dimensions.x->print(ss, base);
        ss << ", ";
        kernel->thread_dimensions.y->print(ss, base);
        ss << ", ";
        kernel->thread_dimensions.z->print(ss, base);
        ss << ");\n";
        ss << new_base << kernel->get_name() << "<<<blocks, threads>>>(";
        std::vector<abstract_identifier_ptr> arguments;
        for (auto &c: kernel->used_constants)
            arguments.push_back(c.second);
        for (auto &b: kernel->used_buffers)
            arguments.push_back(b.second);
        for (auto it = arguments.begin(); it != arguments.end();)
        {
            (*it)->print(ss, base);
            if (++it != arguments.end())
            {
                ss << ", ";
            }
        }
        ss << ");\n" << base << "}";
    }

    int cuda_ast::kernel::kernel_count = 0;

    cuda_ast::kernel_definition::kernel_definition(kernel_ptr kernel) : statement(p_none), kernel(kernel){}

    std::vector<cuda_ast::abstract_identifier_ptr> cuda_ast::kernel::get_arguments() {
        std::vector<abstract_identifier_ptr> arguments;
        for (auto &c: used_constants)
            arguments.push_back(c.second);
        for (auto &b: used_buffers)
            arguments.push_back(b.second);
        return arguments;
    }



    void cuda_ast::kernel_definition::print(std::stringstream &ss, const std::string &base) {
        ss << "static __global__ void " << kernel->get_name() << "(";
        auto arguments = kernel->get_arguments();
        for (auto it = arguments.begin(); it != arguments.end();)
        {
            (*it)->print_declaration(ss, base);
            if (++it != arguments.end())
            {
                ss << ", ";
            }
        }
        ss<<")\n" << base;
        kernel->body->print_body(ss, base);
    }


    cuda_ast::gpu_iterator_read::gpu_iterator_read(gpu_iterator it) : statement(global::get_loop_iterator_data_type()), it(it), simplified(true){}
    cuda_ast::gpu_iterator_read::gpu_iterator_read(gpu_iterator it, bool simplified) : statement(global::get_loop_iterator_data_type()), it(it), simplified(simplified){}

    void cuda_ast::gpu_iterator_read::print(std::stringstream &ss, const std::string &base) {
        if (simplified)
        {
            ss << it.simplified_name();
        }
        else {
            switch (it.type) {
                case gpu_iterator::type_t::BLOCK:
                    ss << "blockIdx";
                    break;
                case gpu_iterator::type_t::THREAD:
                    ss << "threadIdx";
                    break;
            }
            ss << '.';
            switch (it.dimension) {
                case gpu_iterator::dimension_t::x:
                    ss << 'x';
                    break;
                case gpu_iterator::dimension_t::y:
                    ss << 'y';
                    break;
                case gpu_iterator::dimension_t::z:
                    ss << 'z';
                    break;
            }
        }
    }

    void cuda_ast::kernel::add_used_scalar(scalar_ptr scalar) {
        used_constants[scalar->get_name()] = scalar;
    }

    void cuda_ast::kernel::add_used_buffer(buffer_ptr buffer) {
        if (buffer->get_location() != memory_location::shared
                && buffer->get_location() != memory_location::local
                && buffer->get_location() != memory_location::constant) {
            used_buffers[buffer->get_name()] = buffer;
        }
    }

    cuda_ast::host_function::host_function(primitive_t type, std::string name, const std::vector<abstract_identifier_ptr> &arguments, statement_ptr body) :
            statement(type), name(name), body(body), arguments(arguments){}

    void cuda_ast::host_function::print(std::stringstream &ss, const std::string &base) {
        ss << "extern \"C\" " << tiramisu_type_to_cuda_type(this->get_type());
        if (pointer_return)
            ss << "*";
        ss << " " << name << "(";
        for (int i = 0; i < arguments.size();)
        {
            arguments[i]->print_declaration(ss, base);
            if (++i < arguments.size())
            {
                ss << ", ";
            }
        }
        ss << ")\n";
        body->print_body(ss, base);
    }

    cuda_ast::memcpy::memcpy(buffer_ptr from, buffer_ptr to) : statement(p_none), from(from), to(to) {}
    void cuda_ast::memcpy::print(std::stringstream &ss, const std::string &base)
    {
        ss << "cudaMemcpy(" << to->get_name() << ", ";
        ss << from->get_name() << ", ";
        to->print_size(ss, base, " * ");
        ss << " * sizeof(" << tiramisu_type_to_cuda_type(to->get_type()) << "), ";
        if (from->get_location() == memory_location::host)
            ss << "cudaMemcpyHostToDevice";
        else
            ss << "cudaMemcpyDeviceToHost";
        ss << ")";
    }

    cuda_ast::allocate::allocate(buffer_ptr b) : statement(p_none), b(b){}
    cuda_ast::free::free(buffer_ptr b) : statement(p_none), b(b){}

    void cuda_ast::allocate::print(std::stringstream &ss, const std::string &base) {
        switch(b->get_location())
        {
            case memory_location::host:
                ss << b->get_name() << " = (" << tiramisu_type_to_cuda_type(b->get_type()) << "*)malloc(";
                break;
            case memory_location::global:
                ss << "cudaMalloc(&" << b->get_name() << ", ";
                break;
            default:
                assert(!"Can only allocate global or host memory");
        }
        b->print_size(ss, base, " * ");
        ss << " * sizeof(" << tiramisu_type_to_cuda_type(b->get_type()) << "))";
    }
    void cuda_ast::free::print(std::stringstream &ss, const std::string &base) {
        switch(b->get_location())
        {
            case memory_location::host:
                ss << "free";
                break;
            case memory_location::global:
                ss << "cudaFree";
                break;
            default:
                assert(!"Can only free global or host memory");
        }
        ss << "(";
        b->print(ss, base);
        ss << ")";
    }

    std::pair<cuda_ast::statement_ptr, cuda_ast::statement_ptr> cuda_ast::statement::extract_min_cap() {
        return std::make_pair(statement_ptr{}, statement_ptr{});
    }
    std::pair<cuda_ast::statement_ptr, cuda_ast::statement_ptr> cuda_ast::function_call::extract_min_cap() {
        if (this->name == "min")
            return std::make_pair(this->arguments[0], this->arguments[1]);
        return std::make_pair(statement_ptr{}, statement_ptr{});
    }

    cuda_ast::statement_ptr cuda_ast::cast::replace_iterators(
            std::unordered_map<std::string, gpu_iterator> &iterators) {
        auto replaced = to_be_cast->replace_iterators(iterators);
        return statement_ptr{new cast{get_type(), replaced}};
    }

    std::unordered_set<std::string> cuda_ast::cast::extract_scalars() {
        return to_be_cast->extract_scalars();
    }

    cuda_ast::statement_ptr cuda_ast::scalar::replace_iterators(
            std::unordered_map<std::string, gpu_iterator> &iterators) {
        auto it = iterators.find(get_name());
        if (it != iterators.end())
            return statement_ptr{new gpu_iterator_read{it->second}};
        return statement_ptr{new scalar{get_type(), get_name(), get_location()}};
    }

    std::unordered_set<std::string> cuda_ast::scalar::extract_scalars() {
        return std::unordered_set<std::string>{get_name()};
    }

    cuda_ast::statement_ptr cuda_ast::function_call::replace_iterators(
            std::unordered_map<std::string, gpu_iterator> &iterators) {
        std::vector<statement_ptr> new_args;
        for (auto &arg: arguments)
            new_args.push_back(arg->replace_iterators(iterators));
        return statement_ptr{new function_call{get_type(), name, new_args}};
    }

    std::unordered_set<std::string> cuda_ast::function_call::extract_scalars() {
        std::unordered_set<std::string> result;
        for (auto &arg: arguments)
        {
            auto subresult = arg->extract_scalars();
            result.insert(subresult.begin(), subresult.end());
        }
        return result;
    }


    cuda_ast::statement_ptr cuda_ast::unary::replace_iterators(
            std::unordered_map<std::string, gpu_iterator> &iterators) {
        return statement_ptr{new unary{get_type(), m_operands[0]->replace_iterators(iterators), std::string{m_op_symbol}}};
    }


    cuda_ast::statement_ptr cuda_ast::binary::replace_iterators(
            std::unordered_map<std::string, gpu_iterator> &iterators) {
        return statement_ptr{new binary{get_type(), m_operands[0]->replace_iterators(iterators), m_operands[1]->replace_iterators(iterators), std::string{m_op_symbol}}};
    }

    cuda_ast::statement_ptr cuda_ast::ternary::replace_iterators(
            std::unordered_map<std::string, gpu_iterator> &iterators) {
        return statement_ptr{new ternary{get_type(), m_operands[0]->replace_iterators(iterators), m_operands[1]->replace_iterators(iterators),
                                         m_operands[2]->replace_iterators(iterators), std::string{m_op_symbol_1}, std::string{m_op_symbol_2}}};
    }

    std::unordered_set<std::string> cuda_ast::op::extract_scalars() {
        std::unordered_set<std::string> result;
        for (auto &op: m_operands)
        {
            auto subresult = op->extract_scalars();
            result.insert(subresult.begin(), subresult.end());
        }
        return result;
    }

    cuda_ast::statement_ptr cuda_ast::statement::replace_iterators(
            std::unordered_map<std::string, gpu_iterator> &iterators) {
        assert(!"replace_iterators not fully supported");
        return statement_ptr{};
    }

    std::unordered_set<std::string> cuda_ast::statement::extract_scalars() {
        return std::unordered_set<std::string>{};
    }

    cuda_ast::value_ptr cuda_ast::value::copy() {
        switch (get_type())
        {
            case p_uint8:
                return value_ptr{new value{u8_val}};
                break;
            case p_int8:
                return value_ptr{new value{i8_val}};
                break;
            case p_uint16:
                return value_ptr{new value{u16_val}};
                break;
            case p_int16:
                return value_ptr{new value{i16_val}};
                break;
            case p_uint32:
                return value_ptr{new value{u32_val}};
                break;
            case p_int32:
                return value_ptr{new value{i32_val}};
                break;
            case p_uint64:
                return value_ptr{new value{u64_val}};
                break;
            case p_int64:
                return value_ptr{new value{i64_val}};
                break;
            case p_float32:
                return value_ptr{new value{f32_val}};
                break;
            case p_float64:
                return value_ptr{new value{f64_val}};
                break;
            default:
                assert(!"Value type not supported.");
        }
    }
    cuda_ast::statement_ptr cuda_ast::value::replace_iterators(
            std::unordered_map<std::string, gpu_iterator> &iterators) {
        return this->copy();
    }

    cuda_ast::statement_ptr cuda_ast::buffer_access::replace_iterators(
            std::unordered_map<std::string, gpu_iterator> &iterators) {
        std::vector<cuda_ast::statement_ptr> new_access;
        for (auto &a: access)
            new_access.push_back(a->replace_iterators(iterators));
        return statement_ptr{new buffer_access{accessed, new_access}};
    }

    std::unordered_set<std::string> cuda_ast::buffer_access::extract_scalars() {
        std::unordered_set<std::string> result;
        for (auto &index: this->access)
        {
            auto subresult = index->extract_scalars();
            result.insert(subresult.begin(), subresult.end());
        }
        return result;

    }


    std::string cuda_ast::gpu_iterator::simplified_name() {
        std::string result = "__";
        switch (type)
        {
            case type_t::THREAD :
                result += "t";
                break;
            case type_t::BLOCK :
                result += "b";
                break;
        }
        switch (dimension)
        {
            case dimension_t::x :
                result += "x";
                break;
            case dimension_t::y :
                result += "y";
                break;
            case dimension_t::z :
                result += "z";
                break;
        }
        return result + "__";
    }

    void cuda_ast::assignment::print_declaration(std::stringstream &ss, const std::string &base) {
        print(ss, base);
    }

    void cuda_ast::scalar_assignment::print_declaration(std::stringstream &ss, const std::string &base) {
        m_scalar->print_declaration(ss, base);
        ss << " = ";
        m_rhs->print(ss, base);
    }

    cuda_ast::statement_ptr cuda_ast::generator::get_scalar_from_name(std::string name)
    {
        auto it = this->gpu_iterators.find(name);
        if (it != this->gpu_iterators.end())
        {
            return statement_ptr {new cuda_ast::gpu_iterator_read{it->second}};
        } else {
            auto data_it = m_scalar_data.find(name);
            assert(data_it != m_scalar_data.end() && "Unknown name");
            auto const &data = data_it->second;
            scalar_ptr used_scalar{new cuda_ast::scalar{data.first, name, data.second}};
            if (this->in_kernel)
            {
                // Check if it was defined inside
                bool defined_inside = gpu_local.find(name) != gpu_local.end();
                if (!defined_inside)
                {
                    for (auto it = iterator_stack.rbegin(); it != iterator_stack.rend(); it++)
                    {
                        if (name == *it)
                        {
                            defined_inside = true;
                            break;
                        }
                        if (gpu_iterators.find(name) != gpu_iterators.end())
                        {
                            break;
                        }
                    }
                }
                if (!defined_inside)
                    this->current_kernel->add_used_scalar(used_scalar);
            }
            return used_scalar;
        }
    }

    cuda_ast::value::value(const tiramisu::expr &expr) : statement(expr.get_data_type()) {
        switch (get_type())
        {
            case p_uint8:
                u8_val = expr.get_uint8_value();
                break;
            case p_int8:
                i8_val = expr.get_int8_value();
                break;
            case p_uint16:
                u16_val = expr.get_uint16_value();
                break;
            case p_int16:
                i16_val = expr.get_int16_value();
                break;
            case p_uint32:
                u32_val = expr.get_uint32_value();
                break;
            case p_int32:
                i32_val = expr.get_int32_value();
                break;
            case p_uint64:
                u64_val = expr.get_uint64_value();
                break;
            case p_int64:
                i64_val = expr.get_int64_value();
                break;
            case p_float32:
                f32_val = expr.get_float32_value();
                break;
            case p_float64:
                f64_val = expr.get_float64_value();
                break;
            default:
                assert(!"Value type not supported.");
        }

    }

    cuda_ast::compiler::compiler(const std::string &code) : code(code){}

    std::string cuda_ast::compiler::get_cpu_obj(const std::string &obj_name) const {
        return obj_name + "_cpu.o";
    }

    std::string cuda_ast::compiler::get_gpu_obj(const std::string &obj_name) const {
        return obj_name + "_gpu.o";
    }

    bool cuda_ast::exec_result::succeed() {
        return exec_succeeded && result == 0;
    }

    bool cuda_ast::exec_result::fail() {
        return !succeed();
    }

    cuda_ast::exec_result cuda_ast::compiler::exec(const std::string &cmd) {
        // TODO support Windows, stderr
        using namespace std;
        stringstream out, err;
        array<char, 128> buffer;
        FILE * file{popen(cmd.c_str(), "r")};
        if (!file)
            return exec_result{false, -1, "", ""};
        while (!feof(file))
        {
            if (fgets(buffer.data(), 128, file) != nullptr)
                out << buffer.data();
        }

        return exec_result{true, pclose(file), out.str(), err.str()};
    }

    bool cuda_ast::compiler::compile_cpu_obj(const std::string &filename, const std::string &obj_name) const {
        using namespace std;
        DEBUG_FCT_NAME(3);
        DEBUG_INDENT(4);

        stringstream command;
        command << NVCC_PATH;
        // Say that this is actually cuda code
        command << " -x cu";
        // Create a .o file
        command << " -c";
        // Export device code
        command << " -dc";
        // Specify input file name
        command << " " << filename;
        // Specify output name
        command << " -o " << get_cpu_obj(obj_name);

        DEBUG(3, cout << "The command to compile the CPU object is : " << command.str());

        auto result = exec(command.str());

        if (result.fail())
        {
            ERROR("Failed to compile the CPU object for the GPU code.", false);
        }

        DEBUG_INDENT(-4);

        return result.succeed();
    }

    bool cuda_ast::compiler::compile_gpu_obj(const std::string &obj_name) const
    {
        using namespace std;
        DEBUG_FCT_NAME(3);
        DEBUG_INDENT(4);

        stringstream command;
        command << NVCC_PATH;
        // Link device object code
        command << " -dlink";
        // Specify input file name
        command << " " << get_cpu_obj(obj_name);
        // Specify output name
        command << " -o " << get_gpu_obj(obj_name);

        DEBUG(3, cout << "The command to compile the GPU object is : " << command.str());

        auto result = exec(command.str());

        if (result.fail())
        {
            ERROR("Failed to compile the GPU object for the GPU code.", false);
        }

        DEBUG_INDENT(-4);

        return result.succeed();
    }

    bool cuda_ast::compiler::compile(const std::string & obj_name) const {
        using namespace std;
        DEBUG_FCT_NAME(3);
        DEBUG_INDENT(4);
        char cwd[500];
        getcwd(cwd, 500);
        cout << cwd << endl;
        ofstream code_file;
        string filename = obj_name + ".cu";
        if (!getenv("CUDA_NO_OVERWRITE")) {
            code_file.open(filename, fstream::out | fstream::trunc);
            if (code_file.fail()) {
                ERROR("Failed to open file " + filename + " for writing. Cannot compile GPU code.", false);
                return false;
            }
            DEBUG(3, cout << "Opened file " << filename << " for writing.");
            code_file << "#include <stdint.h>\n";
            code_file << code;
            code_file.flush();
            if (code_file.fail()) {
                ERROR("Failed to write to file " + filename + ". Cannot compile GPU code.", false);
                return false;
            }
        }

        DEBUG_INDENT(-4);

        return compile_cpu_obj(filename, obj_name) && compile_gpu_obj(obj_name);
    }

    bool cuda_ast::abstract_identifier::is_buffer() const {
        return false;
    }

    bool cuda_ast::buffer::is_buffer() const {
        return true;
    }

    cuda_ast::return_statement::return_statement(statement_ptr return_value) : statement(p_none), return_value(return_value){}

    void cuda_ast::return_statement::print(std::stringstream &ss, const std::string &base) {
        ss << "return ";
        return_value->print(ss, base);
    }

    void cuda_ast::host_function::set_pointer_return(bool val) {
        pointer_return = val;
    }

};

