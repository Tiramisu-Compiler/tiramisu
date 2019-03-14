#include "wrapper_cvtcolordist.h"
#include "../benchmarks.h"
#include "../../include/tiramisu/mpi_comm.h"

#include "halide_image_io.h"

int main(int, char**) {
    int rank = tiramisu_MPI_init();
    std::vector<std::chrono::duration<double,std::milli>> duration_vector_1;
    std::vector<std::chrono::duration<double,std::milli>> duration_vector_2;

    // We will mimic each node starting with its chunk of data by loading the whole image on each node and then just
    // processing on the node's specific portion
    std::string img = "/utils/images/rgb.png";
    Halide::Buffer<uint8_t> input = Halide::Tools::load_image(std::getenv("TIRAMISU") + img);
    Halide::Buffer<uint8_t> output_ref(input.width(), input.height());

    // Create the node-specific buffers
    assert(input.height() % NODES == 0); // Make things simpler so we don't have to worry about edge cases
    int rows_per_node = input.height() / NODES;
    Halide::Buffer<uint8_t> node_input(input.width(), rows_per_node, input.channels());
    Halide::Buffer<uint8_t> node_output(input.width(), rows_per_node);
    int start_row = rank * rows_per_node;
    int end_row = (rank + 1) * rows_per_node;
    for (int chan = 0; chan < input.channels(); chan++) {
        for (int c = 0; c < input.width(); c++) {
            for (int r = start_row; r < end_row; r++) {
                node_input(c, r - start_row, chan) = input(c, r, chan);
            }
        }
    }
    // Warm up code.
    cvtcolordist_tiramisu(node_input.raw_buffer(), node_output.raw_buffer());
    std::cerr << "rank: " << rank << std::endl;
    // write out results from each rank here
    store_dist_results<uint8_t, int32_t /*type to cast result to*/>("benchmark_cvtcolordist", rank, node_output);
    if (rank == 0) {
        // only the 0th rank will run the full reference code
        cvtcolordist_ref(input.raw_buffer(), output_ref.raw_buffer());
    }

    // Tiramisu
    for (int i=0; i<NB_TESTS; i++)
    {
        MPI_Barrier(MPI_COMM_WORLD);
        auto start1 = std::chrono::high_resolution_clock::now();
        cvtcolordist_tiramisu(node_input.raw_buffer(), node_output.raw_buffer());
        auto end1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double,std::milli> duration1 = end1 - start1;
        duration_vector_1.push_back(duration1);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    // Reference
    if (rank == 0) {
        for (int i = 0; i < NB_TESTS; i++) {
            auto start2 = std::chrono::high_resolution_clock::now();
            cvtcolordist_ref(input.raw_buffer(), output_ref.raw_buffer());
            auto end2 = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> duration2 = end2 - start2;
            duration_vector_2.push_back(duration2);
        }
    }

    if (rank == 0) {
        print_time("performance_CPU.csv", "cvtcolor",
                   {"Tiramisu", "Halide"},
                   {median(duration_vector_1), median(duration_vector_2)});
        if (CHECK_CORRECTNESS) {
            combine_dist_results("benchmark_cvtcolordist", {rows_per_node, input.width()}, NODES);
            compare_dist_buffers("benchmark_cvtcolordist", output_ref);
        }
        std::cout << "Distributed cvtcolor passed" << std::endl;
    }

    tiramisu_MPI_cleanup();
    return 0;
}
