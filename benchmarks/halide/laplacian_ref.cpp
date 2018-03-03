#include "Halide.h"
#include "halide_benchmark.h"

using namespace Halide;
using namespace Halide::Tools;

Var x("x"), y("y");

// Downsample with a 1 3 3 1 filter
Func downsample(Func f) {
    Func downx("downx"), downy("downy");
    downx(x, y, _) = (f(2*x-1, y, _) + 3.0f * (f(2*x, y, _) + f(2*x+1, y, _)) + f(2*x+2, y, _)) / 8.0f;
    downy(x, y, _) = (downx(x, 2*y-1, _) + 3.0f * (downx(x, 2*y, _) + downx(x, 2*y+1, _)) + downx(x, 2*y+2, _)) / 8.0f;
    return downy;
}

// Upsample using bilinear interpolation
Func upsample(Func f) {
    Func upx("upx"), upy("upy");
    upx(x, y, _) = 0.25f * f((x/2) - 1 + 2*(x % 2), y, _) + 0.75f * f(x/2, y, _);
    upy(x, y, _) = 0.25f * upx(x, (y/2) - 1 + 2*(y % 2), _) + 0.75f * upx(x, y/2, _);
    return upy;
}

int main(int argc, char **argv) {
    /* THE ALGORITHM */

    // Number of pyramid levels
    int J = 8;
    const int maxJ = 20;

    ImageParam input(UInt(16), 3, "input");
    Param<int> levels;
    Param<float> alpha;
    Param<float> beta;

    // loop variables
    Var c("c"), k("k");

    // Make the remapping function as a lookup table.
    Func remap("remap");
    Expr fx = cast<float>(x) / 256.0f;
    remap(x) = alpha*fx*exp(-fx*fx/2.0f);

    // Set a boundary condition
    Func clamped = BoundaryConditions::repeat_edge(input);

    // Convert to floating point
    Func floating("floating");
    floating(x, y, c) = clamped(x, y, c) / 65535.0f;

    // Get the luminance channel
    Func gray("gray");
    gray(x, y) = 0.299f * floating(x, y, 0) + 0.587f * floating(x, y, 1) + 0.114f * floating(x, y, 2);

    // Make the processed Gaussian pyramid.
    std::vector<Func> gPyramid;
    for (int i = 0; i < maxJ; i++) {
        Func gP("gPyramid_" + std::to_string(i));
        gPyramid.push_back(gP);
    }
    // Do a lookup into a lut with 256 entires per intensity level
    Expr level = k * (1.0f / (levels - 1));
    Expr idx = gray(x, y)*cast<float>(levels-1)*256.0f;
    idx = clamp(cast<int>(idx), 0, (levels-1)*256);
    gPyramid[0](x, y, k) = beta*(gray(x, y) - level) + level + remap(idx - 256*k);

    for (int j = 1; j < J; j++) {
        Func down = downsample(gPyramid[j-1]);
        gPyramid[j](x, y, k) = down(x, y, k);
    }

    // Get its laplacian pyramid
    std::vector<Func> lPyramid;
    for (int i = 0; i < maxJ; i++) {
        Func lP("lPyramid_" + std::to_string(i));
        lPyramid.push_back(lP);
    }
    lPyramid[J-1](x, y, k) = gPyramid[J-1](x, y, k);

    for (int j = J-2; j >= 0; j--) {
        Func up = upsample(gPyramid[j+1]);
        lPyramid[j](x, y, k) = gPyramid[j](x, y, k) - up(x, y, k);
    }

    // Make the Gaussian pyramid of the input
    std::vector<Func> inGPyramid;
    for (int i = 0; i < maxJ; i++) {
        Func inGP("inGPyramid_" + std::to_string(i));
        inGPyramid.push_back(inGP);
    }

    inGPyramid[0](x, y) = gray(x, y);
    for (int j = 1; j < J; j++) {
        inGPyramid[j](x, y) = downsample(inGPyramid[j-1])(x, y);
    }

    // Make the laplacian pyramid of the output
    std::vector<Func> outLPyramid;
    for (int i = 0; i < maxJ; i++) {
        Func outLP("outLPyramid_" + std::to_string(i));
        outLPyramid.push_back(outLP);
    }

    for (int j = 0; j < J; j++) {
        // Split input pyramid value into integer and floating parts
        Expr level = inGPyramid[j](x, y) * cast<float>(levels-1);
        Expr li = clamp(cast<int>(level), 0, levels-2);
        Expr lf = level - cast<float>(li);
        // Linearly interpolate between the nearest processed pyramid levels
        outLPyramid[j](x, y) = (1.0f - lf) * lPyramid[j](x, y, li) + lf * lPyramid[j](x, y, li+1);
    }

    // Make the Gaussian pyramid of the output
    std::vector<Func> outGPyramid;
    for (int i = 0; i < maxJ; i++) {
        Func outGP("outGPyramid_" + std::to_string(i));
        outGPyramid.push_back(outGP);
    }
    outGPyramid[J-1](x, y) = outLPyramid[J-1](x, y);
    for (int j = J-2; j >= 0; j--) {
        outGPyramid[j](x, y) = upsample(outGPyramid[j+1])(x, y) + outLPyramid[j](x, y);
    }

    // Reintroduce color (Connelly: use eps to avoid scaling up noise w/ apollo3.png input)
    Func color("color");
    float eps = 0.01f;
    color(x, y, c) = outGPyramid[0](x, y) * (floating(x, y, c)+eps) / (gray(x, y)+eps);

    Func output("local_laplacian");
    // Convert back to 16-bit
    output(x, y, c) = cast<uint16_t>(clamp(color(x, y, c), 0.0f, 1.0f) * 65535.0f);

    Target target = get_target_from_environment();
    if (target.has_gpu_feature()) {
        Var xi("xi"), yi("yi");
        output.compute_root().gpu_tile(x, y, xi, yi, 16, 8);
        for (int j = 0; j < J; j++) {
            int blockw = 16, blockh = 8;
            if (j > 3) {
                blockw = 2;
                blockh = 2;
            }
            if (j > 0) {
                inGPyramid[j].compute_root().gpu_tile(x, y, xi, yi, blockw, blockh);
                gPyramid[j].compute_root().reorder(k, x, y).gpu_tile(x, y, xi, yi, blockw, blockh);
            }
            outGPyramid[j].compute_root().gpu_tile(x, y, xi, yi, blockw, blockh);
        }
    } else {
        // CPU schedule
        Var yi("yi");
        output.parallel(y, 32).vectorize(x, 8);
        gray.compute_root().parallel(y, 32).vectorize(x, 8);
        for (int j = 0; j < J; j++) {
            if (j > 0) {
                inGPyramid[j].compute_root()
                    .parallel(y, 32)
                    .vectorize(x, 8);
                gPyramid[j].compute_root()
                    .reorder(k, y)
                    .parallel(y, 8)
                    .vectorize(x, 8);
                    //.reorder_storage(x, k, y);
            }
            outGPyramid[j].compute_root().parallel(y, 32).vectorize(x, 8);
        }
        for (int j = 4; j < J; j++) {
            inGPyramid[j].compute_root();
            gPyramid[j].compute_root().parallel(k);
            outGPyramid[j].compute_root();
        }
    }

    output.compile_to_object("build/generated_fct_laplacian_ref.o", {input, levels, alpha, beta}, "laplacian_ref");

    output.compile_to_lowered_stmt("build/generated_fct_laplacian_ref.txt", {input, levels, alpha, beta}, Text);

    return 0;
}
