#include <iostream>
#include <memory>

#include <cg/FPS_camera.h>
#include <data/mesh.h>
#include <ray_tracer/lbvh.h>
#include <ray_tracer/ray_tracer.h>

#include <cmdparser.hpp>

static kouek::FPSCamera camera;
static kouek::RayTracer::RayTracer renderer;

int main(int argc, char **argv) {
    auto cmdParser = cli::Parser(argc, argv);
    cmdParser.set_required<std::string>("obj", "obj-file-path",
                                        "The path of the directory of the scene.");
    cmdParser.run_and_exit_if_error();

    auto mesh = kouek::Data::OBJMesh();
    mesh.LoadFromFile(cmdParser.get<std::string>("obj"));
    if (!mesh.IsComplete())
        return 1;
    std::cout << std::format("Load mesh with {} vertices, {} faces.\n", mesh.GetPositions().size(),
                             mesh.GetFacePositionIndices().size());

    auto lbvh = std::make_shared<kouek::RayTracer::LBVH>();
    lbvh->BuildFrom(
        {.positions = mesh.GetPositions(), .facePositionIndices = mesh.GetFacePositionIndices()});

    return 0;
}
