#include <ray_tracer/ray_tracer.h>

#include <format>
#include <iostream>

#include <cg/math.h>

kouek::RayTracer::RayTracer::~RayTracer() {}

void kouek::RayTracer::RayTracer::SetMesh(const InputMesh &inputMesh) {
    auto clear = [&](auto &&d_v) {
        d_v.clear();
        d_v.shrink_to_fit();
    };
    clear(d_triangles);
    clear(d_triToPositionIndices);
    clear(d_positions);
    clear(d_normals);
    clear(d_texCoords);
    clear(d_lights);
    clear(d_materials);

    d_triangles.resize(inputMesh.facePositionIndices.size());
#define CPY_AOS_TO_SOA(AoS, d_SoA, SoAMember)                                                      \
    {                                                                                              \
        thrust::device_vector<decltype(decltype(d_SoA)::value_type::SoAMember)> d_AoS = AoS;       \
        thrust::for_each(thrust::make_counting_iterator(IndexTy(0)),                               \
                         thrust::make_counting_iterator(static_cast<IndexTy>(d_SoA.size())),       \
                         [SoA = thrust::raw_pointer_cast(d_SoA.data()),                            \
                          _AoS = thrust::raw_pointer_cast(d_AoS.data())] __device__(IndexTy i) {   \
                             SoA[i].SoAMember = _AoS[i];                                           \
                         });                                                                       \
    }
    CPY_AOS_TO_SOA(inputMesh.faceNormalIndices, d_triangles, normIdx)
    CPY_AOS_TO_SOA(inputMesh.faceTexCoordIndices, d_triangles, texCoordIdx)
#undef CPY_AOS_TO_SOA
    {
        thrust::device_vector<IndexTy> d_grpStartFaceIndices = inputMesh.groupStartFaceIndices;
        thrust::for_each(thrust::make_counting_iterator(IndexTy(0)),
                         thrust::make_counting_iterator(static_cast<IndexTy>(d_triangles.size())),
                         [triangles = thrust::raw_pointer_cast(d_triangles.data()),
                          grpStartFaceIndices = thrust::raw_pointer_cast(
                              d_grpStartFaceIndices.data())] __device__(IndexTy fi) {
                             IndexTy gi = 0;
                             while (grpStartFaceIndices[gi] < fi)
                                 ++gi;

                             triangles[fi].grpIdx = gi;
                         });
    }

    d_triToPositionIndices = inputMesh.facePositionIndices;
    d_positions = inputMesh.positions;
    d_normals = inputMesh.normals;
    d_texCoords = inputMesh.texCoords;
    d_lights = inputMesh.lights;
    d_materials = inputMesh.materials;

    rndrParam.Set(&RenderParameter::lightNum, static_cast<IndexTy>(inputMesh.lights.size()));
    rndrParam.Set(&RenderParameter::trianlges, thrust::raw_pointer_cast(d_triangles.data()));
    rndrParam.Set(&RenderParameter::triToPositionIndices,
                  thrust::raw_pointer_cast(d_triToPositionIndices.data()));
    rndrParam.Set(&RenderParameter::positions, thrust::raw_pointer_cast(d_positions.data()));
    rndrParam.Set(&RenderParameter::normals, thrust::raw_pointer_cast(d_normals.data()));
    rndrParam.Set(&RenderParameter::texCoords, thrust::raw_pointer_cast(d_texCoords.data()));
    rndrParam.Set(&RenderParameter::lights, thrust::raw_pointer_cast(d_lights.data()));
    rndrParam.Set(&RenderParameter::materials, thrust::raw_pointer_cast(d_materials.data()));
}

void kouek::RayTracer::RayTracer::SetLBVH(std::shared_ptr<LBVH> lbvh) { this->lbvh = lbvh; }

void kouek::RayTracer::RayTracer::SetWorldToScene(const glm::mat4 &w2s) {
    rndrParam.Set(&RenderParameter::w2s, w2s);
}

void kouek::RayTracer::RayTracer::SetCameraToWorld(const glm::vec3 &eyePos2w,
                                                   const glm::mat3 &eyeRot2w) {
    rndrParamPerFrame.Set(&RenderParameterPerFrame::eyePos2w, eyePos2w);
    rndrParamPerFrame.Set(&RenderParameterPerFrame::eyeRot2w, eyeRot2w);
}

void kouek::RayTracer::RayTracer::SetProjection(const glm::mat4 &proj) {
    auto invProj = Math::InverseProjective(proj);
    rndrParam.Set(&RenderParameter::invProj, invProj);
}

void kouek::RayTracer::RayTracer::SetDisplayHeight(uint8_t displayHeight) {
    rndrParam.Set(&RenderParameter::displayHeight, displayHeight);
}
