#include <ray_tracer/ray_tracer.h>

#include <format>
#include <iostream>

#include <cg/math.h>

kouek::RayTracer::RayTracer::~RayTracer() {}

void kouek::RayTracer::RayTracer::SetMesh(const InputMesh &inputMesh) {
    rndrParam.Set(&RenderParameter::triangleNum,
                  static_cast<IndexTy>(inputMesh.facePositionIndices.size()));

    auto clear = [&](auto &&d_v) {
        d_v.clear();
        d_v.shrink_to_fit();
    };
    clear(d_triangles);
    clear(d_triToPositionIndices);
    clear(d_positions);
    clear(d_triAttrs);

    d_triangles.resize(rndrParam.dat.triangleNum);
    d_triAttrs.resize(rndrParam.dat.triangleNum);
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
    CPY_AOS_TO_SOA(inputMesh.normals, d_triAttrs, norm)
    CPY_AOS_TO_SOA(inputMesh.texCoords, d_triAttrs, texCoord)
#undef CPY_AOS_TO_SOA

    d_triToPositionIndices = inputMesh.facePositionIndices;
    d_positions = inputMesh.positions;

    rndrParam.Set(&RenderParameter::trianlges, thrust::raw_pointer_cast(d_triangles.data()));
    rndrParam.Set(&RenderParameter::triToPositionIndices,
                  thrust::raw_pointer_cast(d_triToPositionIndices.data()));
    rndrParam.Set(&RenderParameter::positions, thrust::raw_pointer_cast(d_positions.data()));
    rndrParam.Set(&RenderParameter::triAttrs, thrust::raw_pointer_cast(d_triAttrs.data()));
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
