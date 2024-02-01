#include <ray_tracer/ray_tracer.h>

#include <cg/ray.h>

__device__ uchar4 renderAABB(const kouek::RayTracer::RayTracer::RenderParameter &rndrParam,
                             const kouek::RayTracer::LBVH::DeviceData &lbvh,
                             const kouek::Ray &eyeRay) {
    using IndexTy = kouek::RayTracer::LBVH::IndexTy;
    using INodeTy = kouek::RayTracer::LBVH::InternalNode;

    auto extent = lbvh.rootAABB.maxPos - lbvh.rootAABB.minPos;
#ifdef __CUDA_ARCH__
#pragma unroll
#endif
    for (uint8_t xyz = 0; xyz < 3; ++xyz)
        if (extent[xyz] < kouek::CUDA::FloatEps)
            extent[xyz] = kouek::CUDA::FloatEps;

    kouek::Ray::HitShellResult hit;
    auto hitNode = [&](IndexTy inodeIdx) {
        if (inodeIdx & INodeTy::ChildIsLeafMask)
            return false;

        auto aabb = lbvh.internalNodes[inodeIdx].aabb;
        hit = eyeRay.Hit(aabb);
        return true;
    };

    hitNode(0);
    if (hit.tEnter > hit.tExit)
        return uchar4{0, 0, 0, 0};

    constexpr auto EnterAlpha = .7f;
    constexpr auto ExitAlpha = 1.f - EnterAlpha;
    glm::vec3 color(-1.f);
    if (rndrParam.displayHeight == 0) {
        auto enterPos = eyeRay.pos + hit.tEnter * eyeRay.dir - lbvh.rootAABB.minPos;
        auto exitPos = eyeRay.pos + hit.tExit * eyeRay.dir - lbvh.rootAABB.minPos;
        color = (EnterAlpha * enterPos + ExitAlpha * exitPos) / extent;
        return uchar4{glm::clamp(static_cast<uint8_t>(255.f * color.r), uint8_t(0), uint8_t(255)),
                      glm::clamp(static_cast<uint8_t>(255.f * color.g), uint8_t(0), uint8_t(255)),
                      glm::clamp(static_cast<uint8_t>(255.f * color.b), uint8_t(0), uint8_t(255)),
                      255};
    }

    thrust::tuple<IndexTy, uint8_t> stk[30];
    uint8_t stkSz = 1;
    thrust::get<0>(stk[stkSz]) = 0;
    thrust::get<1>(stk[stkSz]) = 0;

    auto push = [&](IndexTy inodeIdx) {
        thrust::get<0>(stk[stkSz]) = inodeIdx;
        thrust::get<1>(stk[stkSz]) = 0;
        ++stkSz;
    };
    auto pop = [&]() { --stkSz; };
    auto topINodeIdx = [&]() { return thrust::get<0>(stk[stkSz - 1]); };
    auto topVisitCnt = [&]() { return thrust::get<1>(stk[stkSz - 1]); };
    auto incTop = [&]() { ++thrust::get<1>(stk[stkSz - 1]); };

    auto tNearest = kouek::CUDA::FloatMax;
    while (stkSz != 0) {
        if (topVisitCnt() == 2) {
            pop();
            continue;
        }

        auto chIdx = topVisitCnt() == 0 ? lbvh.internalNodes[topINodeIdx()].lft
                                        : lbvh.internalNodes[topINodeIdx()].rht;
        incTop();
        auto childIsINode = hitNode(chIdx);
        if (childIsINode && hit.tEnter <= hit.tExit)
            if (rndrParam.displayHeight == stkSz - 1 && hit.tEnter < tNearest) {
                auto minPos = lbvh.internalNodes[chIdx].aabb.minPos;
                auto extent =
                    lbvh.internalNodes[chIdx].aabb.maxPos - lbvh.internalNodes[chIdx].aabb.minPos;
#ifdef __CUDA_ARCH__
#pragma unroll
#endif
                for (uint8_t xyz = 0; xyz < 3; ++xyz)
                    if (extent[xyz] < kouek::CUDA::FloatEps)
                        extent[xyz] = kouek::CUDA::FloatEps;

                auto enterPos = eyeRay.pos + hit.tEnter * eyeRay.dir - minPos;
                auto exitPos = eyeRay.pos + hit.tExit * eyeRay.dir - minPos;
                color = (EnterAlpha * enterPos + ExitAlpha * exitPos) / extent;
                tNearest = hit.tEnter;
            } else if (stkSz <= rndrParam.displayHeight)
                push(chIdx);
    }

    if (tNearest == kouek::CUDA::FloatMax)
        return uchar4{0, 0, 0, 0};
    return uchar4{glm::clamp(static_cast<uint8_t>(255.f * color.r), uint8_t(0), uint8_t(255)),
                  glm::clamp(static_cast<uint8_t>(255.f * color.g), uint8_t(0), uint8_t(255)),
                  glm::clamp(static_cast<uint8_t>(255.f * color.b), uint8_t(0), uint8_t(255)), 255};
}

__device__ uchar4 renderTriangles(const kouek::RayTracer::RayTracer::RenderParameter &rndrParam,
                                  const kouek::RayTracer::LBVH::DeviceData &lbvh,
                                  const kouek::Ray &eyeRay) {
    using IndexTy = kouek::RayTracer::LBVH::IndexTy;
    using INodeTy = kouek::RayTracer::LBVH::InternalNode;

    kouek::Ray::HitShellResult hit;
    auto hitNode = [&](IndexTy inodeIdx) {
        if (inodeIdx & INodeTy::ChildIsLeafMask)
            return false;

        auto aabb = lbvh.internalNodes[inodeIdx].aabb;
        hit = eyeRay.Hit(aabb);
        return true;
    };

    hitNode(0);
    if (hit.tEnter > hit.tExit)
        return uchar4{0, 0, 0, 0};

    thrust::tuple<IndexTy, uint8_t> stk[30];
    uint8_t stkSz = 1;
    thrust::get<0>(stk[stkSz]) = 0;
    thrust::get<1>(stk[stkSz]) = 0;

    auto push = [&](IndexTy inodeIdx) {
        thrust::get<0>(stk[stkSz]) = inodeIdx;
        thrust::get<1>(stk[stkSz]) = 0;
        ++stkSz;
    };
    auto pop = [&]() { --stkSz; };
    auto topINodeIdx = [&]() { return thrust::get<0>(stk[stkSz - 1]); };
    auto topVisitCnt = [&]() { return thrust::get<1>(stk[stkSz - 1]); };
    auto incTop = [&]() { ++thrust::get<1>(stk[stkSz - 1]); };

    glm::vec3 color = {0.f, 0.f, 0.f};
    auto tNearest = kouek::CUDA::FloatMax;
    while (stkSz != 0) {
        if (topVisitCnt() == 2) {
            pop();
            continue;
        }
        auto chIdx = topVisitCnt() == 0 ? lbvh.internalNodes[topINodeIdx()].lft
                                        : lbvh.internalNodes[topINodeIdx()].rht;
        incTop();
        auto childIsINode = hitNode(chIdx);
        if (childIsINode && hit.tEnter > hit.tExit)
            continue;
        else if (childIsINode)
            push(chIdx);
        else {
            auto cmpctIdx = chIdx & INodeTy::LeafIndexMask;
            auto fi = lbvh.compactedFaces[cmpctIdx];
            auto fiEnd = cmpctIdx == lbvh.compactedFaceNum - 1 ? lbvh.faceNum
                                                               : lbvh.compactedFaces[cmpctIdx + 1];
            while (fi < fiEnd) {
                auto facePosIdx = rndrParam.triToPositionIndices[lbvh.faces[fi]];
                auto hitTri = eyeRay.Hit(rndrParam.positions[facePosIdx[0]],
                                         rndrParam.positions[facePosIdx[1]],
                                         rndrParam.positions[facePosIdx[2]]);
                if (hitTri.t >= 0.f && hitTri.t < tNearest) {
                    color.r = hitTri.bary.x;
                    color.g = hitTri.bary.y;
                    color.b = 1.f - hitTri.bary.x - hitTri.bary.y;
                    tNearest = hitTri.t;
                }
                ++fi;
            }
        }
    }

    if (tNearest == kouek::CUDA::FloatMax)
        return uchar4{0, 0, 0, 0};
    return uchar4{glm::clamp(static_cast<uint8_t>(255.f * color.r), uint8_t(0), uint8_t(255)),
                  glm::clamp(static_cast<uint8_t>(255.f * color.g), uint8_t(0), uint8_t(255)),
                  glm::clamp(static_cast<uint8_t>(255.f * color.b), uint8_t(0), uint8_t(255)), 255};
}

void kouek::RayTracer::RayTracer::Render(cudaSurfaceObject_t rndrTo, const glm::ivec2 &rndrSz,
                                         RenderTarget rndrTarget) {
    if (!lbvh) {
        std::cerr << std::format("{} at {}:{}. lbvh is not set.\n", ErrTag, __FILE__, __LINE__);
        return;
    }

    auto uploadInNeed = [](auto &modifiable, auto **d_datPtrPtr) {
        using namespace kouek::CUDA;
        if (modifiable.modified) {
            if (!(*d_datPtrPtr))
                KOUEK_CUDA_CHECK(cudaMalloc(d_datPtrPtr, sizeof(**d_datPtrPtr)));

            auto &dat = modifiable.GetAndReset();
            KOUEK_CUDA_CHECK(cudaMemcpy(*d_datPtrPtr, &dat,
                                        sizeof(std::remove_reference_t<decltype(dat)>),
                                        cudaMemcpyHostToDevice));
        }
    };
    uploadInNeed(rndrParam, &d_rndrParamPtr);
    uploadInNeed(rndrParamPerFrame, &d_rndrParamPerFramePtr);

    auto flatRes = static_cast<glm::uint>(rndrSz.x) * rndrSz.y;
    thrust::for_each(
        thrust::make_counting_iterator(decltype(flatRes)(0)),
        thrust::make_counting_iterator(flatRes),
        [rndrTo, rndrSz, rndrTarget, rndrParamPtr = d_rndrParamPtr,
         rndrParamPerFramePtr = d_rndrParamPerFramePtr,
         lbvhPtr = lbvh->GetDeivceDataDevicePtr()] __device__(decltype(flatRes) threadIdx) {
            glm::ivec2 pix;
            pix.y = threadIdx / rndrSz.x;
            pix.x = threadIdx - pix.y * rndrSz.x;

            Ray eyeRay;
            {
                // Map [0, rndrSz.xy - 1] to (-1, 1)
                glm::vec4 tmp;
                tmp.z = rndrSz.x;
                tmp.w = rndrSz.y;
                tmp.x = (2.f * pix.x + 1.f - tmp.z) / tmp.z;
                tmp.y = (2.f * pix.y + 1.f - tmp.w) / tmp.w;

                // Inverse-project
                tmp.z = 1.f;
                tmp.w = 1.f;
                tmp = rndrParamPtr->invProj * tmp;

                eyeRay.dir = tmp;
                eyeRay.dir = rndrParamPerFramePtr->eyeRot2w * glm::normalize(eyeRay.dir);

                // Transform from world to scene
                tmp = glm::vec4(eyeRay.dir, 0.f);
                tmp = rndrParamPtr->w2s * tmp;
                eyeRay.dir = glm::normalize(eyeRay.dir);

                tmp = glm::vec4(rndrParamPerFramePtr->eyePos2w, 1.f);
                tmp = rndrParamPtr->w2s * tmp;
                eyeRay.pos = tmp;
            }

            uchar4 color;
            switch (rndrTarget) {
            case RenderTarget::Scene:
                break;
            case RenderTarget::AABBs:
                color = renderAABB(*rndrParamPtr, *lbvhPtr, eyeRay);
                break;
            case RenderTarget::Triangles:
                color = renderTriangles(*rndrParamPtr, *lbvhPtr, eyeRay);
                break;
            }

            surf2Dwrite(color, rndrTo, sizeof(uchar4) * pix.x, pix.y);
        });
}
