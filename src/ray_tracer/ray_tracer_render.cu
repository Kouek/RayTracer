#include <ray_tracer/ray_tracer.h>

#include <glm/gtc/random.hpp>

#include <cg/math.h>
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
    if (rndrParam.displayTreeHeight == 0) {
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
    thrust::get<0>(stk[0]) = 0;
    thrust::get<1>(stk[0]) = 0;

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
            if (rndrParam.displayTreeHeight == stkSz - 1 && hit.tEnter < tNearest) {
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
            } else if (stkSz <= rndrParam.displayTreeHeight)
                push(chIdx);
    }

    if (tNearest == kouek::CUDA::FloatMax)
        return uchar4{0, 0, 0, 0};
    return uchar4{glm::clamp(static_cast<uint8_t>(255.f * color.r), uint8_t(0), uint8_t(255)),
                  glm::clamp(static_cast<uint8_t>(255.f * color.g), uint8_t(0), uint8_t(255)),
                  glm::clamp(static_cast<uint8_t>(255.f * color.b), uint8_t(0), uint8_t(255)), 255};
}

template <typename TriangleFuncTy>
__device__ uchar4 rayCastLBVH(const kouek::RayTracer::RayTracer::RenderParameter &rndrParam,
                              const kouek::RayTracer::LBVH::DeviceData &lbvh, const kouek::Ray &ray,
                              TriangleFuncTy triFunc) {
    using IndexTy = kouek::RayTracer::LBVH::IndexTy;
    using INodeTy = kouek::RayTracer::LBVH::InternalNode;

    kouek::Ray::HitShellResult hit;
    auto hitNode = [&](IndexTy inodeIdx) {
        if (inodeIdx & INodeTy::ChildIsLeafMask)
            return false;

        auto aabb = lbvh.internalNodes[inodeIdx].aabb;
        hit = ray.Hit(aabb);
        return true;
    };

    hitNode(0);
    if (hit.tEnter > hit.tExit)
        return uchar4{0, 0, 0, 0};

    thrust::tuple<IndexTy, uint8_t> stk[30];
    uint8_t stkSz = 1;
    thrust::get<0>(stk[0]) = 0;
    thrust::get<1>(stk[0]) = 0;

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
            auto lbvhFi = lbvh.compactedFaces[cmpctIdx];
            auto lbvhFiEnd = cmpctIdx == lbvh.compactedFaceNum - 1
                                 ? lbvh.faceNum
                                 : lbvh.compactedFaces[cmpctIdx + 1];
            while (lbvhFi < lbvhFiEnd) {
                auto fi = lbvh.faces[lbvhFi];
                auto facePosIdx = rndrParam.triToPositionIndices[fi];
                auto hit =
                    ray.Hit(kouek::Ray::HitTriangle{.p0 = rndrParam.positions[facePosIdx[0]],
                                                    .p1 = rndrParam.positions[facePosIdx[1]],
                                                    .p2 = rndrParam.positions[facePosIdx[2]]});
                if (hit.t >= 0.f && hit.t < tNearest) {
                    triFunc(color, fi, hit);
                    tNearest = hit.t;
                }
                ++lbvhFi;
            }
        }
    }

    if (tNearest == kouek::CUDA::FloatMax)
        return uchar4{0, 0, 0, 0};
    return uchar4{glm::clamp(static_cast<uint8_t>(255.f * color.r), uint8_t(0), uint8_t(255)),
                  glm::clamp(static_cast<uint8_t>(255.f * color.g), uint8_t(0), uint8_t(255)),
                  glm::clamp(static_cast<uint8_t>(255.f * color.b), uint8_t(0), uint8_t(255)), 255};
}

template <bool IsSimple> struct KOUEK_CUDA_ALIGN Nearest {
    float t;
    glm::vec2 bary;
    glm::vec3 tngnt;
    kouek::RayTracer::RayTracer::IndexTy fi;
};

template <> struct KOUEK_CUDA_ALIGN Nearest<true> {
    float t;
};

template <bool IsSimple = false>
__device__ Nearest<IsSimple>
rayCastLBVH(const kouek::RayTracer::RayTracer::RenderParameter &rndrParam,
            const kouek::RayTracer::LBVH::DeviceData &lbvh, const kouek::Ray &ray) {
    using INodeTy = kouek::RayTracer::LBVH::InternalNode;
    using IndexTy = kouek::RayTracer::RayTracer::IndexTy;

    Nearest<IsSimple> nearest{.t = kouek::CUDA::FloatMax};

    kouek::Ray::HitShellResult hit;
    auto hitNode = [&](IndexTy inodeIdx) {
        if (inodeIdx & INodeTy::ChildIsLeafMask)
            return false;

        auto aabb = lbvh.internalNodes[inodeIdx].aabb;
        hit = ray.Hit(aabb);
        return true;
    };

    hitNode(0);
    if (hit.tEnter > hit.tExit)
        return nearest;

    thrust::tuple<IndexTy, uint8_t> stk[30];
    uint8_t stkSz;
    auto push = [&](IndexTy inodeIdx) {
        thrust::get<0>(stk[stkSz]) = inodeIdx;
        thrust::get<1>(stk[stkSz]) = 0;
        ++stkSz;
    };
    auto pop = [&]() { --stkSz; };
    auto topINodeIdx = [&]() { return thrust::get<0>(stk[stkSz - 1]); };
    auto topVisitCnt = [&]() { return thrust::get<1>(stk[stkSz - 1]); };
    auto incTop = [&]() { ++thrust::get<1>(stk[stkSz - 1]); };

    stkSz = 1;
    thrust::get<0>(stk[0]) = 0;
    thrust::get<1>(stk[0]) = 0;
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
            auto lbvhFi = lbvh.compactedFaces[cmpctIdx];
            auto lbvhFiEnd = cmpctIdx == lbvh.compactedFaceNum - 1
                                 ? lbvh.faceNum
                                 : lbvh.compactedFaces[cmpctIdx + 1];
            while (lbvhFi < lbvhFiEnd) {
                auto fi = lbvh.faces[lbvhFi];
                auto facePosIdx = rndrParam.triToPositionIndices[fi];
                kouek::Ray::HitTriangle hitTri{.p0 = rndrParam.positions[facePosIdx[0]],
                                               .p1 = rndrParam.positions[facePosIdx[1]],
                                               .p2 = rndrParam.positions[facePosIdx[2]]};
                auto hit = ray.Hit(hitTri);
                if (hit.t >= 0.f && hit.t < nearest.t) {
                    nearest.t = hit.t;
                    if constexpr (!IsSimple) {
                        nearest.bary = hit.bary;
                        nearest.tngnt = hitTri.p1 - hitTri.p0;
                        nearest.fi = fi;
                    }
                }
                ++lbvhFi;
            }
        }
    }

    return nearest;
}

__device__ uchar4 renderLights(const kouek::RayTracer::RayTracer::RenderParameter &rndrParam,
                               const kouek::Ray &eyeRay) {
    using IndexTy = kouek::RayTracer::RayTracer::IndexTy;

    glm::vec3 color = {0.f, 0.f, 0.f};
    auto tNearest = kouek::CUDA::FloatMax;

    for (IndexTy li = 0; li < rndrParam.lightNum; ++li) {
        auto lht = rndrParam.lights[li];

        switch (lht.type) {
        case kouek::RayTracer::Light::Type::Quad: {
            auto hit =
                eyeRay.Hit(kouek::Ray::HitQuad{.o = lht.quad.o, .u = lht.quad.u, .v = lht.quad.v});
            if (hit.t >= 0.f && hit.t < tNearest) {
                color.r = hit.u / glm::length(lht.quad.u);
                color.g = hit.v / glm::length(lht.quad.v);
                tNearest = hit.t;
            }
        } break;
        case kouek::RayTracer::Light::Type::Sphere: {
            auto hit =
                eyeRay.Hit(kouek::Ray::HitSphere{.rSqr = lht.sphere.rSqr, .o = lht.sphere.o});
            if (hit.tEnter <= hit.tExit && hit.tEnter < tNearest) {
                auto dir = eyeRay.pos + hit.tEnter * eyeRay.dir - lht.sphere.o;
                dir = glm::normalize(dir);
                color = glm::abs(dir);
                tNearest = hit.tEnter;
            }
        } break;
        }
    }

    if (tNearest == kouek::CUDA::FloatMax)
        return uchar4{0, 0, 0, 0};
    return uchar4{glm::clamp(static_cast<uint8_t>(255.f * color.r), uint8_t(0), uint8_t(255)),
                  glm::clamp(static_cast<uint8_t>(255.f * color.g), uint8_t(0), uint8_t(255)),
                  glm::clamp(static_cast<uint8_t>(255.f * color.b), uint8_t(0), uint8_t(255)), 255};
}

__device__ glm::vec3 renderScene(const kouek::RayTracer::RayTracer::RenderParameter &rndrParam,
                                 const kouek::RayTracer::LBVH::DeviceData &lbvh,
                                 curandState &randState, kouek::Ray ray) {
    using INodeTy = kouek::RayTracer::LBVH::InternalNode;
    using IndexTy = kouek::RayTracer::RayTracer::IndexTy;
    auto Eps = kouek::RayTracer::RayTracer::Eps;

    constexpr auto RR = .8f;

    glm::vec3 preMult(1.f);
    glm::vec3 Lo(0.f);
    for (uint8_t depth = 0; depth < rndrParam.maxPathDepth; ++depth) {
        if (glm::max(preMult.r, glm::max(preMult.g, preMult.b)) < kouek::CUDA::FloatEps)
            break;

        auto nearest = rayCastLBVH(rndrParam, lbvh, ray);
        if (nearest.t == kouek::CUDA::FloatMax)
            break;

        auto triangle = rndrParam.trianlges[nearest.fi];
        auto matr = rndrParam.materials[triangle.grpIdx];

        if (matr.emissive) {
            Lo += preMult * matr.emission;
            break;
        }

        auto u = curand_uniform(&randState);
        if (u > RR)
            break;

        auto norm = [&]() {
            glm::vec3 norm[] = {rndrParam.normals[triangle.normIdx[0]],
                                rndrParam.normals[triangle.normIdx[1]],
                                rndrParam.normals[triangle.normIdx[2]]};
            return nearest.bary.x * norm[0] + nearest.bary.y * norm[1] +
                   (1.f - nearest.bary.x - nearest.bary.y) * norm[2];
        }();
        auto dirOut = -ray.dir;
        ray.pos = ray.pos + nearest.t * ray.dir + Eps * norm;

        struct Direct {
            glm::vec3 Li;
            float pdf;
        };
        auto direct = [&]() -> Direct {
            auto lht = [&]() {
                auto u = curand_uniform(&randState) * rndrParam.lightNum;
                u = glm::floor(u);

                auto li = static_cast<IndexTy>(u);
                if (li == rndrParam.lightNum)
                    return rndrParam.lights[0];
                return rndrParam.lights[li];
            }();
            auto lhtSample = [&]() {
                auto u0 = curand_uniform(&randState);
                auto u1 = curand_uniform(&randState);

                switch (lht.type) {
                case decltype(lht)::Type::Quad:
                    return lht.SampleQuad(u0, u1);
                case decltype(lht)::Type::Sphere:
                default:
                    return lht.SampleSphere(u0, u1, ray.pos);
                }
            }();
            lhtSample.pos += Eps * lhtSample.norm;

            auto in = lhtSample.pos - ray.pos;
            auto sqrIn = glm::dot(in, in);

            ray.dir = glm::normalize(in);
            auto nearest = rayCastLBVH<true>(rndrParam, lbvh, ray);
            if (nearest.t < glm::sqrt(sqrIn) - Eps)
                return {glm::vec3(0.f), 0.f};

            auto pdf = lhtSample.pdf / rndrParam.lightNum;
            return {matr.BRDF(ray.dir, dirOut, norm) * lht.radiance *
                        glm::max(glm::dot(ray.dir, norm), 0.f) *
                        glm::max(glm::dot(-ray.dir, lhtSample.norm), 0.f) / sqrIn / pdf,
                    pdf};
        }();

        float pdfIndrct;
        u = curand_uniform(&randState);
        auto u0 = curand_uniform(&randState);
        auto u1 = curand_uniform(&randState);
        if (u <= matr.kdOverKdAddKs) {
            nearest.tngnt = glm::normalize(nearest.tngnt);
            ray.dir = kouek::Math::ThetaPhiToDirection(
                glm::acos(glm::sqrt(u0)), glm::two_pi<float>() * u1, norm, nearest.tngnt);
            pdfIndrct = glm::dot(ray.dir, norm) * glm::one_over_pi<float>();
        } else {
            auto reflect = glm::normalize(2.f * norm - dirOut);
            auto tngnt = kouek::Math::GenerateTangent(reflect);
            ray.dir =
                kouek::Math::ThetaPhiToDirection(glm::acos(glm::pow(u0, 1.f / (matr.ns + 1.f))),
                                                 glm::two_pi<float>() * u1, reflect, tngnt);
            pdfIndrct = (matr.ns + 1.f) * glm::one_over_two_pi<float>() *
                        glm::pow(glm::max(glm::dot(reflect, ray.dir), 0.f), matr.ns);
        }

        Lo += preMult * direct.Li;
        preMult *= matr.BRDF(ray.dir, dirOut, norm) * glm::dot(ray.dir, norm) / pdfIndrct;

        // auto pdfDrctPower = direct.pdf * direct.pdf;
        // auto pdfIndrctPower = pdfIndrct * pdfIndrct;
        // auto pdfPowerSum = pdfDrctPower + pdfIndrctPower;
        //
        // Lo += preMult * pdfDrctPower / pdfPowerSum * direct.Li;

        // preMult *= matr.BRDF(ray.dir, dirOut, norm) * glm::dot(ray.dir, norm) / pdfIndrct *
        //            pdfIndrctPower / pdfPowerSum;
    }

    Lo /= RR;
    kouek::Math::HDRToLDRCorrect(Lo);
    kouek::Math::GammaCorrect(Lo);
    return Lo;
}

void kouek::RayTracer::RayTracer::Render(cudaSurfaceObject_t rndrTo, const glm::ivec2 &rndrSz,
                                         RenderTarget rndrTarget) {
    if (!lbvh) {
        std::cerr << std::format("{} at {}:{}. lbvh is not set.\n", ErrTag, __FILE__, __LINE__);
        return;
    }

    auto needRedraw = false;
    auto uploadInNeed = [&](auto &modifiable, auto **d_datPtrPtr) {
        if (modifiable.modified) {
            if (!(*d_datPtrPtr))
                KOUEK_CUDA_CHECK(cudaMalloc(d_datPtrPtr, sizeof(**d_datPtrPtr)));

            auto &dat = modifiable.GetAndReset();
            KOUEK_CUDA_CHECK(cudaMemcpy(*d_datPtrPtr, &dat,
                                        sizeof(std::remove_reference_t<decltype(dat)>),
                                        cudaMemcpyHostToDevice));

            needRedraw = true;
        }
    };
    uploadInNeed(rndrParam, &d_rndrParamPtr);
    uploadInNeed(rndrParamPerFrame, &d_rndrParamPerFramePtr);

    auto flatRes = static_cast<glm::uint>(rndrSz.x) * rndrSz.y;
    if (prevRndrSz != rndrSz) {
        d_accuRndrTo.assign(flatRes, glm::vec3(0.f));
        d_randStates.assign(flatRes, curandState());
    }

    if (rndrTarget != RenderTarget::Scene)
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
                case RenderTarget::AABBs:
                    color = renderAABB(*rndrParamPtr, *lbvhPtr, eyeRay);
                    break;
                case RenderTarget::Triangles:
                    color = rayCastLBVH(*rndrParamPtr, *lbvhPtr, eyeRay,
                                        [&](glm::vec3 &color, IndexTy fi,
                                            const kouek::Ray::HitTriangleResult &hit) {
                                            color.r = hit.bary.x;
                                            color.g = hit.bary.y;
                                            color.b = 1.f - hit.bary.x - hit.bary.y;
                                        });
                    break;
                case RenderTarget::Lights:
                    color = renderLights(*rndrParamPtr, eyeRay);
                    break;
                case RenderTarget::Positions:
                    color = rayCastLBVH(*rndrParamPtr, *lbvhPtr, eyeRay,
                                        [&](glm::vec3 &color, IndexTy fi,
                                            const kouek::Ray::HitTriangleResult &hit) {
                                            auto pos = eyeRay.pos + hit.t * eyeRay.dir;
                                            color = (pos - lbvhPtr->rootAABB.minPos) /
                                                    (lbvhPtr->rootAABB.maxPos -
                                                     lbvhPtr->rootAABB.minPos);
                                        });
                    break;
                case RenderTarget::Normals:
                    color = rayCastLBVH(
                        *rndrParamPtr, *lbvhPtr, eyeRay,
                        [&](glm::vec3 &color, IndexTy fi,
                            const kouek::Ray::HitTriangleResult &hit) {
                            auto face = rndrParamPtr->trianlges[fi];
                            glm::vec3 norms[3] = {rndrParamPtr->normals[face.normIdx[0]],
                                                  rndrParamPtr->normals[face.normIdx[1]],
                                                  rndrParamPtr->normals[face.normIdx[2]]};
                            color = glm::abs(hit.bary.x * norms[0] + hit.bary.y * norms[0] +
                                             (1.f - hit.bary.x - hit.bary.y) * norms[2]);
                        });
                    break;
                case RenderTarget::TextureCoords:
                    color = rayCastLBVH(
                        *rndrParamPtr, *lbvhPtr, eyeRay,
                        [&](glm::vec3 &color, IndexTy fi,
                            const kouek::Ray::HitTriangleResult &hit) {
                            auto face = rndrParamPtr->trianlges[fi];
                            glm::vec2 norms[3] = {rndrParamPtr->normals[face.texCoordIdx[0]],
                                                  rndrParamPtr->normals[face.texCoordIdx[1]],
                                                  rndrParamPtr->normals[face.texCoordIdx[2]]};
                            auto rg = glm::abs(hit.bary.x * norms[0] + hit.bary.y * norms[0] +
                                               (1.f - hit.bary.x - hit.bary.y) * norms[2]);
                            color.r = rg.r;
                            color.g = rg.g;
                        });
                    break;
                }

                surf2Dwrite(color, rndrTo, sizeof(uchar4) * pix.x, pix.y);
            });
    else {
        needRedraw |= prevMaxSamplePerPixel != maxSamplePerPixel || prevRndrTo != rndrTo ||
                      prevRndrSz != rndrSz || prevRndrTarget != rndrTarget;
        if (needRedraw) {
            currAccuSamplePerPixel = 1.f;

            thrust::for_each(thrust::make_counting_iterator(decltype(flatRes)(0)),
                             thrust::make_counting_iterator(flatRes),
                             [accuRndrTo = thrust::raw_pointer_cast(d_accuRndrTo.data()),
                              randStates = thrust::raw_pointer_cast(
                                  d_randStates.data())] __device__(decltype(flatRes) threadIdx) {
                                 accuRndrTo[threadIdx] = glm::vec3(0.f);
                                 curand_init(1234, threadIdx, 0, &randStates[threadIdx]);
                             });
        }

        if (currAccuSamplePerPixel > maxSamplePerPixel)
            return;

        thrust::for_each(
            thrust::make_counting_iterator(decltype(flatRes)(0)),
            thrust::make_counting_iterator(flatRes),
            [rndrTo, rndrSz, rndrTarget, currAccuSamplePerPixel = this->currAccuSamplePerPixel,
             rndrParamPtr = d_rndrParamPtr, rndrParamPerFramePtr = d_rndrParamPerFramePtr,
             lbvhPtr = lbvh->GetDeivceDataDevicePtr(),
             accuRndrTo = thrust::raw_pointer_cast(d_accuRndrTo.data()),
             randStates = thrust::raw_pointer_cast(
                 d_randStates.data())] __device__(decltype(flatRes) threadIdx) {
                glm::ivec2 pix;
                pix.y = threadIdx / rndrSz.x;
                pix.x = threadIdx - pix.y * rndrSz.x;

                Ray eyeRay;
                {
                    // Map [0, rndrSz.xy - 1] to (-1, 1)
                    glm::vec4 tmp;
                    tmp.x = 1.f - curand_uniform(&randStates[threadIdx]) + pix.x;
                    tmp.y = 1.f - curand_uniform(&randStates[threadIdx]) + pix.y;
                    tmp.z = rndrSz.x;
                    tmp.w = rndrSz.y;
                    tmp.x = (2.f * tmp.x - tmp.z) / tmp.z;
                    tmp.y = (2.f * tmp.y - tmp.w) / tmp.w;

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

                auto color = renderScene(*rndrParamPtr, *lbvhPtr, randStates[threadIdx], eyeRay);

                /* x_{0...n-1} = (x_0 + x_1 + ... + x_{n-1}) / n
                 * x_{0...n-1} = ((x_0 + x_1 + ... + x_{n-2}) / (n-1) * (n-1) + x_{n-1}) / n
                 * Thus, x_{0...n-1} = (x_{0...n-2} * (n-1) + x_{n-1}) / n
                 * Thus, x_{0...n-1} = x_{0...n-2} + (x_{n-1} - x_{0...n-2}) / n
                 */
                auto prevCol = accuRndrTo[threadIdx];
                accuRndrTo[threadIdx] = color =
                    prevCol + (color - prevCol) / currAccuSamplePerPixel;

                auto colorUCh4 = uchar4{
                    glm::clamp(static_cast<uint8_t>(255.f * color.r), uint8_t(0), uint8_t(255)),
                    glm::clamp(static_cast<uint8_t>(255.f * color.g), uint8_t(0), uint8_t(255)),
                    glm::clamp(static_cast<uint8_t>(255.f * color.b), uint8_t(0), uint8_t(255)),
                    255};
                surf2Dwrite(colorUCh4, rndrTo, sizeof(uchar4) * pix.x, pix.y);
            });

        prevMaxSamplePerPixel = maxSamplePerPixel;
        currAccuSamplePerPixel += 1.f;
    }

    prevRndrTo = rndrTo;
    prevRndrSz = rndrSz;
    prevRndrTarget = rndrTarget;
}
