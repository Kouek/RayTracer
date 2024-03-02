#ifndef KOUEK_MESH_H
#define KOUEK_MESH_H

#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <optional>
#include <source_location>
#include <string>
#include <string_view>

#include <array>
#include <unordered_map>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>

namespace kouek {
namespace Data {

class OBJMesh {
  public:
    using IndexTy = uint32_t;
    struct Light {
        enum class Type : uint8_t { Quad, Sphere };

        Type type;
        union {
            struct {
                glm::vec3 o, u, v;
            } quad;
            struct {
                glm::vec3 o;
                float r;
            } sphere;
        };
        glm::vec3 radiance;
    };
    struct Material {
        glm::vec3 kd;
        glm::vec3 ks;
        glm::vec3 tr;
        float ni;
        float ns;
    };

  private:
    bool isComplete = false;

    std::vector<glm::vec3> positions;
    std::vector<glm::vec3> normals;
    std::vector<glm::vec2> texCoords;
    std::vector<IndexTy> grpStartFaceIndices;
    std::vector<Light> lights;
    std::vector<glm::vec<3, IndexTy>> facePositionIndices;
    std::vector<glm::vec<3, IndexTy>> faceNormalIndices;
    std::vector<glm::vec<3, IndexTy>> faceTexCoordIndices;
    std::unordered_map<IndexTy, std::string> grp2mtlNames;
    std::unordered_map<std::string, Material> name2mtls;

    std::filesystem::path objPath;
    std::filesystem::path mtllibPath;

    static constexpr std::string_view StatTag = "[OBJMesh Status]";
    static constexpr std::string_view ErrTag = "[OBJMesh Error]";

    enum class Tag : uint8_t {
        Group,
        Face,
        Vertex,
        VertexNormal,
        VertexTextureCoordinate,
        MaterialLibrary,
        UseMaterial
    };
    enum class MTLTag : uint8_t { Kd, Ks, Tr, Ni, Ns, NewMaterial };
    static constexpr std::array<std::string_view, 7> Tags{"g",  "f",      "v",     "vn",
                                                          "vt", "mtllib", "usemtl"};
    static constexpr std::array<std::string_view, 7> MTLTags{"Kd", "Ks", "Tr",
                                                             "Ni", "Ns", "newmtl"};

  public:
    struct InputLight {
        glm::vec3 radiance;
        Light::Type type;
    };
    OBJMesh(const std::filesystem::path &path,
            std::optional<std::unordered_map<std::string, InputLight>> inputLights = {}) {
        using SrcLoc = std::source_location;

        std::ifstream in(path, std::ios::in);
        if (!in.is_open()) {
            std::cerr << std::format("{} at {}:{}. Cannot open file at \"{}\".\n", ErrTag,
                                     SrcLoc::current().file_name(), SrcLoc::current().line(),
                                     path.string());
            return;
        }

        objPath = path;

        auto isPrevGrpLight = false;
        InputLight prevGrpLight;

        uint32_t ln = 0;
        auto processParseErr = [&](std::source_location srcLoc = std::source_location::current()) {
            std::cerr << std::format("{} at {}:{}. Failed parsing line {}.\n", ErrTag,
                                     srcLoc.file_name(), srcLoc.line(), ln);
        };
        std::string lnBuf;
        while (std::getline(in, lnBuf)) {
            ++ln;

            if (lnBuf.empty())
                continue; // skip line "\n"

            auto [tagIdx, offs] = parseTag<false>(lnBuf);
            if (tagIdx == Tags.size()) {
                processParseErr();
                return;
            }

            switch (auto tag = static_cast<Tag>(tagIdx)) {
            case Tag::Group:
                grpStartFaceIndices.emplace_back(facePositionIndices.size());
                break;
            case Tag::Face: {
                std::array<size_t, 9> v;
                auto validNum =
                    sscanf(lnBuf.c_str() + offs + 1, "%zu/%zu/%zu %zu/%zu/%zu %zu/%zu/%zu", &v[0],
                           &v[1], &v[2], &v[3], &v[4], &v[5], &v[6], &v[7], &v[8]);
                if (validNum != v.size()) {
                    processParseErr();
                    return;
                }

                for (auto &i : v)
                    --i;
                facePositionIndices.emplace_back(v[0], v[3], v[6]);
                faceTexCoordIndices.emplace_back(v[1], v[4], v[7]);
                faceNormalIndices.emplace_back(v[2], v[5], v[8]);
            } break;
            case Tag::Vertex:
            case Tag::VertexNormal: {
                std::array<float, 3> v;
                auto validNum = sscanf(lnBuf.c_str() + offs + 1, "%f%f%f", &v[0], &v[1], &v[2]);
                if (validNum != v.size()) {
                    processParseErr();
                    return;
                }
                if (tag == Tag::Vertex)
                    positions.emplace_back(v[0], v[1], v[2]);
                else
                    normals.emplace_back(v[0], v[1], v[2]);
            } break;
            case Tag::VertexTextureCoordinate: {
                std::array<float, 2> v;
                auto validNum = sscanf(lnBuf.c_str() + offs + 1, "%f%f", &v[0], &v[1]);
                if (validNum != v.size()) {
                    processParseErr();
                    return;
                }
                texCoords.emplace_back(v[0], v[1]);
            } break;
            case Tag::MaterialLibrary:
                mtllibPath = objPath.parent_path() / lnBuf.substr(offs + 1);
                if (!loadMTLLib())
                    return;
                break;
            case Tag::UseMaterial: {
                if (isPrevGrpLight && !appendLight(prevGrpLight, grpStartFaceIndices.size() - 2))
                    return;

                auto [itr, inserted] = grp2mtlNames.emplace(
                    std::piecewise_construct, std::forward_as_tuple(grpStartFaceIndices.size() - 1),
                    std::forward_as_tuple(lnBuf.substr(offs + 1)));
                if (!inserted) {
                    processParseErr();
                    return;
                }

                isPrevGrpLight =
                    !inputLights.has_value() ? false : inputLights.value().contains(itr->second);
                if (isPrevGrpLight)
                    prevGrpLight = inputLights.value().at(itr->second);
            } break;
            }
        }
        if (isPrevGrpLight && !appendLight(prevGrpLight, grpStartFaceIndices.size() - 1))
            return;

        std::cout << std::format("{} at {}:{}. Loaded mesh at {}:\n", StatTag,
                                 SrcLoc::current().file_name(), SrcLoc::current().line(),
                                 objPath.string());
        std::cout << std::format("\tvertex num: {}, face num: {}\n", positions.size(),
                                 facePositionIndices.size());

        std::cout << std::format("\tgroup num {}\n", grpStartFaceIndices.size());
        for (auto &[gi, mtlName] : grp2mtlNames)
            std::cout << std::format("\t\tmaterial of group {}: {}\n", gi, mtlName);
        std::cout << std::format("\tlight num {}\n", lights.size());
        for (auto &lht : lights) {
            switch (lht.type) {
            case Light::Type::Quad:
                std::cout << std::format("\t\tquad light: o:({},{},{}), u:({},{},{}), "
                                         "v:({},{},{}), radiance:({},{},{})\n",
                                         lht.quad.o.x, lht.quad.o.y, lht.quad.o.z, lht.quad.u.x,
                                         lht.quad.u.y, lht.quad.u.z, lht.quad.v.x, lht.quad.v.y,
                                         lht.quad.v.z, lht.radiance.r, lht.radiance.g,
                                         lht.radiance.b);
                break;
            case Light::Type::Sphere:
                std::cout << std::format(
                    "\t\tsphere light: o:({},{},{}), r:{}, radiance:({},{},{})\n", lht.sphere.o.x,
                    lht.sphere.o.y, lht.sphere.o.z, lht.sphere.r, lht.radiance.r, lht.radiance.g,
                    lht.radiance.b);
                break;
            }
        }

        std::cout << std::format("{} at {}:{}. Loaded materials at {}:\n", StatTag,
                                 SrcLoc::current().file_name(), SrcLoc::current().line(),
                                 mtllibPath.string());
        for (auto &[name, mtl] : name2mtls) {
            std::cout << std::format("\t{}:\n", name);
            std::cout << std::format("\t\tKd: ({}, {}, {})\n", mtl.kd.r, mtl.kd.g, mtl.kd.b);
            std::cout << std::format("\t\tKs: ({}, {}, {})\n", mtl.ks.r, mtl.ks.g, mtl.ks.b);
            std::cout << std::format("\t\tTr: ({}, {}, {})\n", mtl.tr.r, mtl.tr.g, mtl.tr.b);
            std::cout << std::format("\t\tNs:{}, Ni:{}\n", mtl.ns, mtl.ni);
        }

        isComplete = true;
    }

    bool IsComplete() const { return isComplete; }

#define CONST_REF_GETTER(member, memberNameInFunc)                                                 \
    const decltype(member) &Get##memberNameInFunc() const { return member; }
    CONST_REF_GETTER(positions, Positions)
    CONST_REF_GETTER(normals, Normals)
    CONST_REF_GETTER(texCoords, TextureCoordinates)
    CONST_REF_GETTER(grpStartFaceIndices, GroupStartFaceIndices)
    CONST_REF_GETTER(facePositionIndices, FacePositionIndices)
    CONST_REF_GETTER(faceTexCoordIndices, FaceTextureCoordinateIndices)
    CONST_REF_GETTER(faceNormalIndices, FaceNormalIndices)
    CONST_REF_GETTER(grp2mtlNames, GroupToMaterialNames)
    CONST_REF_GETTER(name2mtls, NameToMaterials)
    CONST_REF_GETTER(lights, Lights)
    CONST_REF_GETTER(objPath, OBJPath)
    CONST_REF_GETTER(mtllibPath, MaterialLibraryPath)
#undef CONST_REF_GETTER

  public:
    static OBJMesh CreateGridTestMesh(const glm::vec<3, IndexTy> &dim, const glm::vec3 &sz) {
        auto dSz = sz / glm::vec3(dim);
        auto gridNum = static_cast<size_t>(dim[0]) * dim[1] * dim[2];
        auto gridMin = glm::zero<glm::vec3>();
        OBJMesh mesh;
        mesh.positions.reserve(gridNum * 3);
        mesh.facePositionIndices.reserve(gridNum * 3);
        for (IndexTy z = 0; z < dim[2]; ++z) {
            gridMin.y = 0.f;
            for (IndexTy y = 0; y < dim[1]; ++y) {
                gridMin.x = 0.f;
                for (IndexTy x = 0; x < dim[0]; ++x) {
                    auto startIdx = mesh.positions.size();

                    mesh.positions.emplace_back(gridMin);
                    mesh.positions.emplace_back(gridMin + glm::vec3{dSz.x, 0.f, 0.f});
                    mesh.positions.emplace_back(gridMin + glm::vec3{0.f, dSz.y, dSz.z});

                    mesh.facePositionIndices.emplace_back(startIdx, startIdx + 1, startIdx + 2);

                    gridMin.x += dSz.x;
                }
                gridMin.y += dSz.y;
            }
            gridMin.z += dSz.z;
        }
        mesh.isComplete = true;

        return mesh;
    }

  private:
    OBJMesh() = default;

    template <bool ParseMTL> std::tuple<uint8_t, size_t> parseTag(const std::string &lnBuf) {
        auto offs = lnBuf.find_first_of(' ');
        if (offs == std::string::npos)
            offs = lnBuf.size();

        uint8_t tagIdx = 0;
        if constexpr (ParseMTL) {
            for (; tagIdx < MTLTags.size(); ++tagIdx) {
                if (offs != MTLTags[tagIdx].size())
                    continue;
                uint8_t chIdx = 0;
                for (; chIdx < MTLTags[tagIdx].size(); ++chIdx)
                    if (lnBuf[chIdx] != MTLTags[tagIdx][chIdx])
                        break;
                if (chIdx == MTLTags[tagIdx].size())
                    break;
            }
        } else {
            for (; tagIdx < Tags.size(); ++tagIdx) {
                if (offs != Tags[tagIdx].size())
                    continue;
                uint8_t chIdx = 0;
                for (; chIdx < Tags[tagIdx].size(); ++chIdx)
                    if (lnBuf[chIdx] != Tags[tagIdx][chIdx])
                        break;
                if (chIdx == Tags[tagIdx].size())
                    break;
            }
        }

        return {tagIdx, offs};
    }

    bool appendLight(const InputLight &inputLight, IndexTy gi) {
        using SrcLoc = std::source_location;

        auto &lht = lights.emplace_back();
        lht.type = inputLight.type;
        lht.radiance = inputLight.radiance;

        auto fiEnd = gi == grpStartFaceIndices.size() - 1 ? faceNormalIndices.size() - 1
                                                          : grpStartFaceIndices[gi + 1];
        switch (inputLight.type) {
        case Light::Type::Quad: {
            auto faceNumInGrp = fiEnd - grpStartFaceIndices[gi];
            if (faceNumInGrp != 2) {
                std::cerr << std::format("{} at {}:{}. Triangle number of group {} is not 2.\n",
                                         ErrTag, SrcLoc::current().file_name(),
                                         SrcLoc::current().line(), gi);
                return false;
            }

            auto facePosIdx = facePositionIndices[grpStartFaceIndices[gi]];
            std::array vecs = {positions[facePosIdx[1]] - positions[facePosIdx[0]],
                               positions[facePosIdx[2]] - positions[facePosIdx[1]],
                               positions[facePosIdx[0]] - positions[facePosIdx[2]]};
            std::array dots = {glm::dot(vecs[0], vecs[1]), glm::dot(vecs[1], vecs[2]),
                               glm::dot(vecs[2], vecs[0])};

            uint8_t i;
            for (i = 0; i < 3; ++i)
                if (glm::abs(dots[i]) < std::numeric_limits<float>::epsilon()) {
                    lht.quad.o = positions[facePosIdx[(i + 1) % 3]];
                    lht.quad.u = positions[facePosIdx[(i + 2) % 3]] - lht.quad.o;
                    lht.quad.v = positions[facePosIdx[i]] - lht.quad.o;
                    break;
                }

            if (i == 3) {
                std::cerr << std::format("{} at {}:{}. Triangle at {} is not right.\n", ErrTag,
                                         SrcLoc::current().file_name(), SrcLoc::current().line(),
                                         grpStartFaceIndices[gi]);
                return false;
            }
        } break;
        case Light::Type::Sphere: {
            lht.sphere.o = glm::vec3(0.f);
            auto fiBeg = grpStartFaceIndices[gi];
            auto div = 3.f * (fiEnd - fiBeg);
            for (auto fi = fiBeg; fi < fiEnd; ++fi)
                for (uint8_t i = 0; i < 3; ++i)
                    lht.sphere.o += positions[facePositionIndices[fi][i]] / div;

            lht.sphere.r = 0.f;
            for (auto fi = fiBeg; fi < fiEnd; ++fi)
                for (uint8_t i = 0; i < 3; ++i)
                    lht.sphere.r +=
                        glm::distance(positions[facePositionIndices[fi][i]], lht.sphere.o) / div;
        } break;
        }

        return true;
    }

    bool loadMTLLib() {
        using SrcLoc = std::source_location;

        std::ifstream in(mtllibPath, std::ios::in);
        if (!in.is_open()) {
            std::cerr << std::format("{} at {}:{}. Cannot open file at \"{}\".\n", ErrTag,
                                     SrcLoc::current().file_name(), SrcLoc::current().line(),
                                     mtllibPath.string());
            return false;
        }

        name2mtls.clear();
        auto currMatrItr = name2mtls.end();

        uint32_t ln = 0;
        auto processParseErr = [&](std::source_location srcLoc = std::source_location::current()) {
            std::cerr << std::format("{} at {}:{}. Failed parsing line {}.\n", ErrTag,
                                     srcLoc.file_name(), srcLoc.line(), ln);
        };
        std::string lnBuf;
        while (std::getline(in, lnBuf)) {
            ++ln;

            if (lnBuf.empty())
                continue; // skip line "\n"

            auto [tagIdx, offs] = parseTag<true>(lnBuf);
            if (tagIdx == MTLTags.size()) {
                processParseErr();
                return false;
            }

            switch (auto tag = static_cast<MTLTag>(tagIdx)) {
            case MTLTag::NewMaterial: {
                auto name = lnBuf.substr(offs + 1);
                if (name2mtls.contains(name)) {
                    processParseErr();
                    return false;
                }

                auto [itr, _] = name2mtls.emplace(name, Material());
                currMatrItr = itr;
            } break;
            case MTLTag::Ni:
            case MTLTag::Ns: {
                float v;
                auto validNum = sscanf(lnBuf.c_str() + offs + 1, "%f", &v);
                if (validNum != 1) {
                    processParseErr();
                    return false;
                }

                if (tag == MTLTag::Ni)
                    currMatrItr->second.ni = v;
                else
                    currMatrItr->second.ns = v;
            } break;
            case MTLTag::Kd:
            case MTLTag::Ks:
            case MTLTag::Tr: {
                glm::vec3 v;
                auto validNum = sscanf(lnBuf.c_str() + offs + 1, "%f%f%f", &v[0], &v[1], &v[2]);
                if (validNum != 3) {
                    processParseErr();
                    return false;
                }

                if (tag == MTLTag::Kd)
                    currMatrItr->second.kd = v;
                else if (tag == MTLTag::Ks)
                    currMatrItr->second.ks = v;
                else
                    currMatrItr->second.tr = v;
            } break;
            }
        }

        return true;
    }
};

} // namespace Data
} // namespace kouek

#endif // !KOUEK_MESH_H
