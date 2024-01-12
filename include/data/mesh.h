#ifndef KOUEK_MESH_H
#define KOUEK_MESH_H

#include <format>
#include <fstream>
#include <functional>
#include <iostream>
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

  private:
    bool isComplete = false;

    std::vector<glm::vec3> positions;
    std::vector<glm::vec3> normals;
    std::vector<glm::vec2> texCoords;
    std::vector<IndexTy> groups;
    std::vector<glm::vec<3, IndexTy>> facePositionIndices;
    std::vector<glm::vec<3, IndexTy>> faceNormalIndices;
    std::vector<glm::vec<3, IndexTy>> faceTexCoordIndices;
    std::unordered_map<IndexTy, std::string> grp2mtls;

    std::string objPath;
    std::string mtllibPath;

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
    static constexpr std::array<std::string_view, 7> Tags{"g",  "f",      "v",     "vn",
                                                          "vt", "mtllib", "usemtl"};

  public:
    OBJMesh() = default;
    OBJMesh(const std::string &path) { LoadFromFile(path); }

    void LoadFromFile(const std::string &path) {
        isComplete = false;

        std::ifstream in(path, std::ios::in);
        if (!in.is_open()) {
            auto srcLoc = std::source_location::current();
            std::cerr << std::format("{} at {}:{}. Cannot open file at \"{}\".\n", ErrTag,
                                     srcLoc.file_name(), srcLoc.line(), path);
            return;
        }

        objPath = path;

        uint32_t ln = 0;
        auto processParseErr = [&](std::source_location srcLoc = std::source_location::current()) {
            std::cerr << std::format("{} at {}:{}. Failed parsing line {}.\n", ErrTag,
                                     srcLoc.file_name(), srcLoc.line(), ln);
            positions.clear();
            normals.clear();
            texCoords.clear();
            groups.clear();
        };
        std::string lnBuf;
        while (std::getline(in, lnBuf)) {
            ++ln;

            if (lnBuf.empty())
                continue; // skip line "\n"

            auto offs = lnBuf.find_first_of(' ');
            if (offs == std::string::npos)
                offs = lnBuf.size();

            uint8_t tagIdx = 0;
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
            if (tagIdx == Tags.size()) {
                processParseErr();
                return;
            }

            switch (auto tag = static_cast<Tag>(tagIdx)) {
            case Tag::Group:
                groups.emplace_back(facePositionIndices.size());
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
                mtllibPath = getPathPrefix(objPath) + lnBuf.substr(offs + 1);
                break;
            case Tag::UseMaterial:
                grp2mtls.emplace(std::piecewise_construct, std::forward_as_tuple(groups.size() - 1),
                                 std::forward_as_tuple(lnBuf.substr(offs + 1)));
                break;
            }
        }

        isComplete = true;
    }

    bool IsComplete() const { return isComplete; }

#define CONST_REF_GETTER(member, memberNameInFunc)                                                 \
    const decltype(member) &Get##memberNameInFunc() const { return member; }
    CONST_REF_GETTER(positions, Positions)
    CONST_REF_GETTER(normals, Normals)
    CONST_REF_GETTER(texCoords, TextureCoordinates)
    CONST_REF_GETTER(groups, Groups)
    CONST_REF_GETTER(facePositionIndices, FacePositionIndices)
    CONST_REF_GETTER(faceTexCoordIndices, FaceTextureCoordinateIndices)
    CONST_REF_GETTER(faceNormalIndices, FaceNormalIndices)
    CONST_REF_GETTER(grp2mtls, GroupToMaterials)
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

        return mesh;
    }

  private:
    std::string getPathPrefix(const std::string &path) {
        auto lastSep = path.find_last_of('/');
        if (lastSep == std::string::npos)
            lastSep = path.find_last_of('\\');
        if (lastSep == std::string::npos)
            return "";

        return path.substr(0, lastSep + 1);
    }
};

} // namespace Data
} // namespace kouek

#endif // !KOUEK_MESH_H
