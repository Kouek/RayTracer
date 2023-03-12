#ifndef KOUEK_MESH_H
#define KOUEK_MESH_H

#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

#include <algorithm>
#include <array>
#include <unordered_map>
#include <vector>

#include <glm/gtc/matrix_transform.hpp>

namespace kouek {

class Mesh {
  public:
    static constexpr auto NoneIdx = std::numeric_limits<glm::uint>::max();

    struct Face2Idx3 {
        glm::uvec3 v;
        glm::uvec3 t;
        glm::uvec3 n;
    };

    struct Light {
        enum class Type { Quad, Sphere };
        Type type;
        float area;
        glm::vec3 radiance;
        union {
            struct {
                glm::vec3 pos;
                glm::vec3 uv[2];
            } quad;
            struct {
                glm::vec3 pos;
                float radius;
            } sphere;
        };
    };

    struct Material {
        glm::uint lightIdx;
        glm::vec3 kd;
        glm::vec3 ks;
        glm::vec3 tr;
        float ns;
        float ni;
    };

  private:
    glm::vec3 minPos, maxPos;

    std::vector<glm::vec3> vs;
    std::vector<glm::vec2> vts;
    std::vector<glm::vec3> vns;
    std::vector<Material> mtls;
    std::vector<Light> ls;
    std::vector<Face2Idx3> fs;
    std::vector<glm::uint> gs;
    std::vector<glm::uint> g2mtls;

    std::unordered_map<glm::uint, std::string> g2mtlNames;

  public:
    inline const auto &GetMinMaxPos() const {
        return std::make_tuple(minPos, maxPos);
    }
    inline const auto &GetVS() const { return vs; }
    inline const auto &GetVTS() const { return vts; }
    inline const auto &GetVNS() const { return vns; }
    inline const auto &GetMtls() const { return mtls; }
    inline const auto &GetFS() const { return fs; }
    inline const auto &GetGS() const { return gs; }
    inline const auto &GetLS() const { return ls; }
    inline const auto &GetG2Mtls() const { return g2mtls; }
    inline const auto &GetG2MtlNames() const { return g2mtlNames; }

    void LoadFromFile(const std::string &objPath,
                      const std::string &mtlPath = "", bool swapXYZ = false) {
        Clear();

        using namespace std;

        std::unordered_map<std::string, glm::uint> mtlName2IDs;
        readInMtl(mtlPath, mtlName2IDs);
        readInObj(objPath, mtlName2IDs, swapXYZ);
    }

    void
    SetLights(const std::vector<std::tuple<std::string, glm::vec3>> &lights) {
        ls.clear();
        ls.reserve(lights.size());

        auto addLight = [&](glm::uint gi, const glm::vec3 &radiance) {
            mtls[g2mtls[gi]].lightIdx = ls.size();
            ls.emplace_back();
            auto &lht = ls.back();

            lht.radiance = radiance;
            if (gs[gi + 1] - gs[gi] == 2) {
                lht.type == Light::Type::Quad;

                auto vi3 = [&]() {
                    std::array<glm::uint, 3> ret;
                    auto &vi3 = fs[gs[gi]].v;
                    std::array e3{vs[vi3[1]] - vs[vi3[0]],
                                  vs[vi3[2]] - vs[vi3[1]],
                                  vs[vi3[0]] - vs[vi3[2]]};

                    if (glm::dot(e3[0], e3[1]) <
                        std::numeric_limits<float>::epsilon()) {
                        ret[0] = vi3[0];
                        ret[1] = vi3[1];
                        ret[2] = vi3[2];
                    } else if (glm::dot(e3[1], e3[2]) <
                               std::numeric_limits<float>::epsilon()) {
                        ret[0] = vi3[1];
                        ret[1] = vi3[2];
                        ret[2] = vi3[0];
                    } else if (glm::dot(e3[2], e3[0]) <
                               std::numeric_limits<float>::epsilon()) {
                        ret[0] = vi3[2];
                        ret[1] = vi3[0];
                        ret[2] = vi3[1];
                    }
                    return ret;
                }();

                lht.quad.pos = vs[vi3[1]];
                lht.quad.uv[0] = vs[vi3[0]] - vs[vi3[1]];
                lht.quad.uv[1] = vs[vi3[2]] - vs[vi3[1]];

                lht.area =
                    glm::length(glm::cross(lht.quad.uv[0], lht.quad.uv[1]));
            } else {
                lht.type == Light::Type::Quad;

                lht.sphere.pos = glm::zero<glm::vec3>();
                float divGrp = 1.f / (gs[gi + 1] - gs[gi]);
                float divFace = 1.f / 3.f;
                for (glm::uint fi = gs[gi]; fi < gs[gi + 1]; ++fi) {
                    auto &vi3 = fs[fi].v;
                    auto mid = divFace * (vs[vi3[0]] + vs[vi3[1]] + vs[vi3[2]]);
                    lht.sphere.pos += divGrp * mid;
                }

                lht.sphere.radius = 0.f;
                for (glm::uint fi = gs[gi]; fi < gs[gi + 1]; ++fi) {
                    auto &vi3 = fs[fi].v;
                    for (uint8_t i = 0; i < 3; ++i) {
                        auto dist = glm::distance(vs[vi3[i]], lht.sphere.pos);
                        if (lht.sphere.radius < dist)
                            lht.sphere.radius = dist;
                    }
                }

                lht.area =
                    glm::pi<float>() * lht.sphere.radius * lht.sphere.radius;
            }
        };

        for (const auto &[name, radiance] : lights)
            for (const auto &[gi, _name] : g2mtlNames) {
                if (_name != name)
                    continue;
                addLight(gi, radiance);
            }
    }

    inline void Clear() {
        minPos = glm::vec3{std::numeric_limits<float>::max()};
        maxPos = glm::vec3{std::numeric_limits<float>::min()};

        vs.clear();
        vs.shrink_to_fit();
        vts.clear();
        vts.shrink_to_fit();
        vns.clear();
        vns.shrink_to_fit();
        mtls.clear();
        mtls.shrink_to_fit();
        fs.clear();
        fs.shrink_to_fit();
        ls.clear();
        ls.shrink_to_fit();
        gs.clear();
        gs.shrink_to_fit();
        g2mtls.clear();
        g2mtls.shrink_to_fit();

        g2mtlNames.clear();
    }

  private:
    void readInMtl(const std::string &path,
                   std::unordered_map<std::string, glm::uint> &mtlName2IDs) {
        using namespace std;

        if (path.empty())
            return;

        ifstream in(path.c_str());
        if (!in.is_open())
            throw std::runtime_error(
                string(path) +
                " is NOT a valid path. Load model material failed.");

        static constexpr auto Newmtl = std::string_view("newmtl ");
        static constexpr auto NewmtlLen = Newmtl.size();

        string buf;
        while (getline(in, buf)) {
            if (buf[0] == 'K' && buf[1] == 'd') {
                auto &v3 = mtls.back().kd;
                sscanf(buf.c_str() + 3, "%f%f%f", &v3.x, &v3.y, &v3.z);
            } else if (buf[0] == 'K' && buf[1] == 's') {
                auto &v3 = mtls.back().ks;
                sscanf(buf.c_str() + 3, "%f%f%f", &v3.x, &v3.y, &v3.z);

            } else if (buf[0] == 'T' && buf[1] == 'r') {
                auto &v3 = mtls.back().tr;
                sscanf(buf.c_str() + 3, "%f%f%f", &v3.x, &v3.y, &v3.z);

            } else if (buf[0] == 'N' && buf[1] == 's') {
                auto &s = mtls.back().ns;
                sscanf(buf.c_str() + 3, "%f", &s);
            } else if (buf[0] == 'N' && buf[1] == 'i') {
                auto &s = mtls.back().ni;
                sscanf(buf.c_str() + 3, "%f", &s);
            } else if (buf.substr(0, NewmtlLen) == Newmtl) {
                mtls.emplace_back();
                mtls.back().lightIdx = NoneIdx;
                mtlName2IDs.emplace(
                    std::piecewise_construct,
                    std::forward_as_tuple(buf.substr(NewmtlLen)),
                    std::forward_as_tuple((glm::uint)(mtls.size() - 1)));
            }
        }
    }

    void readInObj(const std::string &path,
                   std::unordered_map<std::string, glm::uint> &mtlName2IDs,
                   bool swapXYZ = false) {
        using namespace std;

        ifstream in(path.c_str());
        if (!in.is_open())
            throw std::runtime_error(
                string(path) + " is NOT a valid path. Load model obj failed.");

        static constexpr auto Usemtl = std::string_view("usemtl ");
        static constexpr auto UsemtlLen = Usemtl.size();

        string buf;
        Face2Idx3 f2i3;
        while (getline(in, buf)) {
            if (buf[0] == 'g')
                gs.emplace_back((glm::uint)fs.size());
            else if (buf[0] == 'v' && buf[1] == ' ') {
                double x, y, z;
                sscanf(buf.c_str() + 2, "%lf%lf%lf", &x, &y, &z);
                if (swapXYZ)
                    vs.emplace_back(z, x, y);
                else
                    vs.emplace_back(x, y, z);

                for (uint8_t xyz = 0; xyz < 3; ++xyz) {
                    if (minPos[xyz] > vs.back()[xyz])
                        minPos[xyz] = vs.back()[xyz];
                    if (maxPos[xyz] < vs.back()[xyz])
                        maxPos[xyz] = vs.back()[xyz];
                }
            } else if (buf[0] == 'v' && buf[1] == 't' && buf[2] == ' ') {
                double x, y;
                sscanf(buf.c_str() + 3, "%lf%lf", &x, &y);
                vts.emplace_back(x, y);
            } else if (buf[0] == 'v' && buf[1] == 'n' && buf[2] == ' ') {
                double x, y, z;
                sscanf(buf.c_str() + 3, "%lf%lf%lf", &x, &y, &z);
                if (swapXYZ)
                    vns.emplace_back(z, x, y);
                else
                    vns.emplace_back(x, y, z);
            } else if (buf[0] == 'f' && buf[1] == ' ') {
                std::array<uint32_t, 4> vIds{0, 0, 0, 0};
                std::array<uint32_t, 4> tIds{0, 0, 0, 0};
                std::array<uint32_t, 4> nIds{0, 0, 0, 0};

                sscanf(buf.c_str() + 2, "%d/%d/%d", &vIds[0], &tIds[0],
                       &nIds[0]);
                auto spIdx = buf.find(' ', 2);
                sscanf(buf.c_str() + spIdx + 1, "%d/%d/%d", &vIds[1], &tIds[1],
                       &nIds[1]);
                spIdx = buf.find(' ', spIdx + 1);
                sscanf(buf.c_str() + spIdx + 1, "%d/%d/%d", &vIds[2], &tIds[2],
                       &nIds[2]);
                spIdx = buf.find(' ', spIdx + 1);

                bool isQuad = false;
                if (spIdx < buf.size() && buf[spIdx + 1] != '\0') {
                    sscanf(buf.c_str() + spIdx + 1, "%d/%d/%d", &vIds[3],
                           &tIds[3], &nIds[3]);
                    isQuad = true;
                }

                for (uint8_t i = 0; i < 4; ++i) {
                    --vIds[i];
                    --tIds[i];
                    --nIds[i];
                }

                for (uint8_t i = 0; i < 3; ++i) {
                    f2i3.v[i] = vIds[i];
                    f2i3.t[i] = vIds[i];
                    f2i3.n[i] = vIds[i];
                }
                fs.emplace_back(f2i3);

                if (isQuad) {
                    for (uint8_t i = 0; i < 3; ++i) {
                        auto j = (i + 2) & 0b11;
                        f2i3.v[i] = vIds[j];
                        f2i3.t[i] = vIds[j];
                        f2i3.n[i] = vIds[j];
                    }
                    fs.emplace_back(f2i3);
                }
            } else if (buf.size() > UsemtlLen &&
                       buf.substr(0, UsemtlLen) == Usemtl) {
                if (gs.empty())
                    gs.emplace_back(0);
                glm::uint gi = gs.size() - 1;

                auto name = buf.substr(UsemtlLen);
                g2mtlNames.emplace(std::piecewise_construct,
                                   std::forward_as_tuple(gi),
                                   std::forward_as_tuple(name));

                auto itr = mtlName2IDs.find(g2mtlNames[gi]);
                if (itr == mtlName2IDs.end())
                    throw std::runtime_error(
                        "No material named " + g2mtlNames[gi] +
                        "exists for group " + std::to_string(gi) +
                        ". Load model failed.");
                else {
                    const auto &[name, id] = *itr;
                    g2mtls.emplace_back(id);
                }
            }
        }
        if (gs.empty())
            gs.emplace_back(0);
        gs.emplace_back(fs.size());

        in.close();

        if (vs.empty() || fs.empty())
            throw std::runtime_error(
                "Obj File has no vertices or faces. Load model failed.");
    }
};

} // namespace kouek

#endif // !KOUEK_MESH_H
