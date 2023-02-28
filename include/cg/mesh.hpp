#ifndef KOUEK_MESH_H
#define KOUEK_MESH_H

#include <fstream>
#include <stdexcept>
#include <string>
#include <string_view>

#include <array>
#include <unordered_map>
#include <vector>

#include <glad/glad.h>
#include <glm/glm.hpp>

namespace kouek {

class Mesh {
  public:
    struct Face2Idx3 {
        glm::uvec3 v;
        glm::uvec3 t;
        glm::uvec3 n;
    };

  private:
    glm::vec3 minPos, maxPos;

    std::vector<glm::vec3> vs;
    std::vector<glm::vec2> vts;
    std::vector<glm::vec3> vns;
    std::vector<Face2Idx3> fs;
    std::vector<glm::uint> gs;

    std::unordered_map<glm::uint, std::string> grp2MatrNames;

  public:
    inline const auto &GetMinMaxPos() const {
        return std::make_tuple(minPos, maxPos);
    }
    inline const auto &GetVS() const { return vs; }
    inline const auto &GetVTS() const { return vts; }
    inline const auto &GetVNS() const { return vns; }
    inline const auto &GetFS() const { return fs; }
    inline const auto &GetGS() const { return gs; }
    inline const auto &GetGrp2MatrNames() const { return grp2MatrNames; }
    void LoadFromFile(const std::string &path, bool swapXYZ = false) {
        Clear();

        using namespace std;
        ifstream in(path.c_str());

        if (!in.is_open())
            throw std::runtime_error(
                string(path) + " is NOT a valid path. Load model failed.");

        static constexpr auto UsemtlLen = std::string_view("usemtl ").size();

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
                       buf.substr(0, UsemtlLen) == "usemtl ") {
                if (gs.empty())
                    gs.emplace_back(0);
                grp2MatrNames.emplace(
                    std::piecewise_construct,
                    std::forward_as_tuple(gs.size() - 1),
                    std::forward_as_tuple(buf.substr(UsemtlLen)));
            }
        }
        if (gs.empty())
            gs.emplace_back(0);
        gs.emplace_back(fs.size());

        in.close();

        if (vs.empty() || fs.empty())
            throw std::runtime_error(
                "File has no vertices or faces. Load model failed.");
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
        fs.clear();
        fs.shrink_to_fit();
    }
};

class Model {};

} // namespace kouek

#endif // !KOUEK_MESH_H
