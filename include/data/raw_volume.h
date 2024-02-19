#ifndef KOUEK_RAW_VOLUME_H
#define KOUEK_RAW_VOLUME_H

#include <fstream>
#include <iostream>
#include <source_location>
#include <string_view>

#include <array>
#include <vector>

#include <glm/gtc/matrix_transform.hpp>

namespace kouek {
namespace Data {

template <typename T> class RAWVolume {
  private:
    bool isComplete = false;

    glm::vec<3, uint32_t> dim;
    std::vector<T> dat;

    static constexpr std::string_view ErrTag = "[RAWVolume Error]";

  public:
    RAWVolume(const std::string &path, const glm::vec<3, uint32_t> &dim,
              const glm::i8vec3 &trAxis = {1, 2, 3}) {
        auto voxNum = static_cast<size_t>(dim.x) * dim.y * dim.z;
        auto volSz = sizeof(T) * voxNum;

        dat.resize(voxNum);

        if (auto check =
                [&]() {
                    std::array<uint8_t, 3> cnt{0};
                    for (uint8_t xyz = 0; xyz < 3; ++xyz) {
                        auto ax = std::abs(trAxis[xyz]);
                        if (ax > 3)
                            return false;
                        else
                            ++cnt[ax - 1];
                    }
                    return cnt[0] == 1 && cnt[1] == 1 && cnt[2] == 1;
                }();
            !check) {
            auto srcLoc = std::source_location::current();
            std::cerr << std::format("{} at {}:{}. Parameter trAxis is invalid.\n", ErrTag,
                                     srcLoc.file_name(), srcLoc.line());
            return;
        }

        std::ifstream is(path, std::ios::in | std::ios::binary | std::ios::ate);
        if (!is.is_open()) {
            auto srcLoc = std::source_location::current();
            std::cerr << std::format("{} at {}:{}. Cannot open file at \"{}\".\n", ErrTag,
                                     srcLoc.file_name(), srcLoc.line(), path);
            return;
        }

        if (is.tellg() < volSz) {
            auto srcLoc = std::source_location::current();
            std::cerr << std::format(
                "{} at {}:{}. File {} is too small to contain volume of size ({},{},{})\n", ErrTag,
                srcLoc.file_name(), srcLoc.line(), path, dim.x, dim.y, dim.z);
            goto TERMINAL;
        }
        is.seekg(0);

        if (trAxis[0] == 1 && trAxis[1] == 2 && trAxis[2] == 3) {
            this->dim = dim;
            is.read(reinterpret_cast<char *>(dat.data()), volSz);
        } else {
            decltype(dat) tmp(voxNum);
            is.read(reinterpret_cast<char *>(tmp.data()), volSz);

            glm::u8vec3 trAxisMap = {std::abs(trAxis.x) - 1, std::abs(trAxis.y) - 1,
                                     std::abs(trAxis.z) - 1};
            this->dim = {dim[trAxisMap.x], dim[trAxisMap.y], dim[trAxisMap.z]};

            auto trVoxYxX = static_cast<size_t>(this->dim.y) * this->dim.x;
            size_t offs = 0;
            glm::vec<3, uint32_t> coord;
            for (coord.z = 0; coord.z < dim.z; ++coord.z)
                for (coord.y = 0; coord.y < dim.y; ++coord.y)
                    for (coord.x = 0; coord.x < dim.x; ++coord.x) {
                        glm::vec<3, uint32_t> trCoord = {
                            trAxis.x > 0 ? coord[trAxisMap.x]
                                         : this->dim.x - 1 - coord[trAxisMap.x],
                            trAxis.y > 0 ? coord[trAxisMap.y]
                                         : this->dim.y - 1 - coord[trAxisMap.y],
                            trAxis.z > 0 ? coord[trAxisMap.z]
                                         : this->dim.z - 1 - coord[trAxisMap.z]};
                        dat[trCoord.z * trVoxYxX + trCoord.y * this->dim.x + trCoord.x] = tmp[offs];
                        ++offs;
                    }
        }

        isComplete = true;

    TERMINAL:
        is.close();
    }

    bool IsComplete() const { return isComplete; }

#define CONST_REF_GETTER(member, memberNameInFunc)                                                 \
    const decltype(member) &Get##memberNameInFunc() const { return member; }
    CONST_REF_GETTER(dat, );
    CONST_REF_GETTER(dim, Dimension);
#undef CONST_REF_GETTER
};

} // namespace Data
} // namespace kouek

#endif // !KOUEK_RAW_VOLUME_H
