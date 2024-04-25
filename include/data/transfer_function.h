#ifndef KOUEK_DATA_TRANSFER_FUNCTION_H
#define KOUEK_DATA_TRANSFER_FUNCTION_H

#include <fstream>
#include <iostream>
#include <numeric>
#include <string>

#include <array>
#include <map>
#include <vector>

#include <glm/glm.hpp>

namespace kouek {
namespace Data {

template <typename T, T Min, T Max, T IncStep> class TransferFunction {
  public:
    static constexpr auto MaxScalar = Max;
    static constexpr auto MinScalar = Min;

  private:
    bool isComplete = false;

    std::array<T, 2> scalarRng;
    std::map<float, glm::vec4> pnts;
    std::vector<glm::vec4> dat;
    std::vector<glm::vec<2, T>> emptyScalarRngs;

    static constexpr std::string_view ErrTag = "[TransferFunction Error]";

  public:
    TransferFunction(const std::string &path) {
        scalarRng = {Max, Min};

        std::ifstream is(path, std::ios::in);
        if (!is.is_open()) {
            auto srcLoc = std::source_location::current();
            std::cerr << std::format("{} at {}:{}. Cannot open file at \"{}\".\n", ErrTag,
                                     srcLoc.file_name(), srcLoc.line(), path);
            return;
        }

        std::string buf;
        size_t ln = 0;
        while (std::getline(is, buf)) {
            float scalar;
            glm::vec4 col;
            auto cnt =
                sscanf(buf.c_str(), "%f%f%f%f%f", &scalar, &col[0], &col[1], &col[2], &col[3]);
            if (cnt != 5) {
                auto srcLoc = std::source_location::current();
                std::cerr << std::format(
                    "{} at {}:{}. Line {} of file \"{}\" is of invalid format.\n", ErrTag,
                    srcLoc.file_name(), srcLoc.line(), ln, path);
                return;
            }

            if (scalarRng[0] > scalar)
                scalarRng[0] = scalar;
            if (scalarRng[1] < scalar)
                scalarRng[1] = scalar;

            col /= 255.f;
            pnts.emplace(std::piecewise_construct, std::forward_as_tuple(scalar),
                         std::forward_as_tuple(col));

            ++ln;
        }

        is.close();

        if (scalarRng[0] != Min || scalarRng[1] != Max) {
            auto srcLoc = std::source_location::current();
            std::cerr << std::format("{} at {}:{}. File {} does not have enough TF points.\n",
                                     ErrTag, srcLoc.file_name(), srcLoc.line(), path);
            return;
        }
        computeDataFromPoints();

        isComplete = true;
    }

    bool IsComplete() const { return isComplete; }

#define CONST_REF_GETTER(member, memberNameInFunc)                                                 \
    const decltype(member) &Get##memberNameInFunc() const { return member; }
    CONST_REF_GETTER(dat, );
    CONST_REF_GETTER(pnts, Points);
    CONST_REF_GETTER(scalarRng, ScalarRange);
    CONST_REF_GETTER(emptyScalarRngs, EmptyScalarRanges);
#undef CONST_REF_GETTER

  private:
    void computeDataFromPoints() {
        dat.clear();
        dat.shrink_to_fit();
        dat.reserve(static_cast<size_t>((Max - Min) / IncStep));
        emptyScalarRngs.clear();
        emptyScalarRngs.shrink_to_fit();

        auto pnt = pnts.begin();
        auto prevPnt = pnts.begin();

        glm::vec2 emptyScalarRng = {Max, Min};
        for (auto scalar = Min; scalar < Max; scalar += IncStep) {
            if (scalar > pnt->first)
                prevPnt = pnt++;
            auto &[s, c] = *pnt;
            auto &[prevS, prevC] = *prevPnt;

            if (prevC.a == 0.f && c.a == 0.f) {
                if (emptyScalarRng[0] > prevS)
                    emptyScalarRng[0] = prevS;
                if (emptyScalarRng[1] < s)
                    emptyScalarRng[1] = s;
            } else if (emptyScalarRng[0] != Max && emptyScalarRng[1] != Min) {
                emptyScalarRngs.emplace_back(emptyScalarRng);
                emptyScalarRng = {Max, Min};
            }

            auto a = s == prevS ? 1.f : (scalar - prevS) / (s - prevS);
            dat.emplace_back((1.f - a) * prevC + a * c);
        }
        dat.emplace_back(pnt->second);
    }
};

} // namespace Data
} // namespace kouek

#endif // !KOUEK_DATA_TRANSFER_FUNCTION_H
