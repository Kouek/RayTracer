#include <algorithm>
#include <format>
#include <iostream>
#include <string_view>

#include <cmdparser.hpp>

#include <cuda/texture.h>
#include <data/raw_volume.h>
#include <data/transfer_function.h>
#include <ray_caster/dpbx_vdb.h>
#include <ray_caster/ray_caster.h>

#include "glfw_gl_app.h"

inline cli::Parser parseCommand(int argc, char **argv) {
    cli::Parser parser(argc, argv);

    parser.set_required<std::string>("vt", "vox-ty", "Type of voxel");
    parser.set_required<std::string>("vol", "volume", "Path of raw volume file");
    parser.set_required<uint32_t>("dx", "dim-x",
                                  "Dimension of volume on X-axis (before axis transformation)");
    parser.set_required<uint32_t>("dy", "dim-y",
                                  "Dimension of volume on Y-axis (before axis transformation)");
    parser.set_required<uint32_t>("dz", "dim-z",
                                  "Dimension of volume on Z-axis (before axis transformation)");
    parser.set_optional<std::string>("tf", "transfer-func", "",
                                     "Path of TXT file storing transfer function");
    parser.set_optional<float>("sx", "spacing-x", 1.f, "Spacing of voxel on X-axis");
    parser.set_optional<float>("sy", "spacing-y", 1.f, "Spacing of voxel on Y-axis");
    parser.set_optional<float>("sz", "spacing-z", 1.f, "Spacing of voxel on Z-axis");
    parser.set_optional<uint8_t>("ld0", "log2-dim-0", 5, "Log2-dimension of VDB at level 0");
    parser.set_optional<uint8_t>("ld1", "log2-dim-1", 4, "Log2-dimension of VDB at level 1");
    parser.set_optional<uint8_t>("ld2", "log2-dim-2", 3, "Log2-dimension of VDB at level 2");
    {
        constexpr std::string_view trDesc("Transform of {}-axis, 1 for X, 2 for Y and 3 for Z, "
                                          "while -1 for -X, -2 for -Y and -3 for -Z");
        parser.set_optional<int8_t>("tx", "trans-x", 1, std::format(trDesc, 'X'));
        parser.set_optional<int8_t>("ty", "trans-y", 2, std::format(trDesc, 'Y'));
        parser.set_optional<int8_t>("tz", "trans-z", 3, std::format(trDesc, 'Z'));
    }

    return parser;
}

static std::array<uint8_t, kouek::RayCaster::DepthBoxVDB::MaxLevelNum> vdbLog2Dims;
static int rndrTarget[2] = {0};
static int vdbRootLev = 0;
static int vdbDisplayLev = 0;
static int vdbDisplayLeafLayer = 0;
static bool rndrRAW = false;
static bool useDepthBox = true;
static bool inPerfTest = false;
static bool perfTestInVol = false;
static bool showFPS = true;
static float samplingRate = 3.f;
static float currentFPS = 0.f;
static const char *rndrTargetNames[2] = {"Scene\0AABB",
                                         "Scene\0AABBs\0Depths\0Positions after Depth-skip"};

static struct {
    bool used = false;
    float ka = .5f;
    float kd = .4f;
    float ks = .4f;
    float shiness = 8.f;
    glm::vec3 lightPos = {0.f, 2.f, 0.f};
    glm::vec3 lightCol = {1.f, 1.f, 1.f};
} shading;

static std::function<void(void)> onSliderChanged_VDBDisplayLev;
static std::function<void(void)> onInputIntChanged_VDBDisplayLeafLayer;
static std::function<void(void)> onInputFloatChanged_SamplingRate;
static std::function<void(void)> onShadingChanged;
static std::function<bool(void)> onVDBChanged;

static std::function<void(void)> onPerfTestStarted;

inline void drawUI() {
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    ImGui::Begin("Statistics");
    ImGui::Checkbox("Show FPS", &showFPS);
    if (showFPS)
        ImGui::LabelText("FPS", "%.2f", currentFPS);
    ImGui::End();

    ImGui::Begin("Rendering");
    ImGui::Checkbox("Render RAW Volume without VDB", &rndrRAW);
    if (rndrRAW)
        ImGui::Combo("Render Target", &rndrTarget[0], rndrTargetNames[0]);
    else {
        ImGui::Combo("Render Target", &rndrTarget[1], rndrTargetNames[1]);
        if (rndrTarget[1] ==
            static_cast<int>(kouek::RayCaster::RayCaster::DPBXVDBRenderTarget::AABBs)) {
            if (ImGui::SliderInt("Display Level", &vdbDisplayLev, 0, vdbRootLev))
                onSliderChanged_VDBDisplayLev();
        }
        if (rndrTarget[1] ==
                static_cast<int>(kouek::RayCaster::RayCaster::DPBXVDBRenderTarget::Depths) ||
            rndrTarget[1] ==
                static_cast<int>(
                    kouek::RayCaster::RayCaster::DPBXVDBRenderTarget::PositionsAfterDepthSkip)) {
            if (ImGui::InputInt("Display Leaf Layer", &vdbDisplayLeafLayer))
                onInputIntChanged_VDBDisplayLeafLayer();
        }
    }
    if (ImGui::InputFloat("Sampling Rate", &samplingRate, .1f, 1.f)) {
        if (samplingRate < .1f)
            samplingRate = .1f;
        else if (samplingRate > 10.f)
            samplingRate = 10.f;

        onInputFloatChanged_SamplingRate();
    }
    if ([&]() {
            bool changed = false;
            changed |= ImGui::Checkbox("Use Shading", &shading.used);
            if (shading.used) {
                auto inputFloat = [&](float *ptr, const char *name, float min, float max) {
                    if (ImGui::InputFloat(name, ptr, (max - min) / 100.f, (max - min) / 10.f))
                        if (*ptr < min)
                            *ptr = min;
                        else if (*ptr > max)
                            *ptr = max;
                    changed = true;
                };
                inputFloat(&shading.ka, "ka", 0.f, 10.f);
                inputFloat(&shading.kd, "kd", 0.f, 10.f);
                inputFloat(&shading.ks, "ks", 0.f, 10.f);
                inputFloat(&shading.shiness, "shiness", 1.f, 1024.f);
            }
            return changed;
        }())
        onShadingChanged();
    if (ImGui::Button("Start Perf Test")) {
        if (!inPerfTest)
            onPerfTestStarted();
    }
    ImGui::Checkbox("Perf Test into the Volume", &perfTestInVol);
    ImGui::End();

    ImGui::Begin("VDB");
    if ([&]() {
            bool changed = false;
            changed |= ImGui::Checkbox("Use Depth Box", &useDepthBox);
            static std::array<int, kouek::RayCaster::DepthBoxVDB::MaxLevelNum> log2Dims = {
                vdbLog2Dims[0], vdbLog2Dims[1], vdbLog2Dims[2]};
            char name[] = "levx";
            for (uint8_t lev = 0; lev < log2Dims.size(); ++lev) {
                name[3] = lev + '0';
                if (ImGui::InputInt(name, &log2Dims[lev])) {
                    if (log2Dims[lev] == 0)
                        log2Dims[lev] = 1;
                    else if (log2Dims[lev] > kouek::RayCaster::DepthBoxVDB::MaxLog2Dim)
                        log2Dims[lev] = kouek::RayCaster::DepthBoxVDB::MaxLog2Dim;
                    vdbLog2Dims[lev] = log2Dims[lev];
                    changed = true;
                }
            }
            return changed;
        }())
        onVDBChanged();
    ImGui::End();

    ImGui::EndFrame();
    ImGui::Render();

    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

template <typename VoxelType> int run(const cli::Parser &parser) {
    vdbLog2Dims[0] = parser.get<uint8_t>("ld0");
    vdbLog2Dims[1] = parser.get<uint8_t>("ld1");
    vdbLog2Dims[2] = parser.get<uint8_t>("ld2");
    for (auto log2Dim : vdbLog2Dims)
        if (log2Dim == 0 || log2Dim > kouek::RayCaster::DepthBoxVDB::MaxLog2Dim) {
            std::cerr << "Invalid log2-dimension input.\n";
            return 1;
        }

    auto vol = [&]() {
        auto volPath = parser.get<std::string>("vol");
        glm::vec<3, uint32_t> dim;
        dim.x = parser.get<uint32_t>("dx");
        dim.y = parser.get<uint32_t>("dy");
        dim.z = parser.get<uint32_t>("dz");
        glm::i8vec3 trAxis;
        trAxis.x = parser.get<int8_t>("tx");
        trAxis.y = parser.get<int8_t>("ty");
        trAxis.z = parser.get<int8_t>("tz");

        kouek::Data::RAWVolume<VoxelType> vol(volPath, dim, trAxis);
        return vol;
    }();
    if (!vol.IsComplete()) {
        std::cerr << vol.GetErrorMessage();
        return 1;
    }

    auto trDim = vol.GetDimension();
    auto spaces = [&]() {
        glm::vec3 spaces;
        spaces.x = parser.get<float>("sx");
        spaces.y = parser.get<float>("sy");
        spaces.z = parser.get<float>("sz");
        return spaces;
    }();
    auto invSpaces = 1.f / spaces;

    auto tf = [&]() {
        auto tfPath = parser.get<std::string>("tf");
        kouek::Data::TransferFunction<VoxelType, 0, std::numeric_limits<VoxelType>::max(), 1> tf(
            tfPath);
        return tf;
    }();
    if (!tf.IsComplete()) {
        std::cerr << tf.GetErrorMessage();
        return 1;
    }

    std::shared_ptr<kouek::CUDA::Array> volArr;
    std::shared_ptr<kouek::CUDA::Texture> volTex;
    std::shared_ptr<kouek::CUDA::Texture> volTexPointFilter;
    {
        volArr = std::make_shared<decltype(volArr)::element_type>(vol.Get(), vol.GetDimension());
        if (!volArr->IsComplete())
            return 1;
        volTex = std::make_shared<decltype(volTex)::element_type>(volArr);
        if (!volTex->IsComplete())
            return 1;
        volTexPointFilter =
            std::make_shared<decltype(volTexPointFilter)::element_type>(volArr, [&]() {
                cudaTextureDesc texDesc;
                memset(&texDesc, 0, sizeof(texDesc));
                texDesc.normalizedCoords = 0;
                texDesc.filterMode = cudaFilterModePoint;
                texDesc.addressMode[0] = texDesc.addressMode[1] = texDesc.addressMode[2] =
                    cudaAddressModeBorder;
                texDesc.readMode = cudaReadModeElementType;
                return texDesc;
            }());
        if (!volTexPointFilter->IsComplete())
            return 1;
    }

    std::shared_ptr<kouek::CUDA::Array> tfArr;
    std::shared_ptr<kouek::CUDA::Texture> tfTex;
    {
        tfArr = std::make_shared<decltype(tfArr)::element_type>(
            tf.Get(),
            glm::vec<3, uint32_t>{decltype(tf)::MaxScalar - decltype(tf)::MinScalar + 1, 0, 0},
            cudaCreateChannelDesc(8 * sizeof(float), 8 * sizeof(float), 8 * sizeof(float),
                                  8 * sizeof(float), cudaChannelFormatKindFloat));
        if (!tfArr->IsComplete())
            return 1;

        tfTex = std::make_shared<decltype(tfTex)::element_type>(tfArr, []() {
            cudaTextureDesc texDesc;
            memset(&texDesc, 0, sizeof(texDesc));
            texDesc.normalizedCoords = 1;
            texDesc.filterMode = cudaFilterModeLinear;
            texDesc.addressMode[0] = texDesc.addressMode[1] = texDesc.addressMode[2] =
                cudaAddressModeBorder;
            texDesc.readMode = cudaReadModeElementType;
            return texDesc;
        }());
        if (!tfTex->IsComplete())
            return 1;
    }

    kouek::RayCaster::RayCaster rayCaster;
    auto w2s = glm::translate(glm::identity<glm::mat4>(), glm::vec3(.5f));
    rayCaster.SetWorldToScene(w2s);
    rayCaster.SetRAWVolume(volTex);
    rayCaster.SetTransferFunctionTexture(tfTex);

    onSliderChanged_VDBDisplayLev = [&]() { rayCaster.SetDepthBoxVDBDisplayLevel(vdbDisplayLev); };
    onInputIntChanged_VDBDisplayLeafLayer = [&]() {
        rayCaster.SetDepthBoxVDBDisplayLeafLayer(vdbDisplayLeafLayer);
    };
    onInputFloatChanged_SamplingRate = [&]() { rayCaster.SetSamplingRate(samplingRate); };
    onVDBChanged = [&]() {
        auto vdb = std::make_shared<kouek::RayCaster::DepthBoxVDB>();
        vdb->BuildFrom<VoxelType>({.apronWid = 1,
                                   .log2Dims = vdbLog2Dims,
                                   .useDepthBox = useDepthBox,
                                   .volTex = *volTexPointFilter,
                                   .emptyScalarRngs = tf.GetEmptyScalarRanges()});
        if (!vdb->IsComplete()) {
            std::cerr << vdb->GetErrorMessage();
            return false;
        }
        vdbRootLev = vdb->GetDeviceData().vdbParam.rootLev;

        rayCaster.SetDepthBoxVDB(vdb);
        return true;
    };
    onShadingChanged = [&]() {
        if (shading.used)
            rayCaster.SetShading(
                {.ka = shading.ka,
                 .kd = shading.kd,
                 .ks = shading.ks,
                 .shiness = shading.shiness,
                 .lightPosInVolumeSpace =
                     glm::vec3{trDim} * glm::vec3{w2s * glm::vec4{shading.lightPos, 1.f}},
                 .lightCol = shading.lightCol});
        else
            rayCaster.UnsetShading();
    };

    onSliderChanged_VDBDisplayLev();
    onInputIntChanged_VDBDisplayLeafLayer();
    onInputFloatChanged_SamplingRate();
    if (!onVDBChanged())
        return 1;
    onShadingChanged();

    GLFWxGLxCUDAApp app(/*default window size*/ {800, 700},
                        /*window title*/ "Direct Volume Render",
                        /*resized callback*/
                        [&](const glm::ivec2 &rndrSz) {
                            auto proj = glm::perspectiveFov(glm::radians(90.f),
                                                            static_cast<float>(rndrSz.x),
                                                            static_cast<float>(rndrSz.y), 1.f, 3.f);
                            rayCaster.SetProjection(proj);

                            std::cout << std::format("Render size is {}x{}.\n", rndrSz.x, rndrSz.y);
                        },
                        /*camera changed callback*/
                        [&](const kouek::FPSCamera &cam) {
                            const auto &[R, F, U, P] = cam.GetRFUP();
                            glm::mat3 rot(R, U, -F);
                            rayCaster.SetCameraToWorld(P, rot);
                        });
    if (!app.ok)
        return 1;
    app.camera.LookAt({0.f, 0.f, 1.f}, {0.f, 0.f, 0.f});
    app.movSens = .01f;
    app.callerOnCameraChanged(app.camera);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float perfCost;
    float perfRotDeg;
    kouek::FPSCamera perfCam;
    static constexpr auto PerfRotDltDeg = .5f;
    onPerfTestStarted = [&]() {
        perfCost = 0.f;
        perfRotDeg = 0.f;
        perfCam.LookAt({0.f, 0.f, perfTestInVol ? .2f : 1.f}, {0.f, 0.f, 0.f});
        app.banInput = true;

        inPerfTest = true;
    };

    app.Run([&]() {
        if (inPerfTest) {
            cudaEventRecord(start);
            perfCam.Revolve(glm::distance(perfCam.GetPos(), glm::zero<glm::vec3>()),
                            static_cast<float>(PerfRotDltDeg), 0.f);
            const auto &[R, F, U, P] = perfCam.GetRFUP();
            glm::mat3 rot(R, U, -F);
            rayCaster.SetCameraToWorld(P, rot);

            perfRotDeg += PerfRotDltDeg;
        } else if (showFPS)
            cudaEventRecord(start);

        auto surfRndtTo = app.MapGLResourceToCUDA();
        if (rndrRAW)
            rayCaster.RenderRAWVolume(
                surfRndtTo, app.rndrSz,
                static_cast<kouek::RayCaster::RayCaster::RAWVolumeRenderTarget>(rndrTarget[0]));
        else
            rayCaster.RenderDepthBoxVDB<VoxelType>(
                surfRndtTo, app.rndrSz,
                static_cast<kouek::RayCaster::RayCaster::DPBXVDBRenderTarget>(rndrTarget[1]));
        app.UnmapGLResourceFromCUDA(surfRndtTo);

        if (inPerfTest) {
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&currentFPS, start, stop);
            perfCost += currentFPS;
            currentFPS = 1000.f / currentFPS;

            if (perfRotDeg >= 360.f) {
                std::cout << std::format(
                    "[Perf] {}ms costed, avg {} FPS with render size {}x{}, VDB {}x{}x{}, "
                    "shainding {}, depth box {}.\n",
                    perfCost, (perfRotDeg / PerfRotDltDeg) * 1000.f / perfCost, app.rndrSz.x,
                    app.rndrSz.y, vdbLog2Dims[0], vdbLog2Dims[1], vdbLog2Dims[2],
                    shading.used ? "ON" : "OFF", useDepthBox ? "ON" : "OFF");

                app.banInput = false;
                inPerfTest = false;
                app.callerOnCameraChanged(app.camera);
            }
        } else if (showFPS) {
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&currentFPS, start, stop);
            currentFPS = 1000.f / currentFPS;
        }

        glBindFramebuffer(GL_READ_FRAMEBUFFER, app.offScrnFBO);
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
        glBlitFramebuffer(0, 0, app.rndrSz.x, app.rndrSz.y, 0, 0, app.rndrSz.x, app.rndrSz.y,
                          GL_COLOR_BUFFER_BIT, GL_NEAREST);

        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        drawUI();
    });

    return 0;
}

enum class SupportedVoxelTypes { UInt8, UInt16 };
inline SupportedVoxelTypes fromNameToVoxelType(const std::string &name) {
    auto lowercase = name;
    std::transform(lowercase.begin(), lowercase.end(), lowercase.begin(),
                   [](unsigned char ch) { return std::tolower(ch); });

    if (lowercase == "uint8")
        return SupportedVoxelTypes::UInt8;
    if (lowercase == "uint16")
        return SupportedVoxelTypes::UInt16;
}

int main(int argc, char **argv) {
    auto parser = parseCommand(argc, argv);
    parser.run_and_exit_if_error();

    switch (fromNameToVoxelType(parser.get<std::string>("vt"))) {
    case SupportedVoxelTypes::UInt8:
        return run<uint8_t>(parser);
    case SupportedVoxelTypes::UInt16:
        return run<uint16_t>(parser);
    default:
        std::cerr << std::format("Invalid voxel type input.\n");
        return 1;
    }
}
