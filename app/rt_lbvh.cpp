#include <iostream>
#include <memory>

#include <cg/FPS_camera.h>
#include <data/mesh.h>
#include <ray_tracer/lbvh.h>
#include <ray_tracer/ray_tracer.h>

#include <cmdparser.hpp>
#include <tinyxml/tinyxml2.h>

#include "glfw_gl_app.h"

static kouek::FPSCamera camera;

static bool showFPS = true;
static float currentFPS = 0.f;
static float currentSPP = 0.f;
static float maxSPP = 0.f;
static int maxDepthPerPath = (kouek::RayTracer::RayTracer::RenderParameter().maxPathDepth);
static int rndrTarget = 0;
static int displayTreeHeight = (kouek::RayTracer::RayTracer::RenderParameter().displayTreeHeight);

static const char *rndrTargetNames =
    "Scene\0AABBs\0Triangles\0Lights\0Positions\0Normals\0Texture Coords";

static std::function<void(void)> onDisplayHeightChanged;
static std::function<void(void)> onMaxDepthPerPathChanged;

void drawUI() {
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    ImGui::Begin("Statistics");
    ImGui::Checkbox("Show FPS", &showFPS);
    if (showFPS)
        ImGui::LabelText("FPS", "%.2f", currentFPS);
    ImGui::LabelText("SPP", "%.2f / %.2f", currentSPP, maxSPP);
    ImGui::End();

    ImGui::Begin("Rendering");
    ImGui::Combo("Render Target", &rndrTarget, rndrTargetNames);
    if (rndrTarget ==
        static_cast<decltype(rndrTarget)>(kouek::RayTracer::RayTracer::RenderTarget::AABBs))
        if (ImGui::SliderInt("Display Height", &displayTreeHeight, 0, 29))
            onDisplayHeightChanged();
    if (rndrTarget ==
        static_cast<decltype(rndrTarget)>(kouek::RayTracer::RayTracer::RenderTarget::Scene))
        if (ImGui::SliderInt("Max Depth per Path", &maxDepthPerPath, 1, 16))
            onMaxDepthPerPathChanged();
    ImGui::End();

    ImGui::EndFrame();
    ImGui::Render();

    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

struct SceneConfig {
    bool isComplete = false;
    glm::ivec2 rndrSz;
    glm::vec3 eye, cntr, up;

    std::unordered_map<std::string, kouek::Data::OBJMesh::InputLight> lights;

    SceneConfig(const std::filesystem::path &objPath) {
        auto path = objPath.parent_path() / (objPath.stem().string() + ".xml");

        tinyxml2::XMLDocument xml;
        if (xml.LoadFile(path.string().c_str()) != tinyxml2::XML_SUCCESS)
            return;
        auto cam = xml.FirstChildElement("camera");
        if (!cam)
            return;

        auto cPtr = cam->Attribute("type");
        if (!cPtr)
            return;
        if (strcmp(cPtr, "perspective") != 0)
            return;

        cPtr = cam->Attribute("width");
        if (!cPtr || sscanf(cPtr, "%d", &rndrSz.x) != 1)
            return;
        cPtr = cam->Attribute("height");
        if (!cPtr || sscanf(cPtr, "%d", &rndrSz.y) != 1)
            return;

        auto parse = [&](glm::vec3 &v3, const char *name) {
            auto tmp = cam->FirstChildElement(name);
            if (!tmp)
                return false;

            auto parse = [&](float &v, const char *name) {
                auto cPtr = tmp->Attribute(name);
                if (!cPtr)
                    return false;
                if (sscanf(cPtr, "%f", &v) != 1)
                    return false;
                return true;
            };
            auto ok = true;
            ok &= parse(v3.x, "x");
            ok &= parse(v3.y, "y");
            ok &= parse(v3.z, "z");
            return ok;
        };
        if (!parse(eye, "eye"))
            return;
        if (!parse(cntr, "lookat"))
            return;
        if (!parse(up, "up"))
            return;

        auto lht = xml.FirstChildElement("light");
        while (lht) {
            kouek::Data::OBJMesh::InputLight inLht;

            cPtr = lht->Attribute("type");
            if (!cPtr)
                return;
            if (strcmp(cPtr, "Quad") == 0)
                inLht.type = kouek::Data::OBJMesh::Light::Type::Quad;
            else if (strcmp(cPtr, "Sphere") == 0)
                inLht.type = kouek::Data::OBJMesh::Light::Type::Sphere;
            else
                return;

            cPtr = lht->Attribute("radiance");
            if (!cPtr || sscanf(cPtr, "%f,%f,%f", &inLht.radiance.r, &inLht.radiance.g,
                                &inLht.radiance.b) != 3)
                return;

            cPtr = lht->Attribute("mtlname");
            if (!cPtr)
                return;

            lights.emplace(std::piecewise_construct, std::forward_as_tuple(cPtr),
                           std::forward_as_tuple(inLht));
            lht = lht->NextSiblingElement("light");
        }

        isComplete = true;
    }
};

int main(int argc, char **argv) {
    auto cmdParser = cli::Parser(argc, argv);
    cmdParser.set_required<std::string>("obj", "obj-path", "The path of the file of the obj.");
    cmdParser.set_optional<std::string>("lbvh-lg", "lbvh-log-path", "",
                                        "The path of the log-file of LBVH.");
    cmdParser.run_and_exit_if_error();

    std::filesystem::path objPath(cmdParser.get<std::string>("obj"));

    SceneConfig scnCfg(objPath);
    if (!scnCfg.isComplete) {
        std::cerr << std::format("[App Error]. Cannot open scene config.\n");
        return 1;
    }

    kouek::Data::OBJMesh mesh(objPath, scnCfg.lights);
    if (!mesh.IsComplete())
        return 1;

    auto lbvh = std::make_shared<kouek::RayTracer::LBVH>();
    lbvh->BuildFrom(
        {.positions = mesh.GetPositions(), .facePositionIndices = mesh.GetFacePositionIndices()},
        cmdParser.get<std::string>("lbvh-lg"));
    if (!lbvh->IsComplete())
        return 1;

    auto deviceDat = lbvh->GetDeviceData();

    kouek::RayTracer::RayTracer rayTracer;
    rayTracer.SetWorldToScene(glm::identity<glm::mat4>());
    rayTracer.SetMesh(
        {.positions = mesh.GetPositions(),
         .normals = mesh.GetNormals(),
         .texCoords = mesh.GetTextureCoordinates(),
         .groupStartFaceIndices = mesh.GetGroupStartFaceIndices(),
         .facePositionIndices = mesh.GetFacePositionIndices(),
         .faceNormalIndices = mesh.GetFaceNormalIndices(),
         .faceTexCoordIndices = mesh.GetFaceTextureCoordinateIndices(),
         .lights =
             [&]() {
                 auto &objLhts = mesh.GetLights();
                 std::vector<kouek::RayTracer::Light> lhts;
                 lhts.reserve(objLhts.size());
                 for (auto &objLht : objLhts) {
                     auto &lht = lhts.emplace_back();

                     if (objLht.type == kouek::Data::OBJMesh::Light::Type::Quad)
                         lht = kouek::RayTracer::Light::CreateQuad(objLht.quad.o, objLht.quad.u,
                                                                   objLht.quad.v, objLht.radiance);
                     else
                         lht = kouek::RayTracer::Light::CreateSphere(
                             objLht.sphere.r, objLht.sphere.o, objLht.radiance);
                 }

                 return lhts;
             }(),
         .materials =
             [&]() {
                 auto &grp2mtlNames = mesh.GetGroupToMaterialNames();
                 auto &name2mtls = mesh.GetNameToMaterials();
                 auto &scnLhts = scnCfg.lights;

                 std::vector<kouek::RayTracer::Material> matrs;
                 matrs.resize(grp2mtlNames.size());
                 for (auto &[gi, mtlName] : grp2mtlNames) {
                     auto &objMatr = name2mtls.at(mtlName);

                     auto lhtItr = scnLhts.find(mtlName);
                     if (lhtItr == scnLhts.end())
                         matrs[gi] = kouek::RayTracer::Material::Create(
                             objMatr.kd, objMatr.ks, objMatr.tr, objMatr.ni, objMatr.ns);
                     else
                         matrs[gi] = kouek::RayTracer::Material::Create(
                             objMatr.kd, objMatr.ks, objMatr.tr, objMatr.ni, objMatr.ns,
                             lhtItr->second.radiance);
                 }

                 return matrs;
             }()});
    rayTracer.SetLBVH(lbvh);

    onDisplayHeightChanged = [&]() { rayTracer.SetDisplayTreeHeight(displayTreeHeight); };
    onMaxDepthPerPathChanged = [&]() { rayTracer.SetMaxPathDepth(maxDepthPerPath); };

    onDisplayHeightChanged();
    onMaxDepthPerPathChanged();

    GLFWxGLxCUDAApp app(/*default window size*/ scnCfg.rndrSz,
                        /*window title*/ "Path Tracer",
                        /*resized callback*/
                        [&](const glm::ivec2 &rndrSz) {
                            auto proj = glm::perspectiveFov(glm::radians(60.f),
                                                            static_cast<float>(rndrSz.x),
                                                            static_cast<float>(rndrSz.y), 1.f, 3.f);
                            rayTracer.SetProjection(proj);

                            std::cout << std::format("Render size is {}x{}.\n", rndrSz.x, rndrSz.y);
                        },
                        /*camera changed callback*/
                        [&](const kouek::FPSCamera &cam) {
                            const auto &[R, F, U, P] = cam.GetRFUP();
                            glm::mat3 rot(R, U, -F);
                            rayTracer.SetCameraToWorld(P, rot);
                        });
    {
        auto dir = glm::normalize(scnCfg.cntr - scnCfg.eye);
        app.camera.LookAt(scnCfg.eye, scnCfg.cntr, scnCfg.up);
        app.revolveCntr =
            scnCfg.eye +
            glm::dot(.5f * (deviceDat.rootAABB.minPos + deviceDat.rootAABB.maxPos) - scnCfg.eye,
                     dir) *
                dir;
        auto extent = deviceDat.rootAABB.maxPos - deviceDat.rootAABB.minPos;
        app.movSens = .05f * std::min({extent.x, extent.y, extent.z});
    }
    app.callerOnCameraChanged(app.camera);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    app.Run([&]() {
        if (showFPS)
            cudaEventRecord(start);

        auto surfRndrTo = app.MapGLResourceToCUDA();

        rayTracer.Render(surfRndrTo, app.rndrSz,
                         static_cast<kouek::RayTracer::RayTracer::RenderTarget>(rndrTarget));
        maxSPP = rayTracer.GetMaxSamplePerPixel();
        currentSPP = rayTracer.GetCurrentAccumulatedSamplePerPixel();

        app.UnmapGLResourceFromCUDA(surfRndrTo);

        if (showFPS) {
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

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
