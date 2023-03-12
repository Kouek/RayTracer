#include <iostream>

#include <array>
#include <memory>
#include <vector>

#include <glad/glad.h>

#include <glfw/glfw3.h>

#include <cg/FPS_camera.hpp>
#include <cg/mesh.hpp>
#include <cg/shader.hpp>
#include <cmake_in.h>

#include <cmdparser.hpp>
#include <tinyxml/tinyxml2.h>

#include <ray_tracer.h>

using namespace kouek;

static glm::uvec2 rndrSz{1024, 1024};

static GLuint rndrTex;

static RayTracer renderer;
static std::unique_ptr<RayTraceScn> scn;

static FPSCamera cam(glm::vec3{1.5f}, glm::vec3{0.f});
static auto cntrPos = glm::zero<glm::vec3>();
static auto proj = glm::perspectiveFov(glm::radians(60.f), (float)rndrSz.x,
                                       (float)rndrSz.y, .1f, 10.f);

inline void onCamModified() {
    auto [R, F, U, P] = cam.GetRFUP();
    renderer.SetCamera({glm::vec4{R, 0}, glm::vec4{U, 0}, glm::vec4{-F, 0},
                        glm::vec4{P, 1.f}});
}

inline void onKeyActivated(GLFWwindow *window, int key, int scancode,
                           int action, int mods) {
    static constexpr auto ROT_SENS = glm::radians(30.f);
    static constexpr auto MOV_SENS = 2.f;

    switch (key) {
    case GLFW_KEY_ESCAPE:
        if (action == GLFW_RELEASE)
            glfwSetWindowShouldClose(window, GLFW_TRUE);
        break;
    case GLFW_KEY_UP:
        if (action != GLFW_RELEASE)
            cam.Revolve(glm::distance(cam.GetPos(), cntrPos), 0.f, +ROT_SENS);
        break;
    case GLFW_KEY_DOWN:
        if (action != GLFW_RELEASE)
            cam.Revolve(glm::distance(cam.GetPos(), cntrPos), 0.f, -ROT_SENS);
        break;
    case GLFW_KEY_LEFT:
        if (action != GLFW_RELEASE)
            cam.Revolve(glm::distance(cam.GetPos(), cntrPos), +ROT_SENS, 0.f);
        break;
    case GLFW_KEY_RIGHT:
        if (action != GLFW_RELEASE)
            cam.Revolve(glm::distance(cam.GetPos(), cntrPos), -ROT_SENS, 0.f);
        break;
    case GLFW_KEY_Q:
        if (action != GLFW_RELEASE)
            cam.Move(0.f, 0.f, -MOV_SENS);
        break;
    case GLFW_KEY_E:
        if (action != GLFW_RELEASE)
            cam.Move(0.f, 0.f, +MOV_SENS);
        break;
    default:
        break;
    }

    switch (key) {
    case GLFW_KEY_UP:
    case GLFW_KEY_DOWN:
    case GLFW_KEY_LEFT:
    case GLFW_KEY_RIGHT:
    case GLFW_KEY_Q:
    case GLFW_KEY_E:
        onCamModified();
        break;
    default:
        break;
    }
}

int main(int argc, char **argv) {
    // Command cmdParser
    cli::Parser cmdParser(argc, argv);
    cmdParser.set_required<std::string>("s", "scene", "Scene Folder Path");
    cmdParser.run_and_exit_if_error();

    // GLFW context
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow *window =
        glfwCreateWindow(rndrSz.x, rndrSz.y, "Ray Tracer", nullptr, nullptr);
    if (window == NULL) {
        std::cout << "[ERROR] Failed to create GLFW window" << std::endl;
        goto TERMINAL;
    }
    glfwMakeContextCurrent(window);
    glfwSetKeyCallback(window, onKeyActivated);

    // Load OpenGL functions
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cout << "[ERROR] Failed to initialize GLAD" << std::endl;
        goto TERMINAL;
    }

    // Create texture to be rendered in GL
    glGenTextures(1, &rndrTex);
    glBindTexture(GL_TEXTURE_2D, rndrTex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, rndrSz.x, rndrSz.y, 0, GL_RGBA,
                 GL_UNSIGNED_BYTE, (const void *)0);
    glBindTexture(GL_TEXTURE_2D, 0);

    // GL drawing resources
    GLuint VBO = 0, VAO = 0, EBO = 0;
    glGenVertexArrays(1, &VAO);
    glBindVertexArray(VAO);
    glGenBuffers(1, &VBO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    {
        std::vector<GLfloat> verts = {-1.f, -1.f, 0, 0,   1.f, -1.f, 1.f, 0,
                                      -1.f, 1.f,  0, 1.f, 1.f, 1.f,  1.f, 1.f};
        glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * verts.size(),
                     verts.data(), GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(GLfloat) * 4,
                              (const void *)0);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(GLfloat) * 4,
                              (const void *)(sizeof(GLfloat) * 2));
    }
    glGenBuffers(1, &EBO);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    {
        GLushort idxes[] = {0, 1, 3, 0, 3, 2};
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(GLushort) * 6, idxes,
                     GL_STATIC_DRAW);
    }
    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

    Shader screenQuadShader(
        (std::string(PROJECT_SOURCE_DIR) + "/screen_quad.vs").c_str(),
        (std::string(PROJECT_SOURCE_DIR) + "/screen_quad.fs").c_str());
    GLint sqShdrLowResScalePos =
        glGetUniformLocation(screenQuadShader.ID, "lowResScale");
    assert(sqShdrLowResScalePos != -1);

    // Init Renderer
    renderer.Init();
    renderer.SetOutput(rndrTex, rndrSz, 2);
    renderer.SetMaxDepth(10);
    renderer.SetProjection(proj);
    onCamModified();

    // Load Scene
    {
        auto scnPath = cmdParser.get<std::string>("s");
        auto scnName = [&]() {
            auto itr = scnPath.rbegin();
            while (itr != scnPath.rend() && *itr != '/' && *itr != '\\')
                ++itr;
            auto dist2end = std::distance(scnPath.rbegin(), itr);
            return scnPath.substr(scnPath.size() - dist2end);
        }();
        scnPath.append("/");

        auto mesh = std::make_shared<Mesh>();
        try {
            mesh->LoadFromFile(scnPath + scnName + ".obj",
                               scnPath + scnName + ".mtl");

            std::cout << "Load scene: " << scnName << std::endl;
            std::cout << ">> vertices num: " << mesh->GetVS().size()
                      << std::endl;
            std::cout << ">> faces num: " << mesh->GetFS().size() << std::endl;
            auto &gs = mesh->GetGS();
            std::cout << ">> groups num: " << gs.size() - 1 << std::endl;
            for (glm::uint gi = 0; gi < gs.size() - 1; ++gi) {
                std::cout << ">> group " << gi << ", covers faces [" << gs[gi]
                          << ", " << gs[gi + 1] << ")" << std::endl;
                std::cout << ">>>> material name: "
                          << mesh->GetG2MtlNames().at(gi) << std::endl;
            }
        } catch (std::exception &e) {
            std::cout << "[ERROR] Failed to load scene models: " << e.what()
                      << std::endl;
            goto TERMINAL;
        }

        try {
            tinyxml2::XMLDocument doc;
            doc.LoadFile((scnPath + scnName + ".xml").c_str());
            if (doc.Error())
                throw std::runtime_error(doc.ErrorIDToName(doc.ErrorID()));
            {
                auto par = doc.FirstChildElement("camera");
                auto eye = par->FirstChildElement("eye");
                auto cntr = par->FirstChildElement("lookat");
                auto up = par->FirstChildElement("up");
                cam.LookAt({eye->FloatAttribute("x"), eye->FloatAttribute("y"),
                            eye->FloatAttribute("z")},
                           {cntr->FloatAttribute("x"),
                            cntr->FloatAttribute("y"),
                            cntr->FloatAttribute("z")},
                           {up->FloatAttribute("x"), up->FloatAttribute("y"),
                            up->FloatAttribute("z")});

                onCamModified();
            }
            if (!mesh->GetG2MtlNames().empty()) {
                auto light = doc.FirstChildElement("light");
                auto getRadiance = [&](decltype(light) &light) -> glm::vec3 {
                    auto ret = glm::zero<glm::vec3>();
                    const char *buf;
                    auto err = light->QueryStringAttribute("radiance", &buf);
                    if (err != tinyxml2::XML_SUCCESS)
                        throw std::runtime_error("No radiance");
                    sscanf(buf, "%f,%f,%f", &ret.x, &ret.y, &ret.z);
                    return ret;
                };

                std::vector<std::tuple<std::string, glm::vec3>>
                    lights; // stores names and radiances
                std::string name;
                while (light != nullptr) {
                    const char *buf;
                    auto err = light->QueryStringAttribute("mtlname", &buf);
                    if (err != tinyxml2::XML_SUCCESS)
                        throw std::runtime_error("No mtlname");
                    name = buf;

                    lights.emplace_back(
                        std::make_tuple(name, getRadiance(light)));

                    light = light->NextSiblingElement("light");
                }

                mesh->SetLights(lights);
            }
        } catch (std::exception &e) {
            std::cout << "[WARNING] Failed to read scene configuration: "
                      << e.what() << std::endl;
        }

        auto scn = std::make_shared<RayTraceScn>();
        scn->SetBackgroundColor({.1f, .2f, .25f});
        scn->SetModel(mesh);
        scn->BuildBVH();
        renderer.SetScene(scn);
    }

    // Render Loop
    screenQuadShader.use();
    glClearColor(0, 0, 0, 1.f);
    glBindTexture(GL_TEXTURE_2D, rndrTex);
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        renderer.Prepare();
        auto lowResScale = renderer.Render();
        glUniform1f(sqShdrLowResScalePos, lowResScale);

        glClear(GL_COLOR_BUFFER_BIT);
        glBindVertexArray(VAO);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, (const void *)0);
        glBindVertexArray(0);

        glfwSwapBuffers(window);
    }

TERMINAL:
    renderer.Release();

    if (VBO != 0)
        glDeleteBuffers(1, &VBO);
    if (EBO != 0)
        glDeleteBuffers(1, &EBO);
    if (VAO != 0)
        glDeleteVertexArrays(1, &VAO);
    if (rndrTex != 0)
        glDeleteTextures(1, &rndrTex);
    glfwTerminate();

    return 0;
}
