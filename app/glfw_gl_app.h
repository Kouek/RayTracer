#ifndef GLFW_GL_APP_H
#define GLFW_GL_APP_H

#include <memory>
#include <source_location>

#include <glad/glad.h>

#include <glfw/glfw3.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <cg/FPS_camera.h>

#include <cuda/helper.h>
#include <cuda_gl_interop.h>

inline void CheckGL(std::source_location srcLoc = std::source_location::current()) {
    GLenum glErr;
    if ((glErr = glGetError()) != GL_NO_ERROR) {
        std::cerr << std::format("[GL Error] at {}:{}. 0x{:x}\n", srcLoc.file_name(), srcLoc.line(),
                                 glErr);
    }
}

struct GLFWxGLxCUDAApp {
    bool ok = false;
    bool banInput = false;
    glm::ivec2 rndrSz;
    float movSens = 1.f;
    glm::vec3 revolveCntr = glm::zero<glm::vec3>();

    GLuint offScrnFBO = 0;
    GLuint colAtch = 0;
    GLuint depAtch = 0;

    cudaResourceDesc cudaGLResDesc;
    cudaGraphicsResource_t cudaGLRes = nullptr;

    GLFWwindow *window;

    kouek::FPSCamera camera;

    std::function<void(const glm::ivec2 &)> callerOnResized;
    std::function<void(const kouek::FPSCamera &)> callerOnCameraChanged;
    std::function<void(const glm::ivec2 &, int, int, int)> callerOnMouseActivated;

    GLFWxGLxCUDAApp(
        const glm::ivec2 &wndSz, const char *wndTitle,
        std::function<void(const glm::ivec2 &)> callerOnResized,
        std::function<void(const kouek::FPSCamera &)> callerOnCameraChanged,
        std::function<void(const glm::ivec2 &, int, int, int)> callerOnMouseActivated = nullptr) {
        this->callerOnResized = callerOnResized;
        this->callerOnCameraChanged = callerOnCameraChanged;

        // Init GLFW context
        glfwInit();
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

        window = glfwCreateWindow(wndSz.x, wndSz.y, wndTitle, nullptr, nullptr);
        if (window == NULL) {
            std::cerr << "Failed to create GLFW window.\n";
            glfwTerminate();
            return;
        }

        glfwMakeContextCurrent(window);
        glfwSetWindowUserPointer(window, this);

        auto onKeyActivated = [](GLFWwindow *window, int key, int scancode, int action, int mods) {
            static constexpr auto RotSens = 1.f;

            auto &app = *reinterpret_cast<GLFWxGLxCUDAApp *>(glfwGetWindowUserPointer(window));
            auto &camera = app.camera;

            switch (key) {
            case GLFW_KEY_ESCAPE:
                if (action == GLFW_RELEASE)
                    glfwSetWindowShouldClose(window, GLFW_TRUE);
                break;
            case GLFW_KEY_UP:
                camera.Revolve(glm::distance(camera.GetPos(), app.revolveCntr), 0.f, +RotSens);
                break;
            case GLFW_KEY_DOWN:
                camera.Revolve(glm::distance(camera.GetPos(), app.revolveCntr), 0.f, -RotSens);
                break;
            case GLFW_KEY_LEFT:
                camera.Revolve(glm::distance(camera.GetPos(), app.revolveCntr), +RotSens, 0.f);
                break;
            case GLFW_KEY_RIGHT:
                camera.Revolve(glm::distance(camera.GetPos(), app.revolveCntr), -RotSens, 0.f);
                break;
            case GLFW_KEY_Q:
                camera.Move(0.f, 0.f, -app.movSens);
                break;
            case GLFW_KEY_E:
                camera.Move(0.f, 0.f, +app.movSens);
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
                app.onCameraChanged();
                break;
            default:
                break;
            }
        };
        auto onMouseBtnActivated = [](GLFWwindow *window, int btn, int action, int mod) {
            auto &app = *reinterpret_cast<GLFWxGLxCUDAApp *>(glfwGetWindowUserPointer(window));

            glm::dvec2 pos;
            glfwGetCursorPos(window, &pos.x, &pos.y);

            glm::ivec2 posOnFrmBuf = {
                glm::clamp(pos.x, 0., static_cast<double>(app.rndrSz.x - 1)),
                glm::clamp(app.rndrSz.y - 1 - pos.y, 0., static_cast<double>(app.rndrSz.y - 1))};
            if (app.callerOnMouseActivated)
                app.callerOnMouseActivated(posOnFrmBuf, btn, action, mod);
        };
        auto onFrameBufferSizeChanged = [](GLFWwindow *window, int width, int height) {
            if (width == 0 || height == 0)
                return;

            auto &app = *reinterpret_cast<GLFWxGLxCUDAApp *>(glfwGetWindowUserPointer(window));

            glfwGetFramebufferSize(window, &app.rndrSz.x, &app.rndrSz.y);
            app.onResized();
        };
        glfwSetKeyCallback(window, onKeyActivated);
        glfwSetFramebufferSizeCallback(window, onFrameBufferSizeChanged);
        glfwSetMouseButtonCallback(window, onMouseBtnActivated);

        // Load GL functions
        if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
            std::cerr << "Failed to initialize GLAD.\n";
            return;
        }

        // Init ImGui
        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImGui::StyleColorsDark();

        ImGui_ImplGlfw_InitForOpenGL(window, true);
        ImGui_ImplOpenGL3_Init("#version 150");

        // Create offscreen GL framebuffer object and CUDA-binded resource
        glfwGetFramebufferSize(window, &rndrSz.x, &rndrSz.y);

        // Initial callback
        onResized();
        onCameraChanged();
    }
    ~GLFWxGLxCUDAApp() {
        if (cudaGLRes != nullptr) {
            using namespace kouek::CUDA;
            KOUEK_CUDA_CHECK(cudaGraphicsUnregisterResource(cudaGLRes));
            cudaGLRes = nullptr;
        }
        if (offScrnFBO != 0) {
            glDeleteFramebuffers(1, &offScrnFBO);
            offScrnFBO = 0;
        }
        if (colAtch != 0) {
            glDeleteTextures(1, &colAtch);
            colAtch = 0;
        }
        if (depAtch != 0) {
            glDeleteRenderbuffers(1, &depAtch);
            depAtch = 0;
        }

        ImGui_ImplOpenGL3_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();

        glfwTerminate();
    }

    void Run(std::function<void()> func) {
        while (glfwWindowShouldClose(window) == 0) {
            if (!banInput)
                glfwPollEvents();
            func();
            glfwSwapBuffers(window);
        }
    }

    cudaSurfaceObject_t MapGLResourceToCUDA() {
        using namespace kouek::CUDA;

        cudaSurfaceObject_t cudaSurf;
        KOUEK_CUDA_CHECK(cudaGraphicsMapResources(1, &cudaGLRes));
        KOUEK_CUDA_CHECK(
            cudaGraphicsSubResourceGetMappedArray(&cudaGLResDesc.res.array.array, cudaGLRes, 0, 0));
        KOUEK_CUDA_CHECK(cudaCreateSurfaceObject(&cudaSurf, &cudaGLResDesc));
        return cudaSurf;
    }

    void UnmapGLResourceFromCUDA(cudaSurfaceObject_t cudaSurf) {
        using namespace kouek::CUDA;

        KOUEK_CUDA_CHECK(cudaDestroySurfaceObject(cudaSurf));
        KOUEK_CUDA_CHECK(cudaGraphicsUnmapResources(1, &cudaGLRes));
    }

  private:
    void onResized() {
        using namespace kouek::CUDA;
        if (cudaGLRes) {
            // Unbind if is binded
            KOUEK_CUDA_CHECK(cudaGraphicsUnregisterResource(cudaGLRes));
            cudaGLRes = nullptr;
        }

        // Create offscreen GL framebuffer object
        if (offScrnFBO == 0)
            glCreateFramebuffers(1, &offScrnFBO);
        glBindFramebuffer(GL_FRAMEBUFFER, offScrnFBO);

        if (colAtch == 0)
            glGenTextures(1, &colAtch);
        glBindTexture(GL_TEXTURE_2D, colAtch);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, rndrSz.x, rndrSz.y, 0, GL_RGBA, GL_UNSIGNED_BYTE,
                     (const void *)0);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, colAtch, 0);

        if (depAtch == 0)
            glGenRenderbuffers(1, &depAtch);
        glBindRenderbuffer(GL_RENDERBUFFER, depAtch);
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT16, rndrSz.x, rndrSz.y);
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depAtch);

        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glBindTexture(GL_TEXTURE_2D, 0);
        glBindRenderbuffer(GL_RENDERBUFFER, 0);

        ok = GL_FRAMEBUFFER_COMPLETE == glCheckNamedFramebufferStatus(offScrnFBO, GL_FRAMEBUFFER);
        if (!ok) {
            std::cerr << "FBO is not complete.\n";
            return;
        }

        // Bind GL framebuffer object to CUDA surface object
        memset(&cudaGLResDesc, 0, sizeof(cudaGLResDesc));
        cudaGLResDesc.resType = cudaResourceTypeArray;
        ok &= cudaSuccess ==
              KOUEK_CUDA_CHECK(cudaGraphicsGLRegisterImage(&cudaGLRes, colAtch, GL_TEXTURE_2D,
                                                           cudaGraphicsRegisterFlagsWriteDiscard));

        callerOnResized(rndrSz);
    }

    void onCameraChanged() { callerOnCameraChanged(camera); }
};

#endif // !GLFW_GL_APP_H
