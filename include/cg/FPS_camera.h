#ifndef KOUEK_FPS_CAMERA_H
#define KOUEK_FPS_CAMERA_H

#include <tuple>

#include <glm/gtc/matrix_transform.hpp>

namespace kouek {

class FPSCamera {
  private:
    float yawRad, pitchRad;
    glm::vec3 pos;
    glm::vec3 right;
    glm::vec3 worldUp, up;
    glm::vec3 forward;
    glm::mat4x4 view;

  public:
    FPSCamera() : FPSCamera(glm::vec3{1.f, 1.f, 1.f}, glm::vec3{0, 0, 0}) {}
    FPSCamera(const glm::vec3 &eyePos, const glm::vec3 &eyeCenter,
              const glm::vec3 &up = glm::vec3{0, 1.f, 0}) {
        LookAt(eyePos, eyeCenter, up);
    }
    void LookAt(const glm::vec3 &eyePos, const glm::vec3 &eyeCenter,
                const glm::vec3 &up = glm::vec3{0, 1.f, 0}) {
        this->pos = eyePos;
        this->forward = glm::normalize(eyeCenter - eyePos);
        this->worldUp = up;
        updateWithPosForwardUp();
    }
    const glm::mat4 &GetViewMat() const { return view; }
    const glm::vec3 &GetPos() const { return pos; }
    std::tuple<const glm::vec3 &, const glm::vec3 &, const glm::vec3 &, const glm::vec3 &>
    GetRFUP() const {
        return {right, forward, up, pos};
    }
    void SetPos(const glm::vec3 &pos) {
        this->pos = pos;

        view[3][0] = -glm::dot(right, pos);
        view[3][1] = -glm::dot(up, pos);
        view[3][2] = glm::dot(forward, pos);
    }
    void Move(float rightStep, float upStep, float forwardStep) {
        pos += forwardStep * forward + rightStep * right + upStep * up;

        view[3][0] = -glm::dot(right, pos);
        view[3][1] = -glm::dot(up, pos);
        view[3][2] = glm::dot(forward, pos);
    }
    void Rotate(float yawDifDeg, float pitchDifDeg) {
        float pitchDifRad = glm::radians(pitchDifDeg);
        float yawDifRad = glm::radians(yawDifDeg);
        constexpr float PITCH_LIMIT = glm::radians(60.f);
        pitchRad += pitchDifRad;
        // avoid dead lock
        if (pitchRad > PITCH_LIMIT)
            pitchRad = PITCH_LIMIT;
        else if (pitchRad < -PITCH_LIMIT)
            pitchRad = -PITCH_LIMIT;
        yawRad += yawDifRad;
        // avoid reaching float limit
        if (yawRad > glm::pi<float>())
            yawRad -= 2 * glm::pi<float>();
        else if (yawRad < -glm::pi<float>())
            yawRad += 2 * glm::pi<float>();
        updateFromYawPitch();
    }
    void Revolve(float fwdDist, float yawDifDeg, float pitchDifDeg) {
        float pitchDifRad = glm::radians(-pitchDifDeg);
        float yawDifRad = glm::radians(-yawDifDeg);
        constexpr float PITCH_LIMIT = glm::radians(60.f);
        pitchRad += pitchDifRad;
        // avoid dead lock
        if (pitchRad > PITCH_LIMIT)
            pitchRad = PITCH_LIMIT;
        else if (pitchRad < -PITCH_LIMIT)
            pitchRad = -PITCH_LIMIT;
        yawRad += yawDifRad;
        // avoid reaching float limit
        if (yawRad > glm::pi<float>())
            yawRad -= 2 * glm::pi<float>();
        else if (yawRad < -glm::pi<float>())
            yawRad += 2 * glm::pi<float>();

        auto center = pos + fwdDist * forward;

        {
            float cy = cosf(yawRad), sy = sinf(yawRad);
            float cp = cosf(pitchRad), sp = sinf(pitchRad);
            forward.x = cp * cy, forward.y = sp, forward.z = -cp * sy;
            forward = glm::normalize(forward);
        }
        right = glm::normalize(glm::cross(forward, worldUp));
        up = glm::normalize(glm::cross(right, forward));

        pos = center - fwdDist * forward;

        view[0][0] = right.x;
        view[1][0] = right.y;
        view[2][0] = right.z;
        view[0][1] = up.x;
        view[1][1] = up.y;
        view[2][1] = up.z;
        view[0][2] = -forward.x;
        view[1][2] = -forward.y;
        view[2][2] = -forward.z;
        view[3][0] = -glm::dot(right, pos);
        view[3][1] = -glm::dot(up, pos);
        view[3][2] = glm::dot(forward, pos);

        view[3][3] = 1.f;
        view[0][3] = view[1][3] = view[2][3] = 0;
    }

  private:
    void updateWithPosForwardUp() {
        right = glm::normalize(glm::cross(forward, worldUp));
        up = glm::normalize(glm::cross(right, forward));

        pitchRad = asinf(forward.y);
        yawRad = fabsf(forward.x) > fabsf(forward.z) ? acosf(forward.x / cosf(pitchRad))
                                                     : asinf(-forward.z / cosf(pitchRad));

        view[0][0] = right.x;
        view[1][0] = right.y;
        view[2][0] = right.z;
        view[0][1] = up.x;
        view[1][1] = up.y;
        view[2][1] = up.z;
        view[0][2] = -forward.x;
        view[1][2] = -forward.y;
        view[2][2] = -forward.z;
        view[3][0] = -glm::dot(right, pos);
        view[3][1] = -glm::dot(up, pos);
        view[3][2] = glm::dot(forward, pos);

        view[3][3] = 1.f;
        view[0][3] = view[1][3] = view[2][3] = 0;
    }
    void updateFromYawPitch() {
        {
            float cy = cosf(yawRad), sy = sinf(yawRad);
            float cp = cosf(pitchRad), sp = sinf(pitchRad);
            forward.x = cp * cy, forward.y = sp, forward.z = -cp * sy;
            forward = glm::normalize(forward);
        }
        right = glm::normalize(glm::cross(forward, worldUp));
        up = glm::normalize(glm::cross(right, forward));

        view[0][0] = right.x;
        view[1][0] = right.y;
        view[2][0] = right.z;
        view[0][1] = up.x;
        view[1][1] = up.y;
        view[2][1] = up.z;
        view[0][2] = -forward.x;
        view[1][2] = -forward.y;
        view[2][2] = -forward.z;
        view[3][0] = -glm::dot(right, pos);
        view[3][1] = -glm::dot(up, pos);
        view[3][2] = glm::dot(forward, pos);

        view[3][3] = 1.f;
        view[0][3] = view[1][3] = view[2][3] = 0;
    }
};

} // namespace kouek

#endif // !KOUEK_FPS_CAMERA_H
