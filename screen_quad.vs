#version 450 core

layout(location = 0) in vec2 positionIn;
layout(location = 1) in vec2 uvIn;
out vec2 uv;

void main() {
    gl_Position = vec4(positionIn, 0, 1.0);
    uv = uvIn;
}
