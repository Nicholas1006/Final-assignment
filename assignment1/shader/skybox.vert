#version 330 core

layout(location = 0) in vec3 aPos;

out vec3 TexCoord;

uniform mat4 MVP;

void main() {
    TexCoord = aPos;
    gl_Position = MVP * vec4(aPos, 1.0);
    gl_Position = gl_Position.xyww; // Set z = w for max depth
}