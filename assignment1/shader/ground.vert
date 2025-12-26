#version 330 core

layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec2 aTexCoord;

out vec3 FragPos;
out vec3 Normal;
out vec2 TexCoord;

uniform mat4 MVP;
uniform mat4 model;

void main() {
    FragPos = vec3(model * vec4(aPos, 1.0));
    Normal = aNormal; // Simple normal (no transformation for ground)
    TexCoord = aTexCoord;
    
    gl_Position = MVP * vec4(aPos, 1.0);
}