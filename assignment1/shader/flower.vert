#version 330 core

layout(location = 0) in vec3 aPos;
layout(location = 1) in vec2 aTexCoord;
layout(location = 2) in vec3 aInstancePos;
layout(location = 3) in float aGrowth;

out vec2 TexCoord;
out float Growth;

uniform mat4 VP;
uniform vec3 cameraPos;

void main() {
    Growth = aGrowth;
    
    vec3 pos = aInstancePos;
    
    vec3 look = normalize(cameraPos - aInstancePos);
    vec3 up = vec3(0.0, 1.0, 0.0);
    
    vec3 right = normalize(cross(up, look));
    
    vec3 billboardUp = normalize(cross(look, right));
    
    pos += right * aPos.x * aGrowth * 5.0;
    pos += billboardUp * aPos.y * aGrowth * 5.0;
    
    gl_Position = VP * vec4(pos, 1.0);
    TexCoord = aTexCoord;
}