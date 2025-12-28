#version 330 core
in vec2 TexCoord;
out vec4 FragColor;

uniform sampler2D groundTexture;

void main() {
    vec4 texColor = texture(groundTexture, TexCoord);
    FragColor = texColor * vec4(0.8, 1.0, 0.8, 1.0); // Slightly greener tint
}