#version 330 core

in vec2 TexCoord;
in float Growth;
out vec4 FragColor;

uniform sampler2D flowerTexture;

void main() {
    vec4 texColor = texture(flowerTexture, TexCoord);
    
    if (texColor.a < 0.1)
        discard;
        
    float alpha = texColor.a * Growth;
    
    if (alpha < 0.3)
        discard;
    
    FragColor = vec4(texColor.rgb * (0.8 + 0.2 * Growth), alpha);
}