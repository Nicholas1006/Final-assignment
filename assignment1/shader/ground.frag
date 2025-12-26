#version 330 core

in vec3 FragPos;
in vec3 Normal;
in vec2 TexCoord;

out vec4 FragColor;

uniform sampler2D groundTexture;
uniform vec3 viewPos;
uniform vec3 lightPosition;
uniform vec3 lightIntensity;

void main() {
    // Texture color
    vec3 texColor = texture(groundTexture, TexCoord).rgb;
    
    // Simple lighting
    vec3 norm = normalize(Normal);
    vec3 lightDir = normalize(lightPosition - FragPos);
    float diff = max(dot(norm, lightDir), 0.0);
    
    // Ambient + diffuse
    vec3 ambient = vec3(0.2, 0.2, 0.2);
    vec3 diffuse = diff * lightIntensity * 0.000001; // Scale down intensity
    
    vec3 result = (ambient + diffuse) * texColor;
    
    // Simple fog based on distance
    float distance = length(FragPos - viewPos);
    float fogFactor = exp(-distance * 0.001);
    fogFactor = clamp(fogFactor, 0.0, 1.0);
    
    vec3 fogColor = vec3(0.5, 0.6, 0.7);
    result = mix(fogColor, result, fogFactor);
    
    FragColor = vec4(result, 1.0);
}