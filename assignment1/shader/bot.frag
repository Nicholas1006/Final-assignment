#version 330 core

// Input from vertex shader
in vec3 worldPosition;
in vec3 worldNormal;
in vec2 uv;

// Output color
out vec4 color;

// Lighting uniforms
uniform vec3 lightPosition;
uniform vec3 lightIntensity;

void main() {
    // Simple Lambertian shading
    vec3 N = normalize(worldNormal);
    vec3 L = normalize(lightPosition - worldPosition);
    
    // Diffuse lighting
    float diffuse = max(dot(N, L), 0.0);
    
    // Calculate color
    vec3 materialColor = vec3(0.8, 0.8, 0.8);
    vec3 finalColor = materialColor * lightIntensity * diffuse;
    
    // Gamma correction
    finalColor = pow(finalColor, vec3(1.0/2.2));
    
    color = vec4(finalColor, 1.0);
}