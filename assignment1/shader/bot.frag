#version 330 core

in vec3 worldPosition;
in vec3 worldNormal;

out vec3 finalColor;

uniform vec3 lightPosition;
uniform vec3 lightIntensity;

void main() {
    // DEBUG: Solid color for testing
    // finalColor = vec3(1.0, 0.0, 0.0); // Red
    // return;
    
    // Simple lighting
    vec3 lightDir = normalize(lightPosition - worldPosition);
    float diff = max(dot(normalize(worldNormal), lightDir), 0.0);
    
    // Base color (robot material)
    vec3 baseColor = vec3(0.8, 0.3, 0.3); // Reddish
    
    // Ambient + diffuse
    vec3 ambient = vec3(0.2, 0.2, 0.2);
    vec3 diffuse = diff * lightIntensity * 0.000001 * baseColor;
    
    vec3 result = ambient + diffuse;
    
    // Gamma correction
    result = pow(result, vec3(1.0/2.2));
    
    finalColor = result;
}