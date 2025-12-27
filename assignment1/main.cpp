#include <glad/gl.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <vector>
#include <iostream>
#include <iomanip>
#include <string>
#include <sstream>
#include <render/shader.h>

#define STB_IMAGE_IMPLEMENTATION
#include <stb/stb_image.h>
// ====================
// RENDER CONTROLS
// ====================
static bool renderRobot = true;
static bool renderBuildings = true;
static bool renderSky = false;
static bool renderGround = true;

// ====================
// GLOBAL VARIABLES
// ====================
static GLFWwindow *window;
static int windowWidth = 1024;
static int windowHeight = 768;
static bool firstMouse = true;
static float lastX = windowWidth / 2.0f;
static float lastY = windowHeight / 2.0f;

// Camera
static glm::vec3 cameraPos = glm::vec3(0.0f, 100.0f, 500.0f);  // Increased height and distance
static glm::vec3 cameraFront = glm::vec3(0.0f, 0.0f, -1.0f);
static glm::vec3 cameraUp = glm::vec3(0.0f, 1.0f, 0.0f);
static float cameraSpeed = 100.0f;
static float yaw = -90.0f;
static float pitch = -10.0f;  // Looking slightly down
static float FoV = 45.0f;
static float zNear = 0.1f;
static float zFar = 5000.0f;

// Lighting
static glm::vec3 lightIntensity(5e6f, 5e6f, 5e6f);
static glm::vec3 lightPosition(0.0f, 500.0f, 0.0f);

// Animation
static bool playAnimation = true;
static float playbackSpeed = 1.0f;

// ====================
// FUNCTION DECLARATIONS
// ====================
static void mouse_callback(GLFWwindow* window, double xpos, double ypos);
static void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
static void key_callback(GLFWwindow *window, int key, int scancode, int action, int mode);
void processInput(GLFWwindow* window, float deltaTime);


void checkGLError(const char* context) {
    GLenum error = glGetError();
    while (error != GL_NO_ERROR) {
        std::cout << "OpenGL Error at " << context << ": ";
        switch(error) {
            case GL_INVALID_ENUM: std::cout << "GL_INVALID_ENUM"; break;
            case GL_INVALID_VALUE: std::cout << "GL_INVALID_VALUE"; break;
            case GL_INVALID_OPERATION: std::cout << "GL_INVALID_OPERATION"; break;
            case GL_OUT_OF_MEMORY: std::cout << "GL_OUT_OF_MEMORY"; break;
            default: std::cout << "Unknown error " << error; break;
        }
        std::cout << std::endl;
        error = glGetError();
    }
}

int pow(int base, int exp) { 
	if(exp==0) {
		return 1;
	}
	else{
		return base * pow(base, exp - 1);
	}
}

int abs(int x) {
	if(x<0) {
		return -x;
	}
	else {
		return x;
	}
}

// ====================
// SIMPLE SHADER CREATION
// ====================
GLuint createSimpleShader() {
    const char* vertexShaderSource = R"(
        #version 330 core
        layout(location = 0) in vec3 aPos;
        uniform mat4 MVP;
        void main() {
            gl_Position = MVP * vec4(aPos, 1.0);
        }
    )";

    const char* fragmentShaderSource = R"(
        #version 330 core
        out vec3 FragColor;
        void main() {
            FragColor = vec3(1.0, 0.5, 0.2); // Orange color
        }
    )";

    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
    glCompileShader(vertexShader);
    
    GLint success;
    char infoLog[512];
    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
        std::cout << "Simple Vertex shader compilation failed: " << infoLog << std::endl;
    }

    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
    glCompileShader(fragmentShader);
    
    glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(fragmentShader, 512, NULL, infoLog);
        std::cout << "Simple Fragment shader compilation failed: " << infoLog << std::endl;
    }

    GLuint program = glCreateProgram();
    glAttachShader(program, vertexShader);
    glAttachShader(program, fragmentShader);
    glLinkProgram(program);
    
    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(program, 512, NULL, infoLog);
        std::cout << "Simple Shader program linking failed: " << infoLog << std::endl;
    }
    
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
    
    return program;
}

// ====================
// TEST CUBE
// ====================
struct TestCube {
    GLuint vao, vbo, ebo;
    GLuint shader;
    int indexCount;
    
    void initialize() {
        std::cout << "Initializing test cube..." << std::endl;
        
        // Cube vertices
        float vertices[] = {
            // positions
            -50.0f, -50.0f, -50.0f,
             50.0f, -50.0f, -50.0f,
             50.0f,  50.0f, -50.0f,
            -50.0f,  50.0f, -50.0f,
            -50.0f, -50.0f,  50.0f,
             50.0f, -50.0f,  50.0f,
             50.0f,  50.0f,  50.0f,
            -50.0f,  50.0f,  50.0f
        };
        
        // Cube indices
        unsigned int indices[] = {
            0, 1, 2, 2, 3, 0,
            4, 5, 6, 6, 7, 4,
            0, 4, 7, 7, 3, 0,
            1, 5, 6, 6, 2, 1,
            3, 2, 6, 6, 7, 3,
            0, 1, 5, 5, 4, 0
        };
        
        indexCount = 36;
        
        glGenVertexArrays(1, &vao);
        glGenBuffers(1, &vbo);
        glGenBuffers(1, &ebo);
        
        glBindVertexArray(vao);
        
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
        
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
        
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(0);
        
        glBindVertexArray(0);
        
        // Create simple shader
        shader = createSimpleShader();
        std::cout << "Test cube shader ID: " << shader << std::endl;
    }
    
    void render(const glm::mat4& viewProj) {
        if (!renderRobot) return;
        glUseProgram(shader);
        
        glm::mat4 model = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 100.0f, 0.0f));
        glm::mat4 mvp = viewProj * model;
        
        GLuint mvpLoc = glGetUniformLocation(shader, "MVP");
        if (mvpLoc != -1) {
            glUniformMatrix4fv(mvpLoc, 1, GL_FALSE, glm::value_ptr(mvp));
        }
        
        glBindVertexArray(vao);
        glDrawElements(GL_TRIANGLES, indexCount, GL_UNSIGNED_INT, 0);
        glBindVertexArray(0);
    }
    
    void cleanup() {
        glDeleteVertexArrays(1, &vao);
        glDeleteBuffers(1, &vbo);
        glDeleteBuffers(1, &ebo);
        glDeleteProgram(shader);
    }
};

// ====================
// SIMPLE GROUND
// ====================
struct SimpleGround {
    GLuint vao, vbo;
    GLuint shader;
    
    void initialize() {
        std::cout << "Initializing simple ground..." << std::endl;
        
        // Simple quad for ground
        float vertices[] = {
            // positions
            -1000.0f, 0.0f, -1000.0f,
             1000.0f, 0.0f, -1000.0f,
             1000.0f, 0.0f,  1000.0f,
            -1000.0f, 0.0f,  1000.0f
        };
        
        glGenVertexArrays(1, &vao);
        glGenBuffers(1, &vbo);
        
        glBindVertexArray(vao);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
        
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(0);
        
        glBindVertexArray(0);
        
        // Create shader for ground
        const char* groundVert = R"(
            #version 330 core
            layout(location = 0) in vec3 aPos;
            uniform mat4 MVP;
            void main() {
                gl_Position = MVP * vec4(aPos, 1.0);
            }
        )";
        
        const char* groundFrag = R"(
            #version 330 core
            out vec3 FragColor;
            void main() {
                FragColor = vec3(0.3, 0.6, 0.3); // Green color
            }
        )";
        
        GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
        glShaderSource(vertexShader, 1, &groundVert, NULL);
        glCompileShader(vertexShader);
        
        GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
        glShaderSource(fragmentShader, 1, &groundFrag, NULL);
        glCompileShader(fragmentShader);
        
        shader = glCreateProgram();
        glAttachShader(shader, vertexShader);
        glAttachShader(shader, fragmentShader);
        glLinkProgram(shader);
        
        glDeleteShader(vertexShader);
        glDeleteShader(fragmentShader);
        
        std::cout << "Ground shader ID: " << shader << std::endl;
    }
    
    void render(const glm::mat4& viewProj) {
        if (!renderGround) return;
        
        glUseProgram(shader);
        
        glm::mat4 mvp = viewProj; // Identity model matrix
        
        GLuint mvpLoc = glGetUniformLocation(shader, "MVP");
        if (mvpLoc != -1) {
            glUniformMatrix4fv(mvpLoc, 1, GL_FALSE, glm::value_ptr(mvp));
        }
        
        glBindVertexArray(vao);
        glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
        glBindVertexArray(0);
    }
    
    void cleanup() {
        glDeleteVertexArrays(1, &vao);
        glDeleteBuffers(1, &vbo);
        glDeleteProgram(shader);
    }
};

static GLuint LoadTextureTileBox(const char *texture_file_path) {
    int w, h, channels;
    uint8_t* img = stbi_load(texture_file_path, &w, &h, &channels, 3);
    GLuint texture;
    glGenTextures(1, &texture);  
    glBindTexture(GL_TEXTURE_2D, texture);  

    // To tile textures on a box, we set wrapping to repeat
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);	
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    if (img) {
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_RGB, GL_UNSIGNED_BYTE, img);
        glGenerateMipmap(GL_TEXTURE_2D);
    } else {
        std::cout << "Failed to load texture " << texture_file_path << std::endl;
    }
    stbi_image_free(img);

    return texture;
}

struct Building {
    glm::vec3 position;        // Position of the box 
    glm::vec3 scale;           // Size of the box in each axis
    
    // Vertex data - move these to local variables in initialize()
    GLfloat vertex_buffer_data[72];
    GLfloat color_buffer_data[72];
    GLuint index_buffer_data[36];
    GLfloat uv_buffer_data[48];
    
    // OpenGL buffers
    GLuint vertexArrayID; 
    GLuint vertexBufferID; 
    GLuint indexBufferID; 
    GLuint colorBufferID;
    GLuint uvBufferID;
    GLuint textureID;

    // Shader variable IDs
    GLuint mvpMatrixID;
    GLuint textureSamplerID;
    GLuint programID;

    void initialize(glm::vec3 position, glm::vec3 scale, int modelID) {
        std::cout << "Initializing building at (" << position.x << ", " 
                  << position.y << ", " << position.z << ")" << std::endl;
        
        // Initialize vertex data
        GLfloat vertex_buffer_data[72] = {    // Vertex definition for a canonical box
            // Front face
            -1.0f, -1.0f, 1.0f, 
            1.0f, -1.0f, 1.0f, 
            1.0f, 1.0f, 1.0f, 
            -1.0f, 1.0f, 1.0f, 
            
            // Back face 
            1.0f, -1.0f, -1.0f, 
            -1.0f, -1.0f, -1.0f, 
            -1.0f, 1.0f, -1.0f, 
            1.0f, 1.0f, -1.0f,
            
            // Left face
            -1.0f, -1.0f, -1.0f, 
            -1.0f, -1.0f, 1.0f, 
            -1.0f, 1.0f, 1.0f, 
            -1.0f, 1.0f, -1.0f, 

            // Right face 
            1.0f, -1.0f, 1.0f, 
            1.0f, -1.0f, -1.0f, 
            1.0f, 1.0f, -1.0f, 
            1.0f, 1.0f, 1.0f,

            // Top face
            -1.0f, 1.0f, 1.0f, 
            1.0f, 1.0f, 1.0f, 
            1.0f, 1.0f, -1.0f, 
            -1.0f, 1.0f, -1.0f, 

            // Bottom face
            -1.0f, -1.0f, -1.0f, 
            1.0f, -1.0f, -1.0f, 
            1.0f, -1.0f, 1.0f, 
            -1.0f, -1.0f, 1.0f, 
        };
        
        GLfloat color_buffer_data[72];
        for (int i = 0; i < 72; ++i) color_buffer_data[i] = 1.0f;  // White color for all vertices

        GLuint index_buffer_data[36] = {        // 12 triangle faces of a box
            0, 1, 2,     
            0, 2, 3, 
            
            4, 5, 6, 
            4, 6, 7, 

            8, 9, 10, 
            8, 10, 11, 

            12, 13, 14, 
            12, 14, 15, 

            16, 17, 18, 
            16, 18, 19, 

            20, 21, 22, 
            20, 22, 23, 
        };

        GLfloat uv_buffer_data[48] = {
            // Front
            0.0f, 1.0f,
            1.0f, 1.0f,
            1.0f, 0.0f,
            0.0f, 0.0f,

            // Back
            0.0f, 1.0f,
            1.0f, 1.0f,
            1.0f, 0.0f,
            0.0f, 0.0f,

            // Left
            0.0f, 1.0f,
            1.0f, 1.0f,
            1.0f, 0.0f,
            0.0f, 0.0f,

            // Right
            0.0f, 1.0f,
            1.0f, 1.0f,
            1.0f, 0.0f,
            0.0f, 0.0f,

            // Top - we do not want texture the top
            0.0f, 0.0f,
            0.0f, 0.0f,
            0.0f, 0.0f,
            0.0f, 0.0f,

            // Bottom - we do not want texture the bottom
            0.0f, 0.0f,
            0.0f, 0.0f,
            0.0f, 0.0f,
            0.0f, 0.0f,
        };
        
        this->position = position;
        this->scale = scale;
        
        // Adjust UVs for texture tiling
        const float building_texture_repeat = 5.0f * (scale.y / scale.x);
        for (int i = 0; i < 24; ++i) {
            uv_buffer_data[2*i+1] *= building_texture_repeat;
        }

        // Create a vertex array object
        glGenVertexArrays(1, &vertexArrayID);
        glBindVertexArray(vertexArrayID);

        // Create a vertex buffer object to store the vertex data        
        glGenBuffers(1, &vertexBufferID);
        glBindBuffer(GL_ARRAY_BUFFER, vertexBufferID);
        glBufferData(GL_ARRAY_BUFFER, sizeof(vertex_buffer_data), vertex_buffer_data, GL_STATIC_DRAW);

        // Create a vertex buffer object to store the color data
        glGenBuffers(1, &colorBufferID);
        glBindBuffer(GL_ARRAY_BUFFER, colorBufferID);
        glBufferData(GL_ARRAY_BUFFER, sizeof(color_buffer_data), color_buffer_data, GL_STATIC_DRAW);

        // Create a vertex buffer object to store the UV data
        glGenBuffers(1, &uvBufferID);
        glBindBuffer(GL_ARRAY_BUFFER, uvBufferID);
        glBufferData(GL_ARRAY_BUFFER, sizeof(uv_buffer_data), uv_buffer_data, GL_STATIC_DRAW);

        // Create an index buffer object to store the index data that defines triangle faces
        glGenBuffers(1, &indexBufferID);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indexBufferID);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(index_buffer_data), index_buffer_data, GL_STATIC_DRAW);

        // Set up vertex attribute pointers
        // Position attribute
        glEnableVertexAttribArray(0);
        glBindBuffer(GL_ARRAY_BUFFER, vertexBufferID);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);
        
        // Color attribute
        glEnableVertexAttribArray(1);
        glBindBuffer(GL_ARRAY_BUFFER, colorBufferID);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);
        
        // UV attribute
        glEnableVertexAttribArray(2);
        glBindBuffer(GL_ARRAY_BUFFER, uvBufferID);
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 0, (void*)0);

        glBindVertexArray(0);  // Unbind VAO

        // Load shaders - FIX THE PATH HERE!
        // Since you have shader.cpp/.h files, let's use LoadShadersFromString with the shader code
        std::string vertShaderCode = R"(#version 330 core
layout(location = 0) in vec3 vertexPosition;
layout(location = 1) in vec3 vertexColor;
layout(location = 2) in vec2 vertexUV;

out vec3 color;
out vec2 uv;

uniform mat4 MVP;

void main() {
    gl_Position = MVP * vec4(vertexPosition, 1);
    color = vertexColor;
    uv = vertexUV;
})";

        std::string fragShaderCode = R"(#version 330 core
in vec3 color;
in vec2 uv;

uniform sampler2D textureSampler;

out vec3 finalColor;

void main() {
    finalColor = color * texture(textureSampler, uv).rgb;
})";

        programID = LoadShadersFromString(vertShaderCode, fragShaderCode);
        
        if (programID == 0) {
            std::cerr << "Failed to load shaders for building." << std::endl;
        } else {
            std::cout << "Building shader loaded successfully, ID: " << programID << std::endl;
        }

        // Get a handle for our "MVP" uniform
        mvpMatrixID = glGetUniformLocation(programID, "MVP");
        std::cout << "MVP uniform location: " << mvpMatrixID << std::endl;

        // Load a texture - FIX THE PATH HERE!
        // Since we don't have the texture files, let's create a simple texture
        std::string texture_file_path = "../assignment1/model/ground/ground.jpg";  // Simplified path
        
        // Try to load texture, if fails create a fallback
        textureID = LoadTextureTileBox(texture_file_path.c_str());
        if (glIsTexture(textureID) == GL_FALSE && textureID == 0) {
            std::cout << "Creating fallback texture for building..." << std::endl;
            textureID = createFallbackBuildingTexture();
        }
        
        std::cout << "Building texture ID: " << textureID << std::endl;

        // Get a handle for our "textureSampler" uniform
        textureSamplerID = glGetUniformLocation(programID, "textureSampler");
        std::cout << "Texture sampler location: " << textureSamplerID << std::endl;
    }
    
    void render(glm::mat4 viewProj) {
        if (programID == 0 || !renderBuildings) return;
        
        glUseProgram(programID);
        
        // Bind the VAO
        glBindVertexArray(vertexArrayID);
        
        // Enable vertex attributes
        glEnableVertexAttribArray(0);
        glEnableVertexAttribArray(1);
        glEnableVertexAttribArray(2);
        
        // Model transform 
        glm::mat4 modelMatrix = glm::mat4(1.0f);    
        // Scale the box along each axis to make it look like a building
        modelMatrix = glm::translate(modelMatrix, position);
        modelMatrix = glm::scale(modelMatrix, glm::vec3(scale.x, scale.y * 5, scale.z));
        
        // Set model-view-projection matrix
        glm::mat4 mvp = viewProj * modelMatrix;
        glUniformMatrix4fv(mvpMatrixID, 1, GL_FALSE, glm::value_ptr(mvp));
        
        // Bind texture
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, textureID);
        glUniform1i(textureSamplerID, 0);
        
        // Draw the box
        glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, 0);
        
        // Cleanup
        glBindVertexArray(0);
    }
    
    void cleanup() {
        glDeleteBuffers(1, &vertexBufferID);
        glDeleteBuffers(1, &colorBufferID);
        glDeleteBuffers(1, &indexBufferID);
        glDeleteVertexArrays(1, &vertexArrayID);
        glDeleteBuffers(1, &uvBufferID);
        glDeleteTextures(1, &textureID);
        glDeleteProgram(programID);
    }

private:
    // Helper function to create a fallback building texture
    GLuint createFallbackBuildingTexture() {
        GLuint textureID;
        glGenTextures(1, &textureID);
        glBindTexture(GL_TEXTURE_2D, textureID);
        
        // Create a 64x64 brick pattern
        const int width = 64;
        const int height = 64;
        std::vector<unsigned char> pixels(width * height * 3);
        
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int idx = (y * width + x) * 3;
                
                // Create brick pattern
                int brickX = x / 16;
                int brickY = y / 8;
                
                if ((brickX + brickY) % 2 == 0) {
                    // Brick color
                    pixels[idx] = 180;
                    pixels[idx + 1] = 100;
                    pixels[idx + 2] = 60;
                } else {
                    // Mortar color
                    pixels[idx] = 100;
                    pixels[idx + 1] = 100;
                    pixels[idx + 2] = 100;
                }
            }
        }
        
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, pixels.data());
        
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        
        return textureID;
    }
};
// MAIN FUNCTION
// ====================
int main(void) {
    // Initialise GLFW
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW." << std::endl;
        return -1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    window = glfwCreateWindow(windowWidth, windowHeight, "Debug Test - Simple Scene", NULL, NULL);
    if (window == NULL) {
        std::cerr << "Failed to open a GLFW window." << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);

    // Load OpenGL functions
    int version = gladLoadGL(glfwGetProcAddress);
    if (version == 0) {
        std::cerr << "Failed to initialize OpenGL context." << std::endl;
        return -1;
    }

    std::cout << "\n=== OpenGL Info ===" << std::endl;
    std::cout << "OpenGL Version: " << glGetString(GL_VERSION) << std::endl;
    std::cout << "GLSL Version: " << glGetString(GL_SHADING_LANGUAGE_VERSION) << std::endl;
    std::cout << "Vendor: " << glGetString(GL_VENDOR) << std::endl;
    std::cout << "Renderer: " << glGetString(GL_RENDERER) << std::endl << std::endl;

    // Set viewport
    glViewport(0, 0, windowWidth, windowHeight);
    
    // OpenGL settings
    glClearColor(0.1f, 0.1f, 0.15f, 1.0f);
    glEnable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE); // Disable culling for debugging

    // Initialize test objects
    TestCube testCube;
    SimpleGround ground;
    std::vector<Building> buildings;
    
    std::cout << "=== Initializing Test Scene ===" << std::endl;
    testCube.initialize();
    ground.initialize();

    std::cout << "=== Initializing Buildings ===" << std::endl;


    
    const int gridSize = 4;
	const int buildingDistance = 70;
	const int buildingMinSize=10;
	const int buildingMaxSize = 30;
    for (int i = 0; i < gridSize; i++)
	{
		for (int j = 0; j < gridSize; j++)
		{
			Building b;
			const int buildingWidth = abs(pow(5+i, 8+j)) % (buildingMaxSize - buildingMinSize) + 10;
			const int buildingHeight = abs(pow(6+i, 9+j)) % (buildingMaxSize - buildingMinSize) + 10;
			const int buildingDepth = buildingWidth;
			const int buildingModel = abs(pow(4+i, 5+j)) % 5;
			const glm::vec3 buildingSize = glm::vec3(buildingWidth,buildingHeight,buildingDepth);
			const glm::vec3 buildingLocation = glm::vec3(
				(-buildingDistance*gridSize)/2 + (i * buildingDistance), 
				buildingHeight*5,
				(-buildingDistance*gridSize)/2 + (j * buildingDistance)
			);
			b.initialize(buildingLocation, buildingSize, buildingModel);
			buildings.push_back(b);
		}
	}
    
    // Update camera front based on initial pitch/yaw
    glm::vec3 front;
    front.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
    front.y = sin(glm::radians(pitch));
    front.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
    cameraFront = glm::normalize(front);

    // Set up callbacks AFTER window creation
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    glfwSetCursorPosCallback(window, mouse_callback);
    glfwSetScrollCallback(window, scroll_callback);
    glfwSetKeyCallback(window, key_callback);

    // Time tracking
    static double lastTime = glfwGetTime();
    float fTime = 0.0f;
    unsigned long frames = 0;

    std::cout << "\n=== Starting Render Loop ===" << std::endl;
    std::cout << "Camera position: (" << cameraPos.x << ", " << cameraPos.y << ", " << cameraPos.z << ")" << std::endl;
    std::cout << "Controls:" << std::endl;
    std::cout << "  - WASD: Move camera" << std::endl;
    std::cout << "  - Mouse: Look around" << std::endl;
    std::cout << "  - Scroll: Zoom in/out" << std::endl;
    std::cout << "  - R: Reset camera" << std::endl;
    std::cout << "  - 1: Toggle test cube" << std::endl;
    std::cout << "  - 3: Toggle ground" << std::endl;
    std::cout << "  - ESC: Exit" << std::endl << std::endl;

    // Main loop
    do {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Calculate delta time
        double currentTime = glfwGetTime();
        float deltaTime = float(currentTime - lastTime);
        lastTime = currentTime;

        // Process input
        processInput(window, deltaTime);

        // Camera matrices
        glm::mat4 projection = glm::perspective(glm::radians(FoV), 
                                               (float)windowWidth / windowHeight, 
                                               zNear, zFar);
        glm::mat4 view = glm::lookAt(cameraPos, cameraPos + cameraFront, cameraUp);
        glm::mat4 viewProj = projection * view;

        // Render test objects
        ground.render(viewProj);
        testCube.render(viewProj);
        glDisable(GL_DEPTH_TEST);
        for (int i = 0; i < buildings.size(); i++)
		{
			buildings[i].render(viewProj);
            checkGLError(("Building " + std::to_string(i)).c_str());
		}

        // FPS tracking
        frames++;
        fTime += deltaTime;
        if (fTime > 2.0f) {
            float fps = frames / fTime;
            frames = 0;
            fTime = 0;
            
            std::stringstream stream;
            stream << std::fixed << std::setprecision(1) 
                   << "Debug Test | FPS: " << fps 
                   << " | Pos: (" << (int)cameraPos.x << ", " << (int)cameraPos.y << ", " << (int)cameraPos.z << ")"
                   << " | Cube: " << (renderRobot ? "ON" : "OFF")
                   << " | Ground: " << (renderGround ? "ON" : "OFF");
            glfwSetWindowTitle(window, stream.str().c_str());
        }

        // Swap buffers and poll events
        glfwSwapBuffers(window);
        glfwPollEvents();

    } while (!glfwWindowShouldClose(window));

    // Cleanup
    testCube.cleanup();
    for (int i = 0; i < buildings.size(); i++)
    {
        buildings[i].cleanup();
    }
    ground.cleanup();

    glfwTerminate();
    return 0;
}

// ====================
// INPUT PROCESSING
// ====================
void processInput(GLFWwindow* window, float deltaTime) {
    // Movement
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        cameraPos += cameraSpeed * deltaTime * cameraFront;
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        cameraPos -= cameraSpeed * deltaTime * cameraFront;
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        cameraPos -= glm::normalize(glm::cross(cameraFront, cameraUp)) * cameraSpeed * deltaTime;
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        cameraPos += glm::normalize(glm::cross(cameraFront, cameraUp)) * cameraSpeed * deltaTime;
    if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS)
        cameraPos.y -= cameraSpeed * deltaTime;
    if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS)
        cameraPos.y += cameraSpeed * deltaTime;
    
    // Reset camera
    if (glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS) {
        cameraPos = glm::vec3(0.0f, 100.0f, 500.0f);
        yaw = -90.0f;
        pitch = -10.0f;
        
        glm::vec3 front;
        front.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
        front.y = sin(glm::radians(pitch));
        front.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
        cameraFront = glm::normalize(front);
        
        std::cout << "Camera reset to: (" << cameraPos.x << ", " << cameraPos.y << ", " << cameraPos.z << ")" << std::endl;
    }
    
    // Toggle test cube (using key 1)
    static bool key1Pressed = false;
    if (glfwGetKey(window, GLFW_KEY_1) == GLFW_PRESS) {
        if (!key1Pressed) {
            renderRobot = !renderRobot;
            std::cout << "Test cube: " << (renderRobot ? "ON" : "OFF") << std::endl;
            key1Pressed = true;
        }
    } else {
        key1Pressed = false;
    }

    //Toggle buildings (using key 2)
    static bool key2Pressed = false;
    if (glfwGetKey(window, GLFW_KEY_2) == GLFW_PRESS) {
        if (!key2Pressed) {
            renderBuildings = !renderBuildings;
            std::cout << "Buildings: " << (renderBuildings ? "ON" : "OFF") << std::endl;
            key2Pressed = true;
        }
    } else {
        key2Pressed = false;
    }
    
    // Toggle ground (using key 3)
    static bool key3Pressed = false;
    if (glfwGetKey(window, GLFW_KEY_3) == GLFW_PRESS) {
        if (!key3Pressed) {
            renderGround = !renderGround;
            std::cout << "Ground: " << (renderGround ? "ON" : "OFF") << std::endl;
            key3Pressed = true;
        }
    } else {
        key3Pressed = false;
    }
}

static void mouse_callback(GLFWwindow* window, double xpos, double ypos) {
    if (firstMouse) {
        lastX = xpos;
        lastY = ypos;
        firstMouse = false;
    }

    float xoffset = xpos - lastX;
    float yoffset = lastY - ypos;
    lastX = xpos;
    lastY = ypos;

    float sensitivity = 0.1f;
    xoffset *= sensitivity;
    yoffset *= sensitivity;

    yaw += xoffset;
    pitch += yoffset;

    // Constrain pitch
    if (pitch > 89.0f)
        pitch = 89.0f;
    if (pitch < -89.0f)
        pitch = -89.0f;

    glm::vec3 front;
    front.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
    front.y = sin(glm::radians(pitch));
    front.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
    cameraFront = glm::normalize(front);
}

static void scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
    FoV -= (float)yoffset;
    if (FoV < 1.0f)
        FoV = 1.0f;
    if (FoV > 90.0f)
        FoV = 90.0f;
}

static void key_callback(GLFWwindow *window, int key, int scancode, int action, int mode) {
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        glfwSetWindowShouldClose(window, GL_TRUE);
}