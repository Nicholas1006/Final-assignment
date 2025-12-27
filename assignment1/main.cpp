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

// GLTF model loader
#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <tiny_gltf.h>
//#include <stb/stb_image.h>

#define _USE_MATH_DEFINES
#include <math.h>


#define BUFFER_OFFSET(i) ((char *)NULL + (i))
// ====================
// RENDER CONTROLS
// ====================
static bool renderRobot = true;
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


struct MyBot {
	// Shader variable IDs
	GLuint mvpMatrixID;
	GLuint jointMatricesID;
	GLuint lightPositionID;
	GLuint lightIntensityID;
	GLuint programID;
    glm::vec3 lightPosition;
    glm::vec3 lightIntensity;

	tinygltf::Model model;

	// Each VAO corresponds to each mesh primitive in the GLTF model
	struct PrimitiveObject {
		GLuint vao;
		std::map<int, GLuint> vbos;
	};
	std::vector<PrimitiveObject> primitiveObjects;

	// Skinning 
	struct SkinObject {
		// Transforms the geometry into the space of the respective joint
		std::vector<glm::mat4> inverseBindMatrices;  

		// Transforms the geometry following the movement of the joints
		std::vector<glm::mat4> globalJointTransforms;

		// Combined transforms
		std::vector<glm::mat4> jointMatrices;
	};
	std::vector<SkinObject> skinObjects;

	// Animation 
	struct SamplerObject {
		std::vector<float> input;
		std::vector<glm::vec4> output;
		int interpolation;
	};
	struct AnimationObject {
		std::vector<SamplerObject> samplers;	// Animation data
	};
	std::vector<AnimationObject> animationObjects;

	glm::mat4 getNodeTransform(const tinygltf::Node& node) {
		glm::mat4 transform(1.0f); 

		if (node.matrix.size() == 16) {
			transform = glm::make_mat4(node.matrix.data());
		} else {
			if (node.translation.size() == 3) {
				transform = glm::translate(transform, glm::vec3(node.translation[0], node.translation[1], node.translation[2]));
			}
			if (node.rotation.size() == 4) {
				glm::quat q(node.rotation[3], node.rotation[0], node.rotation[1], node.rotation[2]);
				transform *= glm::mat4_cast(q);
			}
			if (node.scale.size() == 3) {
				transform = glm::scale(transform, glm::vec3(node.scale[0], node.scale[1], node.scale[2]));
			}
		}
		return transform;
	}

	void computeLocalNodeTransform(const tinygltf::Model& model,
		int nodeIndex,
		std::vector<glm::mat4>& localTransforms)
	{
		const tinygltf::Node& node = model.nodes[nodeIndex];
		glm::mat4 localTransform = getNodeTransform(node);
		localTransforms[nodeIndex] = localTransform;
		for (int childIndex : node.children) {
			computeLocalNodeTransform(model, childIndex, localTransforms);
		}
	}

	void computeGlobalNodeTransform(const tinygltf::Model& model,
		const std::vector<glm::mat4>& localTransforms,
		int nodeIndex, const glm::mat4& parentTransform,
		std::vector<glm::mat4>& globalTransforms)
	{
		const tinygltf::Node& node = model.nodes[nodeIndex];
		glm::mat4 globalTransform = parentTransform * localTransforms[nodeIndex];
		globalTransforms[nodeIndex] = globalTransform;
		for (int childIndex : node.children) {
			computeGlobalNodeTransform(model, localTransforms, childIndex, globalTransform, globalTransforms);
		}
	}

	std::vector<SkinObject> prepareSkinning(const tinygltf::Model &model) {
		std::vector<SkinObject> skinObjects;

		for (size_t i = 0; i < model.skins.size(); i++) {
			SkinObject skinObject;

			const tinygltf::Skin &skin = model.skins[i];

			// Read inverseBindMatrices
			const tinygltf::Accessor &accessor = model.accessors[skin.inverseBindMatrices];
			assert(accessor.type == TINYGLTF_TYPE_MAT4);
			const tinygltf::BufferView &bufferView = model.bufferViews[accessor.bufferView];
			const tinygltf::Buffer &buffer = model.buffers[bufferView.buffer];
			const float *ptr = reinterpret_cast<const float *>(
				buffer.data.data() + accessor.byteOffset + bufferView.byteOffset);
			
			skinObject.inverseBindMatrices.resize(accessor.count);
			for (size_t j = 0; j < accessor.count; j++) {
				float m[16];
				memcpy(m, ptr + j * 16, 16 * sizeof(float));
				skinObject.inverseBindMatrices[j] = glm::make_mat4(m);
			}

			assert(skin.joints.size() == accessor.count);

			skinObject.globalJointTransforms.resize(skin.joints.size());
			skinObject.jointMatrices.resize(skin.joints.size());

			// Compute local transforms for ALL nodes in the model
			std::vector<glm::mat4> localNodeTransforms(model.nodes.size(), glm::mat4(1.0f));
			
			// Start from scene root nodes
			const tinygltf::Scene& scene = model.scenes[model.defaultScene];
			for (int rootNodeIndex : scene.nodes) {
				computeLocalNodeTransform(model, rootNodeIndex, localNodeTransforms);
			}

			// Compute global transforms for ALL nodes
			std::vector<glm::mat4> globalNodeTransforms(model.nodes.size(), glm::mat4(1.0f));
			for (int rootNodeIndex : scene.nodes) {
				glm::mat4 parentTransform(1.0f);
				computeGlobalNodeTransform(model, localNodeTransforms, rootNodeIndex, 
										parentTransform, globalNodeTransforms);
			}

			// Extract joint transforms from the global node transforms
			for (size_t j = 0; j < skin.joints.size(); ++j) {
				int jointNodeIndex = skin.joints[j];
				skinObject.globalJointTransforms[j] = globalNodeTransforms[jointNodeIndex];
				skinObject.jointMatrices[j] = skinObject.globalJointTransforms[j] * 
											skinObject.inverseBindMatrices[j];
			}

			skinObjects.push_back(skinObject);
		}
		return skinObjects;
	}

	int findKeyframeIndex(const std::vector<float>& times, float animationTime) 
	{
		int left = 0;
		int right = times.size() - 1;

		while (left <= right) {
			int mid = (left + right) / 2;

			if (mid + 1 < times.size() && times[mid] <= animationTime && animationTime < times[mid + 1]) {
				return mid;
			}
			else if (times[mid] > animationTime) {
				right = mid - 1;
			}
			else { // animationTime >= times[mid + 1]
				left = mid + 1;
			}
		}

		// Target not found
		return times.size() - 2;
	}

	std::vector<AnimationObject> prepareAnimation(const tinygltf::Model &model) 
	{
		std::vector<AnimationObject> animationObjects;
		for (const auto &anim : model.animations) {
			AnimationObject animationObject;
			
			for (const auto &sampler : anim.samplers) {
				SamplerObject samplerObject;

				const tinygltf::Accessor &inputAccessor = model.accessors[sampler.input];
				const tinygltf::BufferView &inputBufferView = model.bufferViews[inputAccessor.bufferView];
				const tinygltf::Buffer &inputBuffer = model.buffers[inputBufferView.buffer];

				assert(inputAccessor.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT);
				assert(inputAccessor.type == TINYGLTF_TYPE_SCALAR);

				// Input (time) values
				samplerObject.input.resize(inputAccessor.count);

				const unsigned char *inputPtr = &inputBuffer.data[inputBufferView.byteOffset + inputAccessor.byteOffset];
				const float *inputBuf = reinterpret_cast<const float*>(inputPtr);

				// Read input (time) values
				int stride = inputAccessor.ByteStride(inputBufferView);
				for (size_t i = 0; i < inputAccessor.count; ++i) {
					samplerObject.input[i] = *reinterpret_cast<const float*>(inputPtr + i * stride);
				}
				
				const tinygltf::Accessor &outputAccessor = model.accessors[sampler.output];
				const tinygltf::BufferView &outputBufferView = model.bufferViews[outputAccessor.bufferView];
				const tinygltf::Buffer &outputBuffer = model.buffers[outputBufferView.buffer];

				assert(outputAccessor.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT);

				const unsigned char *outputPtr = &outputBuffer.data[outputBufferView.byteOffset + outputAccessor.byteOffset];
				const float *outputBuf = reinterpret_cast<const float*>(outputPtr);

				int outputStride = outputAccessor.ByteStride(outputBufferView);
				
				// Output values
				samplerObject.output.resize(outputAccessor.count);
				
				for (size_t i = 0; i < outputAccessor.count; ++i) {

					if (outputAccessor.type == TINYGLTF_TYPE_VEC3) {
						memcpy(&samplerObject.output[i], outputPtr + i * 3 * sizeof(float), 3 * sizeof(float));
					} else if (outputAccessor.type == TINYGLTF_TYPE_VEC4) {
						memcpy(&samplerObject.output[i], outputPtr + i * 4 * sizeof(float), 4 * sizeof(float));
					} else {
						std::cout << "Unsupport accessor type ..." << std::endl;
					}

				}

				animationObject.samplers.push_back(samplerObject);			
			}

			animationObjects.push_back(animationObject);
		}
		return animationObjects;
	}

	void updateAnimation(
		const tinygltf::Model &model, 
		const tinygltf::Animation &anim, 
		const AnimationObject &animationObject, 
		float time,
		std::vector<glm::mat4> &nodeTransforms) 
	{
		for (const auto &channel : anim.channels) {
			int targetNodeIndex = channel.target_node;
			const auto &sampler = anim.samplers[channel.sampler];
			
			// Access output (value) data for the channel
			const tinygltf::Accessor &outputAccessor = model.accessors[sampler.output];
			const tinygltf::BufferView &outputBufferView = model.bufferViews[outputAccessor.bufferView];
			const tinygltf::Buffer &outputBuffer = model.buffers[outputBufferView.buffer];

			// Calculate current animation time (wrap if necessary)
			const std::vector<float> &times = animationObject.samplers[channel.sampler].input;
			float animationTime = fmod(time, times.back());
			
			int keyframeIndex = findKeyframeIndex(times, animationTime);
			int nextKeyframeIndex = keyframeIndex + 1;

			const unsigned char *outputPtr = &outputBuffer.data[outputBufferView.byteOffset + 
															outputAccessor.byteOffset];

			float t = 0.0f;
			if (nextKeyframeIndex < times.size() && times[nextKeyframeIndex] > times[keyframeIndex]) {
				float timeBetweenFrames = times[nextKeyframeIndex] - times[keyframeIndex];
				float timeSinceLastFrame = animationTime - times[keyframeIndex];
				t = timeSinceLastFrame / timeBetweenFrames;
			}
			t = glm::clamp(t, 0.0f, 1.0f);
			
			if (channel.target_path == "translation") {
				glm::vec3 translation0, translation1;
				memcpy(&translation0, outputPtr + keyframeIndex * 3 * sizeof(float), 3 * sizeof(float));
				
				glm::vec3 translation1_valid;
				if (nextKeyframeIndex < times.size()) {
					memcpy(&translation1_valid, outputPtr + nextKeyframeIndex * 3 * sizeof(float), 3 * sizeof(float));
				} else {
					translation1_valid = translation0;
				}
				
				glm::vec3 translation = glm::mix(translation0, translation1_valid, t);
				
				// Start with identity matrix if not initialized
				if (nodeTransforms[targetNodeIndex] == glm::mat4(0.0f)) {
					nodeTransforms[targetNodeIndex] = glm::mat4(1.0f);
				}
				nodeTransforms[targetNodeIndex] = glm::translate(nodeTransforms[targetNodeIndex], translation);
				
			} else if (channel.target_path == "rotation") {
				glm::quat rotation0, rotation1;
				memcpy(&rotation0, outputPtr + keyframeIndex * 4 * sizeof(float), 4 * sizeof(float));
				
				glm::quat rotation1_valid;
				if (nextKeyframeIndex < times.size()) {
					memcpy(&rotation1_valid, outputPtr + nextKeyframeIndex * 4 * sizeof(float), 4 * sizeof(float));
				} else {
					rotation1_valid = rotation0;
				}
				
				glm::quat rotation = glm::slerp(rotation0, rotation1_valid, t);
				
				if (nodeTransforms[targetNodeIndex] == glm::mat4(0.0f)) {
					nodeTransforms[targetNodeIndex] = glm::mat4(1.0f);
				}
				nodeTransforms[targetNodeIndex] *= glm::mat4_cast(rotation);
				
			} else if (channel.target_path == "scale") {
				glm::vec3 scale0, scale1;
				memcpy(&scale0, outputPtr + keyframeIndex * 3 * sizeof(float), 3 * sizeof(float));
				
				glm::vec3 scale1_valid;
				if (nextKeyframeIndex < times.size()) {
					memcpy(&scale1_valid, outputPtr + nextKeyframeIndex * 3 * sizeof(float), 3 * sizeof(float));
				} else {
					scale1_valid = scale0;
				} 
				
				glm::vec3 scale = glm::mix(scale0, scale1_valid, t);
				
				if (nodeTransforms[targetNodeIndex] == glm::mat4(0.0f)) {
					nodeTransforms[targetNodeIndex] = glm::mat4(1.0f);
				}
				nodeTransforms[targetNodeIndex] = glm::scale(nodeTransforms[targetNodeIndex], scale);
			}
		}
	}

	void updateSkinning(const std::vector<glm::mat4> &globalNodeTransforms) {
		if (skinObjects.empty() || model.skins.empty()) return;
		
		SkinObject &skinObject = skinObjects[0];
		const tinygltf::Skin &skin = model.skins[0];
		
		// Extract joint transforms from global node transforms
		for (size_t j = 0; j < skin.joints.size(); ++j) {
			int jointNodeIndex = skin.joints[j];
			if (jointNodeIndex < globalNodeTransforms.size()) {
				skinObject.globalJointTransforms[j] = globalNodeTransforms[jointNodeIndex];
				skinObject.jointMatrices[j] = skinObject.globalJointTransforms[j] * 
											skinObject.inverseBindMatrices[j];
			}
		}
	}

	void update(float time) {
		if (model.animations.empty() || animationObjects.empty()) return;
		
		// Initialize node transforms with identity
		std::vector<glm::mat4> nodeTransforms(model.nodes.size(), glm::mat4(1.0f));
		
		// Apply animation
		const auto& anim = model.animations[0];
		const auto& animationObject = animationObjects[0];
		updateAnimation(model, anim, animationObject, time, nodeTransforms);
		
		// MAKE THE BOT WALK STRAIGHT INSTEAD OF IN A CIRCLE
		// Find the root joint (hip bone) - usually the first joint
		if (!model.skins.empty()) {
			const tinygltf::Skin& skin = model.skins[0];
			if (!skin.joints.empty()) {
				int rootJointIndex = skin.joints[0];  // Root joint is usually the first in the skin
				
				// Extract the current root transform
				glm::mat4 rootTransform = nodeTransforms[rootJointIndex];
				
				// Extract translation from root transform
				glm::vec3 translation = glm::vec3(rootTransform[3]);
				
				// Calculate forward distance (Z-axis) from the circular motion
				// Assuming the circular motion has components in both X and Z
				// We'll take the magnitude of the XZ vector as the forward speed
				float forwardDistance = glm::length(glm::vec2(translation.x, translation.z));
				
				// Create new translation - only forward (Z-axis), no lateral movement (X-axis)
				// Keep the original Y (vertical) movement
				glm::vec3 straightTranslation(0.0f, translation.y, forwardDistance);
				
				// Remove any rotation from the root to keep it facing forward
				// Extract scale if any
				glm::vec3 scale = glm::vec3(
					glm::length(glm::vec3(rootTransform[0])),
					glm::length(glm::vec3(rootTransform[1])),
					glm::length(glm::vec3(rootTransform[2]))
				);
				
				// Reconstruct root transform with only forward translation and scale
				// No rotation to keep it facing forward
				nodeTransforms[rootJointIndex] = glm::translate(glm::mat4(1.0f), straightTranslation) * 
												glm::scale(glm::mat4(1.0f), scale);
			}
		}
		
		// Compute global transforms for all nodes
		std::vector<glm::mat4> globalNodeTransforms(model.nodes.size(), glm::mat4(1.0f));
		
		// Start from root nodes (typically from scene)
		const tinygltf::Scene& scene = model.scenes[model.defaultScene];
		for (int rootNodeIndex : scene.nodes) {
			glm::mat4 parentTransform(1.0f);
			computeGlobalNodeTransform(model, nodeTransforms, rootNodeIndex, 
									parentTransform, globalNodeTransforms);
		}
		
		// Update skinning with global transforms
		updateSkinning(globalNodeTransforms);
	}

	bool loadModel(tinygltf::Model &model, const char *filename) {
		tinygltf::TinyGLTF loader;
		std::string err;
		std::string warn;

		bool res = loader.LoadASCIIFromFile(&model, &err, &warn, filename);
		if (!warn.empty()) {
			std::cout << "WARN: " << warn << std::endl;
		}

		if (!err.empty()) {
			std::cout << "ERR: " << err << std::endl;
		}

		if (!res)
			std::cout << "Failed to load glTF: " << filename << std::endl;
		else
			std::cout << "Loaded glTF: " << filename << std::endl;

		return res;
	}

	void initialize() {
		// Initialize light
		lightPosition = glm::vec3(0.0f, 10.0f, 0.0f);
		lightIntensity = glm::vec3(1.0f, 1.0f, 1.0f);

		// Load model
		if (!loadModel(model, "../assignment1/model/bot/bot.gltf")) {
			return;
		}

		// Prepare buffers for rendering 
		primitiveObjects = bindModel(model);

		// Prepare joint matrices
		skinObjects = prepareSkinning(model);

		// Prepare animation data 
		animationObjects = prepareAnimation(model);

		// Create and compile our GLSL program from the shaders
		programID = LoadShadersFromFile("../assignment1/shader/bot.vert", "../assignment1/shader/bot.frag");
		if (programID == 0)
		{
			std::cerr << "Failed to load shaders." << std::endl;
		}

		// Get a handle for GLSL variables
		mvpMatrixID = glGetUniformLocation(programID, "MVP");
		jointMatricesID = glGetUniformLocation(programID, "jointMatrices");
		lightPositionID = glGetUniformLocation(programID, "lightPosition");
		lightIntensityID = glGetUniformLocation(programID, "lightIntensity");
		
		// Initialize with initial transforms
		std::vector<glm::mat4> initialGlobalTransforms(model.nodes.size(), glm::mat4(1.0f));
		const tinygltf::Scene& scene = model.scenes[model.defaultScene];
		for (int rootNodeIndex : scene.nodes) {
			glm::mat4 parentTransform(1.0f);
			std::vector<glm::mat4> localTransforms(model.nodes.size(), glm::mat4(1.0f));
			computeGlobalNodeTransform(model, localTransforms, rootNodeIndex, 
									parentTransform, initialGlobalTransforms);
		}
		updateSkinning(initialGlobalTransforms);
	}

	void bindMesh(std::vector<PrimitiveObject> &primitiveObjects,
				tinygltf::Model &model, tinygltf::Mesh &mesh) {

		std::map<int, GLuint> vbos;
		for (size_t i = 0; i < model.bufferViews.size(); ++i) {
			const tinygltf::BufferView &bufferView = model.bufferViews[i];

			int target = bufferView.target;
			
			if (bufferView.target == 0) { 
				// The bufferView with target == 0 in our model refers to 
				// the skinning weights, for 25 joints, each 4x4 matrix (16 floats), totaling to 400 floats or 1600 bytes. 
				// So it is considered safe to skip the warning.
				continue;
			}

			const tinygltf::Buffer &buffer = model.buffers[bufferView.buffer];
			GLuint vbo;
			glGenBuffers(1, &vbo);
			glBindBuffer(target, vbo);
			glBufferData(target, bufferView.byteLength,
						&buffer.data.at(0) + bufferView.byteOffset, GL_STATIC_DRAW);
			
			vbos[i] = vbo;
		}

		// Each mesh can contain several primitives (or parts), each we need to 
		// bind to an OpenGL vertex array object
		for (size_t i = 0; i < mesh.primitives.size(); ++i) {

			tinygltf::Primitive primitive = mesh.primitives[i];
			tinygltf::Accessor indexAccessor = model.accessors[primitive.indices];

			GLuint vao;
			glGenVertexArrays(1, &vao);
			glBindVertexArray(vao);

			for (auto &attrib : primitive.attributes) {
				tinygltf::Accessor accessor = model.accessors[attrib.second];
				int byteStride =
					accessor.ByteStride(model.bufferViews[accessor.bufferView]);
				glBindBuffer(GL_ARRAY_BUFFER, vbos[accessor.bufferView]);

				int size = 1;
				if (accessor.type != TINYGLTF_TYPE_SCALAR) {
					size = accessor.type;
				}

				int vaa = -1;
				if (attrib.first.compare("POSITION") == 0) vaa = 0;
				if (attrib.first.compare("NORMAL") == 0) vaa = 1;
				if (attrib.first.compare("TEXCOORD_0") == 0) vaa = 2;
				if (attrib.first.compare("JOINTS_0") == 0) vaa = 3;
				if (attrib.first.compare("WEIGHTS_0") == 0) vaa = 4;
				if (vaa > -1) {
					glEnableVertexAttribArray(vaa);
					glVertexAttribPointer(vaa, size, accessor.componentType,
										accessor.normalized ? GL_TRUE : GL_FALSE,
										byteStride, BUFFER_OFFSET(accessor.byteOffset));
				} else {
					std::cout << "vaa missing: " << attrib.first << std::endl;
				}
			}

			// Record VAO for later use
			PrimitiveObject primitiveObject;
			primitiveObject.vao = vao;
			primitiveObject.vbos = vbos;
			primitiveObjects.push_back(primitiveObject);

			glBindVertexArray(0);
		}
	}

	void bindModelNodes(std::vector<PrimitiveObject> &primitiveObjects, 
						tinygltf::Model &model,
						tinygltf::Node &node) {
		// Bind buffers for the current mesh at the node
		if ((node.mesh >= 0) && (node.mesh < model.meshes.size())) {
			bindMesh(primitiveObjects, model, model.meshes[node.mesh]);
		}

		// Recursive into children nodes
		for (size_t i = 0; i < node.children.size(); i++) {
			assert((node.children[i] >= 0) && (node.children[i] < model.nodes.size()));
			bindModelNodes(primitiveObjects, model, model.nodes[node.children[i]]);
		}
	}

	std::vector<PrimitiveObject> bindModel(tinygltf::Model &model) {
		std::vector<PrimitiveObject> primitiveObjects;

		const tinygltf::Scene &scene = model.scenes[model.defaultScene];
		for (size_t i = 0; i < scene.nodes.size(); ++i) {
			assert((scene.nodes[i] >= 0) && (scene.nodes[i] < model.nodes.size()));
			bindModelNodes(primitiveObjects, model, model.nodes[scene.nodes[i]]);
		}

		return primitiveObjects;
	}

	void drawMesh(const std::vector<PrimitiveObject> &primitiveObjects,
				tinygltf::Model &model, tinygltf::Mesh &mesh) {
		
		for (size_t i = 0; i < mesh.primitives.size(); ++i) 
		{
			GLuint vao = primitiveObjects[i].vao;
			std::map<int, GLuint> vbos = primitiveObjects[i].vbos;

			glBindVertexArray(vao);

			tinygltf::Primitive primitive = mesh.primitives[i];
			tinygltf::Accessor indexAccessor = model.accessors[primitive.indices];

			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbos.at(indexAccessor.bufferView));

			glDrawElements(primitive.mode, indexAccessor.count,
						indexAccessor.componentType,
						BUFFER_OFFSET(indexAccessor.byteOffset));

			glBindVertexArray(0);
		}
	}

	void drawModelNodes(const std::vector<PrimitiveObject>& primitiveObjects,
						tinygltf::Model &model, tinygltf::Node &node) {
		// Draw the mesh at the node, and recursively do so for children nodes
		if ((node.mesh >= 0) && (node.mesh < model.meshes.size())) {
			drawMesh(primitiveObjects, model, model.meshes[node.mesh]);
		}
		for (size_t i = 0; i < node.children.size(); i++) {
			drawModelNodes(primitiveObjects, model, model.nodes[node.children[i]]);
		}
	}
	
	void drawModel(const std::vector<PrimitiveObject>& primitiveObjects,
				tinygltf::Model &model) {
		// Draw all nodes
		const tinygltf::Scene &scene = model.scenes[model.defaultScene];
		for (size_t i = 0; i < scene.nodes.size(); ++i) {
			drawModelNodes(primitiveObjects, model, model.nodes[scene.nodes[i]]);
		}
	}

	void render(glm::mat4 cameraMatrix) {
		glUseProgram(programID);
		
		// Set camera
		glm::mat4 mvp = cameraMatrix;
		glUniformMatrix4fv(mvpMatrixID, 1, GL_FALSE, &mvp[0][0]);

		// Set joint matrices if available
		if (!skinObjects.empty()) {
			const SkinObject& skinObject = skinObjects[0];
			// Make sure we have the uniform location
			if (jointMatricesID == (GLuint)-1) {
				jointMatricesID = glGetUniformLocation(programID, "jointMatrices");
			}
			if (jointMatricesID != (GLuint)-1) {
				glUniformMatrix4fv(jointMatricesID, skinObject.jointMatrices.size(), 
								GL_FALSE, glm::value_ptr(skinObject.jointMatrices[0]));
			}
		}

		// Set light data 
		glUniform3fv(lightPositionID, 1, &lightPosition[0]);
		glUniform3fv(lightIntensityID, 1, &lightIntensity[0]);

		// Draw the GLTF model
		drawModel(primitiveObjects, model);
		
		glUseProgram(0);
	}

	void cleanup() {
		glDeleteProgram(programID);
	}
};


// ====================
// SIMPLE GROUND
// ====================
struct SimpleGround {
    GLuint vao, vbo, uvVbo, ebo;
    GLuint shader;
    GLuint textureID;
    GLuint mvpMatrixID;
    GLuint textureSamplerID;
    
    void initialize() {
        std::cout << "Initializing textured ground..." << std::endl;
        
        // Ground vertices (two triangles forming a quad)
        float vertices[] = {
            // positions
            -1000.0f, 0.0f, -1000.0f,
             1000.0f, 0.0f, -1000.0f,
             1000.0f, 0.0f,  1000.0f,
            -1000.0f, 0.0f,  1000.0f
        };
        
        // Texture coordinates (with tiling)
        float uvCoords[] = {
            0.0f, 0.0f,
            20.0f, 0.0f,  // Tiled 20 times
            20.0f, 20.0f,
            0.0f, 20.0f
        };
        
        // Indices for two triangles
        unsigned int indices[] = {
            0, 1, 2,
            0, 2, 3
        };
        
        // Create and bind VAO
        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);
        
        // Vertex buffer
        glGenBuffers(1, &vbo);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(0);
        
        // UV buffer
        glGenBuffers(1, &uvVbo);
        glBindBuffer(GL_ARRAY_BUFFER, uvVbo);
        glBufferData(GL_ARRAY_BUFFER, sizeof(uvCoords), uvCoords, GL_STATIC_DRAW);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(1);
        
        // Index buffer
        glGenBuffers(1, &ebo);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
        
        glBindVertexArray(0);
        
        // Create shader for textured ground
        const char* groundVert = R"(
            #version 330 core
            layout(location = 0) in vec3 aPos;
            layout(location = 1) in vec2 aTexCoord;
            
            out vec2 TexCoord;
            uniform mat4 MVP;
            
            void main() {
                gl_Position = MVP * vec4(aPos, 1.0);
                TexCoord = aTexCoord;
            }
        )";
        
        const char* groundFrag = R"(
            #version 330 core
            in vec2 TexCoord;
            out vec3 FragColor;
            
            uniform sampler2D groundTexture;
            
            void main() {
                FragColor = texture(groundTexture, TexCoord).rgb;
                // Add some variation to make it look more natural
                FragColor *= vec3(0.8, 1.0, 0.8); // Slightly greener tint
            }
        )";
        
        // Compile shaders
        GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
        glShaderSource(vertexShader, 1, &groundVert, NULL);
        glCompileShader(vertexShader);
        
        GLint success;
        char infoLog[512];
        glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
        if (!success) {
            glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
            std::cout << "Ground Vertex shader compilation failed: " << infoLog << std::endl;
        }
        
        GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
        glShaderSource(fragmentShader, 1, &groundFrag, NULL);
        glCompileShader(fragmentShader);
        
        glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
        if (!success) {
            glGetShaderInfoLog(fragmentShader, 512, NULL, infoLog);
            std::cout << "Ground Fragment shader compilation failed: " << infoLog << std::endl;
        }
        
        // Link shader program
        shader = glCreateProgram();
        glAttachShader(shader, vertexShader);
        glAttachShader(shader, fragmentShader);
        glLinkProgram(shader);
        
        glGetProgramiv(shader, GL_LINK_STATUS, &success);
        if (!success) {
            glGetProgramInfoLog(shader, 512, NULL, infoLog);
            std::cout << "Ground Shader program linking failed: " << infoLog << std::endl;
        }
        
        glDeleteShader(vertexShader);
        glDeleteShader(fragmentShader);
        
        // Get uniform locations
        mvpMatrixID = glGetUniformLocation(shader, "MVP");
        textureSamplerID = glGetUniformLocation(shader, "groundTexture");
        
        // Load ground texture
        textureID = loadGroundTexture();
        
        std::cout << "Textured ground initialized. Shader ID: " << shader 
                  << ", Texture ID: " << textureID << std::endl;
    }
    
    GLuint loadGroundTexture() {
        // Try to load grass texture
        const char* texturePath = "../assignment1/model/ground/ground.jpg";  // Adjust path as needed
        
        int width, height, channels;
        unsigned char* data = stbi_load(texturePath, &width, &height, &channels, 0);
        
        GLuint texture;
        glGenTextures(1, &texture);
        glBindTexture(GL_TEXTURE_2D, texture);
        
        if (data) {
            std::cout << "Loaded grass texture: " << texturePath 
                      << " (" << width << "x" << height << ", channels: " << channels << ")" << std::endl;
            
            GLenum format = GL_RGB;
            if (channels == 4) format = GL_RGBA;
            else if (channels == 1) format = GL_RED;
            
            glTexImage2D(GL_TEXTURE_2D, 0, format, width, height, 0, format, GL_UNSIGNED_BYTE, data);
            glGenerateMipmap(GL_TEXTURE_2D);
            
            // Set texture parameters for tiling
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            
            stbi_image_free(data);
        } else {
            std::cout << "Failed to load grass texture. Creating procedural grass texture." << std::endl;
            createProceduralGrassTexture(texture);
        }
        
        return texture;
    }
    
    void createProceduralGrassTexture(GLuint texture) {
        const int width = 256;
        const int height = 256;
        std::vector<unsigned char> pixels(width * height * 3);
        
        // Create a procedural grass pattern
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int idx = (y * width + x) * 3;
                
                // Base grass color with variation
                float noise = 0.2f * ((x * 17 + y * 23) % 100) / 100.0f;
                
                pixels[idx] = static_cast<unsigned char>(40 + 50 * noise);      // R
                pixels[idx + 1] = static_cast<unsigned char>(160 + 40 * noise);  // G
                pixels[idx + 2] = static_cast<unsigned char>(40 + 30 * noise);   // B
                
                // Add some darker patches for variation
                if (((x / 16) + (y / 16)) % 4 == 0) {
                    pixels[idx] = static_cast<unsigned char>(pixels[idx] * 0.8f);
                    pixels[idx + 1] = static_cast<unsigned char>(pixels[idx + 1] * 0.8f);
                }
            }
        }
        
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, pixels.data());
        glGenerateMipmap(GL_TEXTURE_2D);
        
        // Set texture parameters
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        
        std::cout << "Created procedural grass texture." << std::endl;
    }
    
    void render(const glm::mat4& viewProj) {
        if (!renderGround) return;
        
        glUseProgram(shader);
        glBindVertexArray(vao);
        
        // Set MVP matrix
        glm::mat4 mvp = viewProj; // Identity model matrix
        glUniformMatrix4fv(mvpMatrixID, 1, GL_FALSE, glm::value_ptr(mvp));
        
        // Bind texture
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, textureID);
        glUniform1i(textureSamplerID, 0);
        
        // Draw ground
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
        
        glBindVertexArray(0);
    }
    
    void cleanup() {
        glDeleteVertexArrays(1, &vao);
        glDeleteBuffers(1, &vbo);
        glDeleteBuffers(1, &uvVbo);
        glDeleteBuffers(1, &ebo);
        glDeleteTextures(1, &textureID);
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
    MyBot bot;
    
    std::cout << "=== Initializing Test Scene ===" << std::endl;
    testCube.initialize();
    ground.initialize();
    bot.initialize();


    
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
	float time = 0.0f;			// Animation time 
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

        if (playAnimation) {
			time += deltaTime * playbackSpeed;
			bot.update(time);
		}

        // Camera matrices
        glm::mat4 projection = glm::perspective(glm::radians(FoV), 
                                               (float)windowWidth / windowHeight, 
                                               zNear, zFar);
        glm::mat4 view = glm::lookAt(cameraPos, cameraPos + cameraFront, cameraUp);
        glm::mat4 viewProj = projection * view;

        // Render test objects
        bot.render(viewProj);
        ground.render(viewProj);
        testCube.render(viewProj);
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
	bot.cleanup();
    testCube.cleanup();
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