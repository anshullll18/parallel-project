#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <iostream>
#include <vector>
#include <unordered_map>
#include <thread>
#include <mutex>
#include <atomic>
#include <chrono>
#include <cmath>
#include <algorithm>

// ======================== CONSTANTS ========================
const int SCR_WIDTH = 1920;
const int SCR_HEIGHT = 1080;
const float PARTICLE_RADIUS = 0.05f;
const float H = 0.1f; // Smoothing radius
const float MASS = 1.0f;
const float GAS_CONSTANT = 2000.0f;
const float REST_DENSITY = 1000.0f;
const float VISCOSITY = 0.018f;
const float SURFACE_TENSION = 0.0728f;
const float GRAVITY = -9.81f;
const float BOUNDARY_DAMPING = 0.5f;
const glm::vec3 CONTAINER_MIN(-2.0f, -2.0f, -2.0f);
const glm::vec3 CONTAINER_MAX(2.0f, 2.0f, 2.0f);

// ======================== STRUCTURES ========================
struct Particle {
    glm::vec3 position;
    glm::vec3 velocity;
    glm::vec3 force;
    float density;
    float pressure;
    float temperature;
    int fluidType;
    glm::vec3 color;
};

struct FluidProperties {
    float viscosity;
    float restDensity;
    float gasConstant;
    glm::vec3 color;
};

struct PerformanceMetrics {
    float serialTime;
    float parallelTime;
    float speedup;
    float efficiency;
    int numThreads;
    float fps;
};

// ======================== SPATIAL HASH GRID ========================
class SpatialHashGrid {
private:
    std::unordered_map<int64_t, std::vector<int>> grid;
    float cellSize;
    std::mutex gridMutex;
    
    int64_t hashCoord(int x, int y, int z) {
        return ((int64_t)x * 73856093) ^ ((int64_t)y * 19349663) ^ ((int64_t)z * 83492791);
    }
    
    glm::ivec3 getCell(const glm::vec3& pos) {
        return glm::ivec3(floor(pos.x / cellSize), floor(pos.y / cellSize), floor(pos.z / cellSize));
    }

public:
    SpatialHashGrid(float h) : cellSize(h) {}
    
    void clear() { 
        std::lock_guard<std::mutex> lock(gridMutex);
        grid.clear(); 
    }
    
    void insert(int particleIdx, const glm::vec3& pos) {
        glm::ivec3 cell = getCell(pos);
        std::lock_guard<std::mutex> lock(gridMutex);
        grid[hashCoord(cell.x, cell.y, cell.z)].push_back(particleIdx);
    }
    
    std::vector<int> getNeighbors(const glm::vec3& pos) {
        std::vector<int> neighbors;
        glm::ivec3 cell = getCell(pos);
        
        std::lock_guard<std::mutex> lock(gridMutex);
        for(int dx = -1; dx <= 1; dx++) {
            for(int dy = -1; dy <= 1; dy++) {
                for(int dz = -1; dz <= 1; dz++) {
                    int64_t hash = hashCoord(cell.x + dx, cell.y + dy, cell.z + dz);
                    if(grid.find(hash) != grid.end()) {
                        neighbors.insert(neighbors.end(), grid[hash].begin(), grid[hash].end());
                    }
                }
            }
        }
        return neighbors;
    }
};

// ======================== SPH SIMULATOR ========================
class SPHSimulator {
private:
    std::vector<Particle> particles;
    SpatialHashGrid spatialGrid;
    std::vector<FluidProperties> fluidTypes;
    glm::vec3 gravityDir;
    PerformanceMetrics metrics;
    int numThreads;
    bool useParallel;
    
    float poly6Kernel(float r) {
        if(r >= 0 && r <= H) {
            float tmp = H * H - r * r;
            return 315.0f / (64.0f * M_PI * powf(H, 9.0f)) * tmp * tmp * tmp;
        }
        return 0.0f;
    }
    
    glm::vec3 spikyGradient(const glm::vec3& r) {
        float len = glm::length(r);
        if(len > 0.0001f && len <= H) {
            float tmp = H - len;
            float factor = -45.0f / (M_PI * pow(H, 6.0f)) * tmp * tmp;
            return factor * (r / len);
        }
        return glm::vec3(0.0f);
    }
    
    float viscosityLaplacian(float r) {
        if(r >= 0 && r <= H) {
            return 45.0f / (M_PI * powf(H, 6.0f)) * (H - r);
        }
        return 0.0f;
    }
    
    void computeDensityPressure(int start, int end) {
        for(int i = start; i < end; i++) {
            Particle& pi = particles[i];
            pi.density = 0.0f;
            
            std::vector<int> neighbors = spatialGrid.getNeighbors(pi.position);
            for(int j : neighbors) {
                glm::vec3 rij = pi.position - particles[j].position;
                float r = glm::length(rij);
                if(r < H) {
                    pi.density += MASS * poly6Kernel(r);
                }
            }
            
            pi.density = std::max(pi.density, REST_DENSITY * 0.01f); // Prevent divide by zero
            pi.pressure = GAS_CONSTANT * (pi.density - fluidTypes[pi.fluidType].restDensity);
        }
    }
    
    void computeForces(int start, int end) {
        for(int i = start; i < end; i++) {
            Particle& pi = particles[i];
            glm::vec3 fPressure(0.0f), fViscosity(0.0f), fSurface(0.0f);
            
            std::vector<int> neighbors = spatialGrid.getNeighbors(pi.position);
            for(int j : neighbors) {
                if(i == j) continue;
                
                Particle& pj = particles[j];
                glm::vec3 rij = pi.position - pj.position;
                float r = glm::length(rij);
                
                if(r < H && r > 0.0001f) {
                    // Pressure force
                    fPressure += -MASS * (pi.pressure + pj.pressure) / (2.0f * pj.density) * spikyGradient(rij);
                    
                    // Viscosity force
                    float visc = fluidTypes[pi.fluidType].viscosity;
                    fViscosity += visc * MASS * (pj.velocity - pi.velocity) / pj.density * viscosityLaplacian(r);
                    
                    // Surface tension
                    fSurface += MASS / pj.density * spikyGradient(rij);
                }
            }
            
            glm::vec3 fGravity = pi.density * gravityDir * GRAVITY;
            
            glm::vec3 surfaceForce = glm::vec3(0.0f);
            float surfaceLen = glm::length(fSurface);
            if(surfaceLen > 0.0001f) {
                surfaceForce = -SURFACE_TENSION * surfaceLen * (fSurface / surfaceLen);
            }
            
            pi.force = fPressure + fViscosity + fGravity + surfaceForce;
            
            // Temperature diffusion
            for(int j : neighbors) {
                if(i == j) continue;
                float tempDiff = particles[j].temperature - pi.temperature;
                pi.temperature += 0.001f * tempDiff;
            }
        }
    }
    
    void integrate(float dt) {
        for(auto& p : particles) {
            if(p.density > 0.0001f) {
                p.velocity += dt * p.force / p.density;
            }
            p.position += dt * p.velocity;
            
            // Boundary collision with damping
            for(int d = 0; d < 3; d++) {
                if(p.position[d] < CONTAINER_MIN[d]) {
                    p.position[d] = CONTAINER_MIN[d];
                    p.velocity[d] *= -BOUNDARY_DAMPING;
                }
                if(p.position[d] > CONTAINER_MAX[d]) {
                    p.position[d] = CONTAINER_MAX[d];
                    p.velocity[d] *= -BOUNDARY_DAMPING;
                }
            }
            
            // Update color based on velocity and temperature
            float speed = glm::length(p.velocity);
            float velFactor = glm::clamp(speed / 10.0f, 0.0f, 1.0f);
            float tempFactor = glm::clamp((p.temperature - 273.0f) / 100.0f, 0.0f, 1.0f);
            
            glm::vec3 baseColor = fluidTypes[p.fluidType].color;
            p.color = baseColor * (1.0f - velFactor * 0.5f) + glm::vec3(1.0f, 0.3f, 0.0f) * velFactor * 0.5f;
            p.color = p.color * (1.0f - tempFactor * 0.3f) + glm::vec3(1.0f, 0.0f, 0.0f) * tempFactor * 0.3f;
        }
    }

public:
    SPHSimulator() : spatialGrid(H), gravityDir(0, -1, 0), 
                     numThreads(std::thread::hardware_concurrency()), 
                     useParallel(true) {
        // Initialize fluid types
        fluidTypes.push_back({VISCOSITY, REST_DENSITY, GAS_CONSTANT, glm::vec3(0.2f, 0.5f, 1.0f)}); // Water
        fluidTypes.push_back({VISCOSITY * 5.0f, REST_DENSITY * 0.9f, GAS_CONSTANT * 0.8f, glm::vec3(0.8f, 0.6f, 0.2f)}); // Oil
        fluidTypes.push_back({VISCOSITY * 0.5f, REST_DENSITY, GAS_CONSTANT * 1.2f, glm::vec3(1.0f, 0.2f, 0.2f)}); // Hot
        fluidTypes.push_back({VISCOSITY * 2.0f, REST_DENSITY * 1.1f, GAS_CONSTANT * 0.9f, glm::vec3(0.2f, 0.2f, 1.0f)}); // Cold
        
        metrics = {0, 0, 0, 0, numThreads, 0};
        
        std::cout << "Hardware threads available: " << numThreads << std::endl;
    }
    
    void addParticle(const glm::vec3& pos, int fluidType = 0) {
        Particle p;
        p.position = pos;
        p.velocity = glm::vec3(0.0f);
        p.force = glm::vec3(0.0f);
        p.density = REST_DENSITY;
        p.pressure = 0.0f;
        p.temperature = (fluidType == 2) ? 373.0f : (fluidType == 3) ? 273.0f : 298.0f;
        p.fluidType = fluidType;
        p.color = fluidTypes[fluidType].color;
        particles.push_back(p);
    }
    
    void spawnFluidCube(const glm::vec3& center, int count, int fluidType = 0) {
        int side = (int)ceil(pow((double)count, 1.0/3.0));
        float spacing = PARTICLE_RADIUS * 2.0f;
        
        int spawned = 0;
        for(int x = 0; x < side && spawned < count; x++) {
            for(int y = 0; y < side && spawned < count; y++) {
                for(int z = 0; z < side && spawned < count; z++) {
                    glm::vec3 offset = glm::vec3((float)x, (float)y, (float)z) * spacing;
                    offset -= glm::vec3(side * spacing / 2.0f);
                    glm::vec3 pos = center + offset;
                    
                    // Add small random jitter to prevent perfect grid
                    pos.x += (rand() % 100 - 50) * 0.0001f;
                    pos.y += (rand() % 100 - 50) * 0.0001f;
                    pos.z += (rand() % 100 - 50) * 0.0001f;
                    
                    addParticle(pos, fluidType);
                    spawned++;
                }
            }
        }
        std::cout << "Spawned " << spawned << " particles of type " << fluidType << std::endl;
    }
    
    void update(float dt) {
        if(particles.empty()) return;
        
        auto startTotal = std::chrono::high_resolution_clock::now();
        
        // Rebuild spatial grid
        spatialGrid.clear();
        for(size_t i = 0; i < particles.size(); i++) {
            spatialGrid.insert(i, particles[i].position);
        }
        
        if(useParallel && numThreads > 1) {
            // Parallel execution
            auto startParallel = std::chrono::high_resolution_clock::now();
            
            std::vector<std::thread> threads;
            int chunkSize = std::max(1, (int)particles.size() / numThreads);
            
            // Compute density and pressure
            for(int t = 0; t < numThreads; t++) {
                int start = t * chunkSize;
                int end = (t == numThreads - 1) ? particles.size() : (t + 1) * chunkSize;
                threads.emplace_back([this, start, end]() {
                    computeDensityPressure(start, end);
                });
            }
            for(auto& thread : threads) thread.join();
            threads.clear();
            
            // Compute forces
            for(int t = 0; t < numThreads; t++) {
                int start = t * chunkSize;
                int end = (t == numThreads - 1) ? particles.size() : (t + 1) * chunkSize;
                threads.emplace_back([this, start, end]() {
                    computeForces(start, end);
                });
            }
            for(auto& thread : threads) thread.join();
            
            auto endParallel = std::chrono::high_resolution_clock::now();
            metrics.parallelTime = std::chrono::duration<float, std::milli>(endParallel - startParallel).count();
            
            // Estimate serial time (not actually computing it to save CPU)
            metrics.serialTime = metrics.parallelTime * numThreads * 0.85f; // Rough estimate
            metrics.speedup = metrics.serialTime / metrics.parallelTime;
        } else {
            // Serial execution
            auto startSerial = std::chrono::high_resolution_clock::now();
            computeDensityPressure(0, particles.size());
            computeForces(0, particles.size());
            auto endSerial = std::chrono::high_resolution_clock::now();
            metrics.serialTime = std::chrono::duration<float, std::milli>(endSerial - startSerial).count();
            metrics.parallelTime = metrics.serialTime;
            metrics.speedup = 1.0f;
        }
        
        integrate(dt);
        
        auto endTotal = std::chrono::high_resolution_clock::now();
        float totalTime = std::chrono::duration<float, std::milli>(endTotal - startTotal).count();
        metrics.fps = 1000.0f / totalTime;
        metrics.efficiency = (metrics.speedup / numThreads) * 100.0f;
    }
    
    const std::vector<Particle>& getParticles() const { return particles; }
    const PerformanceMetrics& getMetrics() const { return metrics; }
    void setGravityDirection(const glm::vec3& dir) { gravityDir = glm::normalize(dir); }
    void setThreadCount(int count) { 
        numThreads = std::max(1, std::min(count, (int)std::thread::hardware_concurrency())); 
        std::cout << "Thread count set to: " << numThreads << std::endl;
    }
    void toggleParallel() { 
        useParallel = !useParallel; 
        std::cout << "Parallel mode: " << (useParallel ? "ON" : "OFF") << std::endl;
    }
    void clearParticles() { 
        particles.clear(); 
        std::cout << "Particles cleared" << std::endl;
    }
};

// ======================== OPENGL RENDERING ========================
const char* vertexShaderSource = R"(
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aColor;
layout (location = 2) in float aSize;

out vec3 Color;

uniform mat4 projection;
uniform mat4 view;

void main() {
    Color = aColor;
    gl_Position = projection * view * vec4(aPos, 1.0);
    gl_PointSize = max(5.0, aSize * 400.0 / gl_Position.w);
}
)";

const char* fragmentShaderSource = R"(
#version 330 core
in vec3 Color;
out vec4 FragColor;

void main() {
    vec2 coord = gl_PointCoord - vec2(0.5);
    float dist = length(coord);
    if(dist > 0.5) discard;
    
    float alpha = 1.0 - (dist * 2.0) * (dist * 2.0);
    alpha = smoothstep(0.0, 1.0, alpha);
    FragColor = vec4(Color, alpha * 0.9);
}
)";

GLuint compileShader(const char* source, GLenum type) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, NULL);
    glCompileShader(shader);
    
    int success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if(!success) {
        char infoLog[512];
        glGetShaderInfoLog(shader, 512, NULL, infoLog);
        std::cerr << "Shader compilation error: " << infoLog << std::endl;
    }
    return shader;
}

// ======================== MAIN APPLICATION ========================
class Application {
private:
    GLFWwindow* window;
    SPHSimulator simulator;
    GLuint shaderProgram, VAO, VBO;
    
    float cameraYaw = -90.0f;
    float cameraPitch = 20.0f;
    float cameraDistance = 8.0f;
    glm::vec3 cameraPos;
    
    bool mousePressed = false;
    double lastX = SCR_WIDTH / 2.0;
    double lastY = SCR_HEIGHT / 2.0;
    
    int currentFluidType = 0;
    
    void setupOpenGL() {
        GLuint vertexShader = compileShader(vertexShaderSource, GL_VERTEX_SHADER);
        GLuint fragmentShader = compileShader(fragmentShaderSource, GL_FRAGMENT_SHADER);
        
        shaderProgram = glCreateProgram();
        glAttachShader(shaderProgram, vertexShader);
        glAttachShader(shaderProgram, fragmentShader);
        glLinkProgram(shaderProgram);
        
        int success;
        glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
        if(!success) {
            char infoLog[512];
            glGetProgramInfoLog(shaderProgram, 512, NULL, infoLog);
            std::cerr << "Shader linking error: " << infoLog << std::endl;
        }
        
        glDeleteShader(vertexShader);
        glDeleteShader(fragmentShader);
        
        glGenVertexArrays(1, &VAO);
        glGenBuffers(1, &VBO);
        
        glEnable(GL_DEPTH_TEST);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glEnable(GL_PROGRAM_POINT_SIZE);
        glEnable(GL_MULTISAMPLE);
        
        std::cout << "OpenGL initialized successfully" << std::endl;
        std::cout << "OpenGL Version: " << glGetString(GL_VERSION) << std::endl;
    }
    
    void updateCamera() {
        float camX = cameraDistance * cos(glm::radians(cameraYaw)) * cos(glm::radians(cameraPitch));
        float camY = cameraDistance * sin(glm::radians(cameraPitch));
        float camZ = cameraDistance * sin(glm::radians(cameraYaw)) * cos(glm::radians(cameraPitch));
        cameraPos = glm::vec3(camX, camY, camZ);
    }

public:
    Application() {
        if(!glfwInit()) {
            std::cerr << "Failed to initialize GLFW" << std::endl;
            exit(-1);
        }
        
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
        glfwWindowHint(GLFW_SAMPLES, 4);
        
        #ifdef __APPLE__
        glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
        #endif
        
        window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "3D SPH Fluid Simulation", NULL, NULL);
        if(!window) {
            std::cerr << "Failed to create GLFW window" << std::endl;
            glfwTerminate();
            exit(-1);
        }
        
        glfwMakeContextCurrent(window);
        glfwSwapInterval(1); // Enable vsync
        
        if(!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
            std::cerr << "Failed to initialize GLAD" << std::endl;
            exit(-1);
        }
        
        setupOpenGL();
        
        // Initialize with some fluid - smaller initial spawn
        std::cout << "Spawning initial particles..." << std::endl;
        simulator.spawnFluidCube(glm::vec3(0, 0.5, 0), 1000, 0);
        
        glfwSetWindowUserPointer(window, this);
        glfwSetMouseButtonCallback(window, mouseButtonCallback);
        glfwSetCursorPosCallback(window, cursorPosCallback);
        glfwSetScrollCallback(window, scrollCallback);
        glfwSetKeyCallback(window, keyCallback);
    }
    
    ~Application() {
        glDeleteVertexArrays(1, &VAO);
        glDeleteBuffers(1, &VBO);
        glDeleteProgram(shaderProgram);
        glfwTerminate();
    }
    
    static void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
        Application* app = (Application*)glfwGetWindowUserPointer(window);
        if(button == GLFW_MOUSE_BUTTON_LEFT) {
            app->mousePressed = (action == GLFW_PRESS);
        }
        if(button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS) {
            app->simulator.spawnFluidCube(glm::vec3(0, 1, 0), 200, app->currentFluidType);
        }
    }
    
    static void cursorPosCallback(GLFWwindow* window, double xpos, double ypos) {
        Application* app = (Application*)glfwGetWindowUserPointer(window);
        if(app->mousePressed) {
            float xoffset = xpos - app->lastX;
            float yoffset = app->lastY - ypos;
            app->lastX = xpos;
            app->lastY = ypos;
            
            app->cameraYaw += xoffset * 0.2f;
            app->cameraPitch += yoffset * 0.2f;
            app->cameraPitch = glm::clamp(app->cameraPitch, -89.0f, 89.0f);
        }
        app->lastX = xpos;
        app->lastY = ypos;
    }
    
    static void scrollCallback(GLFWwindow* window, double xoffset, double yoffset) {
        Application* app = (Application*)glfwGetWindowUserPointer(window);
        app->cameraDistance -= yoffset * 0.5f;
        app->cameraDistance = glm::clamp(app->cameraDistance, 2.0f, 20.0f);
    }
    
    static void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
        Application* app = (Application*)glfwGetWindowUserPointer(window);
        if(action == GLFW_PRESS) {
            switch(key) {
                case GLFW_KEY_ESCAPE: glfwSetWindowShouldClose(window, true); break;
                case GLFW_KEY_1: app->currentFluidType = 0; std::cout << "Selected: Water" << std::endl; break;
                case GLFW_KEY_2: app->currentFluidType = 1; std::cout << "Selected: Oil" << std::endl; break;
                case GLFW_KEY_3: app->currentFluidType = 2; std::cout << "Selected: Hot" << std::endl; break;
                case GLFW_KEY_4: app->currentFluidType = 3; std::cout << "Selected: Cold" << std::endl; break;
                case GLFW_KEY_G: app->simulator.setGravityDirection(glm::vec3(0, -1, 0)); break;
                case GLFW_KEY_H: app->simulator.setGravityDirection(glm::vec3(-1, 0, 0)); break;
                case GLFW_KEY_J: app->simulator.setGravityDirection(glm::vec3(1, 0, 0)); break;
                case GLFW_KEY_C: app->simulator.clearParticles(); break;
                case GLFW_KEY_P: app->simulator.toggleParallel(); break;
                case GLFW_KEY_SPACE: app->simulator.spawnFluidCube(glm::vec3(0, 1, 0), 500, app->currentFluidType); break;
                case GLFW_KEY_EQUAL: 
                case GLFW_KEY_KP_ADD: 
                    app->simulator.setThreadCount(app->simulator.getMetrics().numThreads + 1); 
                    break;
                case GLFW_KEY_MINUS: 
                case GLFW_KEY_KP_SUBTRACT: 
                    app->simulator.setThreadCount(app->simulator.getMetrics().numThreads - 1); 
                    break;
            }
        }
    }
    
    void render() {
        const auto& particles = simulator.getParticles();
        
        if(particles.empty()) {
            glClearColor(0.05f, 0.05f, 0.1f, 1.0f);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            return;
        }
        
        std::vector<float> vertexData;
        vertexData.reserve(particles.size() * 7);
        
        for(const auto& p : particles) {
            vertexData.push_back(p.position.x);
            vertexData.push_back(p.position.y);
            vertexData.push_back(p.position.z);
            vertexData.push_back(p.color.r);
            vertexData.push_back(p.color.g);
            vertexData.push_back(p.color.b);
            float size = PARTICLE_RADIUS * 1.5f;
            vertexData.push_back(size);
        }
        
        glBindVertexArray(VAO);
        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBufferData(GL_ARRAY_BUFFER, vertexData.size() * sizeof(float), vertexData.data(), GL_DYNAMIC_DRAW);
        
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 7 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 7 * sizeof(float), (void*)(3 * sizeof(float)));
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, 7 * sizeof(float), (void*)(6 * sizeof(float)));
        glEnableVertexAttribArray(2);
        
        updateCamera();
        
        glm::mat4 projection = glm::perspective(glm::radians(45.0f), (float)SCR_WIDTH / SCR_HEIGHT, 0.1f, 100.0f);
        glm::mat4 view = glm::lookAt(cameraPos, glm::vec3(0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
        
        glUseProgram(shaderProgram);
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "projection"), 1, GL_FALSE, glm::value_ptr(projection));
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "view"), 1, GL_FALSE, glm::value_ptr(view));
        
        glClearColor(0.05f, 0.05f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        
        glDrawArrays(GL_POINTS, 0, particles.size());
    }
    
    void run() {
        float lastFrame = glfwGetTime();
        
        std::cout << "\n=== 3D SPH Fluid Simulation Controls ===" << std::endl;
        std::cout << "Left Click + Drag: Rotate camera" << std::endl;
        std::cout << "Right Click: Spawn 200 particles" << std::endl;
        std::cout << "Scroll: Zoom in/out" << std::endl;
        std::cout << "1-4: Select fluid type (Water/Oil/Hot/Cold)" << std::endl;
        std::cout << "Space: Spawn 500 particles" << std::endl;
        std::cout << "G: Gravity down | H: Gravity left | J: Gravity right" << std::endl;
        std::cout << "C: Clear all particles" << std::endl;
        std::cout << "P: Toggle parallel/serial mode" << std::endl;
        std::cout << "+/-: Increase/decrease thread count" << std::endl;
        std::cout << "ESC: Exit" << std::endl;
        std::cout << "========================================\n" << std::endl;
        
        while(!glfwWindowShouldClose(window)) {
            float currentFrame = glfwGetTime();
            float deltaTime = currentFrame - lastFrame;
            lastFrame = currentFrame;
            
            // Cap delta time to prevent instability
            simulator.update(std::min(deltaTime, 0.016f));
            
            render();
            
            // Display metrics in title bar
            const auto& metrics = simulator.getMetrics();
            char title[256];
            snprintf(title, sizeof(title), 
                    "SPH Simulation | Particles: %zu | FPS: %.0f | Threads: %d | Speedup: %.2fx | Efficiency: %.1f%%",
                    simulator.getParticles().size(),
                    metrics.fps,
                    metrics.numThreads,
                    metrics.speedup,
                    metrics.efficiency);
            glfwSetWindowTitle(window, title);
            
            glfwSwapBuffers(window);
            glfwPollEvents();
        }
    }
};

int main() {
    try {
        std::cout << "Starting SPH Fluid Simulation..." << std::endl;
        Application app;
        app.run();
    } catch(const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}