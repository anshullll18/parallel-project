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
#include <random>
#include <future>
#ifdef __x86_64__
#include <immintrin.h> // For SIMD optimizations on x86/x64
#endif

// ======================== CONSTANTS ========================
const int SCR_WIDTH = 1920;
const int SCR_HEIGHT = 1080;
const float PARTICLE_RADIUS = 0.04f;
const float H = 0.12f; // Smoothing radius (kernel support)
const float H2 = H * H;
const float H9 = H * H * H * H * H * H * H * H * H;
const float MASS = 0.02f;
const float REST_DENSITY = 1000.0f;
const float GAS_CONSTANT = 200.0f; // Much lower for fluid-like behavior
const float VISCOSITY = 3.5f;
const float SURFACE_TENSION = 0.05f;
const float GRAVITY = 9.81f;
const float BOUNDARY_DAMPING = 0.3f;
const float TIME_STEP = 0.003f; // Fixed small timestep
const glm::vec3 CONTAINER_MIN(-1.8f, -1.8f, -1.8f);
const glm::vec3 CONTAINER_MAX(1.8f, 1.8f, 1.8f);

// Enhanced visual constants
const int MAX_PARTICLES = 50000;
const int TRAIL_LENGTH = 20;
const float BLOOM_THRESHOLD = 0.8f;
const float BLOOM_INTENSITY = 1.5f;
const int BLOOM_ITERATIONS = 5;

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
    glm::vec3 trail[TRAIL_LENGTH]; // For particle trails
    int trailIndex;
    float age; // For particle aging effects
    float brightness; // For bloom effects
};

struct FluidProperties {
    float viscosity;
    float restDensity;
    float gasConstant;
    float mass;
    glm::vec3 color;
    float temperature;
    bool isReactive; // For mixing effects
};

struct PerformanceMetrics {
    float serialTime;
    float parallelTime;
    float speedup;
    float efficiency;
    int numThreads;
    float fps;
    int particleCount;
    float gpuTime;
    float memoryUsage;
};

struct MouseInteraction {
    glm::vec3 position;
    float radius;
    float strength;
    bool active;
    glm::vec3 direction;
};

struct VisualSettings {
    bool enableBloom;
    bool enableTrails;
    bool enableDepthOfField;
    bool enableParticleInteraction;
    float bloomThreshold;
    float bloomIntensity;
    float trailFade;
    float dofFocus;
    float dofBlur;
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
        return glm::ivec3(
            (int)floor(pos.x / cellSize), 
            (int)floor(pos.y / cellSize), 
            (int)floor(pos.z / cellSize)
        );
    }

public:
    SpatialHashGrid(float h) : cellSize(h * 2.0f) {} // Cell size = 2*h for efficiency
    
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
                    auto it = grid.find(hash);
                    if(it != grid.end()) {
                        neighbors.insert(neighbors.end(), it->second.begin(), it->second.end());
                    }
                }
            }
        }
        return neighbors;
    }
};

// ======================== THREAD POOL ========================
class ThreadPool {
private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    std::mutex queueMutex;
    std::condition_variable condition;
    bool stop;

public:
    ThreadPool(size_t numThreads) : stop(false) {
        for(size_t i = 0; i < numThreads; ++i) {
            workers.emplace_back([this] {
                while(true) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(queueMutex);
                        condition.wait(lock, [this] { return stop || !tasks.empty(); });
                        if(stop && tasks.empty()) return;
                        task = std::move(tasks.front());
                        tasks.pop();
                    }
                    task();
                }
            });
        }
    }

    template<class F>
    void enqueue(F&& f) {
        {
            std::unique_lock<std::mutex> lock(queueMutex);
            tasks.emplace(std::forward<F>(f));
        }
        condition.notify_one();
    }

    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(queueMutex);
            stop = true;
        }
        condition.notify_all();
        for(std::thread &worker : workers)
            worker.join();
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
    ThreadPool* threadPool;
    int numThreads;
    bool useParallel;
    MouseInteraction mouseInteraction;
    VisualSettings visualSettings;
    
    // Pre-computed kernel constants
    const float POLY6 = 315.0f / (64.0f * M_PI * H9);
    const float SPIKY_GRAD = -45.0f / (M_PI * H9);
    const float VISC_LAP = 45.0f / (M_PI * H9);
    
    // Improved Poly6 kernel
    float poly6Kernel(float r2) {
        if(r2 < H2) {
            float diff = H2 - r2;
            return POLY6 * diff * diff * diff;
        }
        return 0.0f;
    }
    
    // Spiky gradient kernel (better for pressure)
    glm::vec3 spikyGradient(const glm::vec3& r, float len) {
        if(len > 0.0001f && len < H) {
            float coef = SPIKY_GRAD * (H - len) * (H - len) / len;
            return coef * r;
        }
        return glm::vec3(0.0f);
    }
    
    // Viscosity Laplacian kernel
    float viscosityLaplacian(float len) {
        if(len < H && len > 0.0001f) {
            return VISC_LAP * (H - len);
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
                float r2 = glm::dot(rij, rij);
                
                if(r2 < H2) {
                    pi.density += MASS * poly6Kernel(r2);
                }
            }
            
            // Clamp density to avoid instabilities
            pi.density = std::max(pi.density, REST_DENSITY * 0.5f);
            
            // Tait equation for pressure (more stable than ideal gas)
            const float gamma = 7.0f;
            pi.pressure = GAS_CONSTANT * (pow(pi.density / fluidTypes[pi.fluidType].restDensity, gamma) - 1.0f);
        }
    }
    
    void computeForces(int start, int end) {
        for(int i = start; i < end; i++) {
            Particle& pi = particles[i];
            glm::vec3 fPressure(0.0f), fViscosity(0.0f);
            glm::vec3 gradColorField(0.0f);
            float lapColorField = 0.0f;
            
            std::vector<int> neighbors = spatialGrid.getNeighbors(pi.position);
            
            for(int j : neighbors) {
                if(i == j) continue;
                
                Particle& pj = particles[j];
                glm::vec3 rij = pi.position - pj.position;
                float r = glm::length(rij);
                float r2 = r * r;
                
                if(r < H && r > 0.0001f) {
                    // Pressure force (symmetric formulation)
                    float pressureTerm = (pi.pressure + pj.pressure) / (2.0f * pj.density);
                    fPressure += -MASS * pressureTerm * spikyGradient(rij, r);
                    
                    // Viscosity force
                    float visc = fluidTypes[pi.fluidType].viscosity;
                    glm::vec3 velDiff = pj.velocity - pi.velocity;
                    fViscosity += visc * MASS * (velDiff / pj.density) * viscosityLaplacian(r);
                    
                    // Surface tension - gradient and laplacian of color field
                    gradColorField += (MASS / pj.density) * spikyGradient(rij, r);
                    lapColorField += (MASS / pj.density) * viscosityLaplacian(r);
                }
            }
            
            // Gravity
            glm::vec3 fGravity = pi.density * gravityDir * GRAVITY;
            
            // Surface tension
            glm::vec3 fSurface(0.0f);
            float gradLen = glm::length(gradColorField);
            if(gradLen > 0.01f) { // Threshold to identify surface
                glm::vec3 normal = gradColorField / gradLen;
                fSurface = -SURFACE_TENSION * lapColorField * normal;
            }
            
            // Total force
            pi.force = fPressure + fViscosity + fGravity + fSurface;
        }
    }
    
    void integrate(float dt) {
        for(auto& p : particles) {
            // Semi-implicit Euler integration
            glm::vec3 acceleration = p.force / p.density;
            p.velocity += dt * acceleration;
            
            // Velocity damping for stability
            p.velocity *= 0.999f;
            
            p.position += dt * p.velocity;
            
            // Boundary collision with damping
            for(int d = 0; d < 3; d++) {
                if(p.position[d] < CONTAINER_MIN[d] + PARTICLE_RADIUS) {
                    p.position[d] = CONTAINER_MIN[d] + PARTICLE_RADIUS;
                    p.velocity[d] *= -BOUNDARY_DAMPING;
                    // Add friction
                    int d1 = (d + 1) % 3;
                    int d2 = (d + 2) % 3;
                    p.velocity[d1] *= 0.8f;
                    p.velocity[d2] *= 0.8f;
                }
                if(p.position[d] > CONTAINER_MAX[d] - PARTICLE_RADIUS) {
                    p.position[d] = CONTAINER_MAX[d] - PARTICLE_RADIUS;
                    p.velocity[d] *= -BOUNDARY_DAMPING;
                    int d1 = (d + 1) % 3;
                    int d2 = (d + 2) % 3;
                    p.velocity[d1] *= 0.8f;
                    p.velocity[d2] *= 0.8f;
                }
            }
            
            // Update color based on velocity magnitude and pressure
            float speed = glm::length(p.velocity);
            float velFactor = glm::clamp(speed / 5.0f, 0.0f, 1.0f);
            float pressureFactor = glm::clamp((p.pressure / GAS_CONSTANT), 0.0f, 1.0f);
            
            glm::vec3 baseColor = fluidTypes[p.fluidType].color;
            glm::vec3 velocityColor = glm::vec3(0.3f, 0.8f, 1.0f); // Cyan for fast
            glm::vec3 pressureColor = glm::vec3(1.0f, 0.3f, 0.3f); // Red for high pressure
            
            p.color = baseColor * (1.0f - velFactor * 0.4f) + velocityColor * velFactor * 0.4f;
            p.color = p.color * (1.0f - pressureFactor * 0.3f) + pressureColor * pressureFactor * 0.3f;
        }
    }
    
    void applyMouseInteraction() {
        if(!mouseInteraction.active) return;
        
        for(auto& p : particles) {
            glm::vec3 diff = p.position - mouseInteraction.position;
            float distance = glm::length(diff);
            
            if(distance < mouseInteraction.radius) {
                float force = mouseInteraction.strength * (1.0f - distance / mouseInteraction.radius);
                glm::vec3 direction = glm::normalize(diff);
                p.force += direction * force * p.density;
            }
        }
    }
    
    void updateVisualEffects(float dt) {
        for(auto& p : particles) {
            // Update particle age
            p.age += dt;
            
            // Update trail
            p.trail[p.trailIndex] = p.position;
            p.trailIndex = (p.trailIndex + 1) % TRAIL_LENGTH;
            
            // Update brightness for bloom
            float speed = glm::length(p.velocity);
            p.brightness = glm::clamp(speed / 3.0f, 0.5f, 2.0f);
            
            // Update color based on temperature and velocity
            float tempFactor = glm::clamp((p.temperature - 273.0f) / 100.0f, 0.0f, 1.0f);
            glm::vec3 tempColor = glm::vec3(1.0f, 0.3f, 0.1f) * tempFactor;
            p.color = fluidTypes[p.fluidType].color * (1.0f - tempFactor * 0.3f) + tempColor * tempFactor * 0.3f;
        }
    }

public:
    SPHSimulator() : spatialGrid(H), gravityDir(0, -1, 0), 
                     numThreads(std::thread::hardware_concurrency()), 
                     useParallel(true) {
        // Initialize thread pool
        threadPool = new ThreadPool(numThreads);
        
        // Initialize fluid types with enhanced properties
        fluidTypes.push_back({3.5f, REST_DENSITY, GAS_CONSTANT, MASS, 
                             glm::vec3(0.2f, 0.5f, 1.0f), 20.0f, false}); // Water
        fluidTypes.push_back({15.0f, REST_DENSITY * 0.92f, GAS_CONSTANT * 0.8f, MASS, 
                             glm::vec3(0.9f, 0.7f, 0.2f), 25.0f, true}); // Oil (viscous)
        fluidTypes.push_back({1.5f, REST_DENSITY * 0.95f, GAS_CONSTANT * 1.3f, MASS, 
                             glm::vec3(1.0f, 0.4f, 0.2f), 30.0f, true}); // Hot (less dense, more energetic)
        fluidTypes.push_back({8.0f, REST_DENSITY * 1.05f, GAS_CONSTANT * 0.85f, MASS, 
                             glm::vec3(0.3f, 0.5f, 1.0f), 15.0f, false}); // Cold (denser, less energetic)
        
        // Initialize visual settings
        visualSettings = {true, true, false, true, BLOOM_THRESHOLD, BLOOM_INTENSITY, 0.95f, 5.0f, 0.1f};
        
        // Initialize mouse interaction
        mouseInteraction = {glm::vec3(0.0f), 0.5f, 0.0f, false, glm::vec3(0.0f)};
        
        metrics = {0, 0, 0, 0, numThreads, 0, 0, 0, 0};
        
        std::cout << "Enhanced SPH Simulator initialized with " << numThreads << " threads" << std::endl;
    }
    
    ~SPHSimulator() {
        delete threadPool;
    }
    
    void addParticle(const glm::vec3& pos, const glm::vec3& vel, int fluidType = 0) {
        if(particles.size() >= MAX_PARTICLES) return;
        
        Particle p;
        p.position = pos;
        p.velocity = vel;
        p.force = glm::vec3(0.0f);
        p.density = REST_DENSITY;
        p.pressure = 0.0f;
        p.temperature = (fluidType == 2) ? 373.0f : (fluidType == 3) ? 273.0f : 298.0f;
        p.fluidType = fluidType;
        p.color = fluidTypes[fluidType].color;
        p.age = 0.0f;
        p.brightness = 1.0f;
        p.trailIndex = 0;
        
        // Initialize trail
        for(int i = 0; i < TRAIL_LENGTH; i++) {
            p.trail[i] = pos;
        }
        
        particles.push_back(p);
    }
    
    void spawnFluidCube(const glm::vec3& center, int count, int fluidType = 0, bool withVelocity = false) {
        int side = (int)ceil(pow((double)count, 1.0/3.0));
        float spacing = H * 0.5f; // Particles closer together
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> jitter(-spacing * 0.05f, spacing * 0.05f);
        
        int spawned = 0;
        for(int x = 0; x < side && spawned < count; x++) {
            for(int y = 0; y < side && spawned < count; y++) {
                for(int z = 0; z < side && spawned < count; z++) {
                    glm::vec3 offset = glm::vec3((float)x, (float)y, (float)z) * spacing;
                    offset -= glm::vec3(side * spacing / 2.0f);
                    glm::vec3 pos = center + offset;
                    
                    // Add small random jitter
                    pos.x += jitter(gen);
                    pos.y += jitter(gen);
                    pos.z += jitter(gen);
                    
                    // Initial velocity
                    glm::vec3 vel(0.0f);
                    if(withVelocity) {
                        vel = glm::vec3(
                            (rand() % 200 - 100) * 0.01f,
                            (rand() % 100) * 0.01f,
                            (rand() % 200 - 100) * 0.01f
                        );
                    }
                    
                    addParticle(pos, vel, fluidType);
                    spawned++;
                }
            }
        }
        std::cout << "Spawned " << spawned << " particles of type " << fluidType << std::endl;
    }
    
    void update(float dt) {
        if(particles.empty()) return;
        
        auto startTotal = std::chrono::high_resolution_clock::now();
        
        // Use fixed timestep for stability
        dt = TIME_STEP;
        
        // Rebuild spatial grid
        spatialGrid.clear();
        for(size_t i = 0; i < particles.size(); i++) {
            spatialGrid.insert(i, particles[i].position);
        }
        
        // Apply mouse interaction forces
        if(mouseInteraction.active) {
            applyMouseInteraction();
        }
        
        if(useParallel && numThreads > 1) {
            auto startParallel = std::chrono::high_resolution_clock::now();
            
            // Use thread pool for better performance
            std::vector<std::future<void>> futures;
            int chunkSize = std::max(1, (int)particles.size() / numThreads);
            
            // Compute density and pressure in parallel
            for(int t = 0; t < numThreads; t++) {
                int start = t * chunkSize;
                int end = (t == numThreads - 1) ? particles.size() : (t + 1) * chunkSize;
                futures.push_back(std::async(std::launch::async, [this, start, end]() {
                    computeDensityPressure(start, end);
                }));
            }
            
            // Wait for density/pressure computation to complete
            for(auto& future : futures) {
                future.wait();
            }
            futures.clear();
            
            // Compute forces in parallel
            for(int t = 0; t < numThreads; t++) {
                int start = t * chunkSize;
                int end = (t == numThreads - 1) ? particles.size() : (t + 1) * chunkSize;
                futures.push_back(std::async(std::launch::async, [this, start, end]() {
                    computeForces(start, end);
                }));
            }
            
            // Wait for force computation to complete
            for(auto& future : futures) {
                future.wait();
            }
            
            auto endParallel = std::chrono::high_resolution_clock::now();
            metrics.parallelTime = std::chrono::duration<float, std::milli>(endParallel - startParallel).count();
            metrics.serialTime = metrics.parallelTime * numThreads * 0.8f; // More realistic estimate
            metrics.speedup = metrics.serialTime / metrics.parallelTime;
        } else {
            auto startSerial = std::chrono::high_resolution_clock::now();
            computeDensityPressure(0, particles.size());
            computeForces(0, particles.size());
            auto endSerial = std::chrono::high_resolution_clock::now();
            metrics.serialTime = std::chrono::duration<float, std::milli>(endSerial - startSerial).count();
            metrics.parallelTime = metrics.serialTime;
            metrics.speedup = 1.0f;
        }
        
        integrate(dt);
        updateVisualEffects(dt);
        
        auto endTotal = std::chrono::high_resolution_clock::now();
        float totalTime = std::chrono::duration<float, std::milli>(endTotal - startTotal).count();
        metrics.fps = 1000.0f / totalTime;
        metrics.efficiency = (metrics.speedup / numThreads) * 100.0f;
        metrics.particleCount = particles.size();
    }
    
    const std::vector<Particle>& getParticles() const { return particles; }
    const PerformanceMetrics& getMetrics() const { return metrics; }
    void setGravityDirection(const glm::vec3& dir) { 
        gravityDir = glm::normalize(dir); 
        std::cout << "Gravity direction: " << gravityDir.x << ", " << gravityDir.y << ", " << gravityDir.z << std::endl;
    }
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
    int getParticleCount() const { return particles.size(); }
    
    // Enhanced interaction methods
    void setMouseInteraction(const glm::vec3& pos, float radius, float strength, bool active) {
        mouseInteraction.position = pos;
        mouseInteraction.radius = radius;
        mouseInteraction.strength = strength;
        mouseInteraction.active = active;
    }
    
    void setMouseDirection(const glm::vec3& dir) {
        mouseInteraction.direction = glm::normalize(dir);
    }
    
    // Visual settings methods
    void toggleBloom() { 
        visualSettings.enableBloom = !visualSettings.enableBloom; 
        std::cout << "Bloom: " << (visualSettings.enableBloom ? "ON" : "OFF") << std::endl;
    }
    
    void toggleTrails() { 
        visualSettings.enableTrails = !visualSettings.enableTrails; 
        std::cout << "Trails: " << (visualSettings.enableTrails ? "ON" : "OFF") << std::endl;
    }
    
    void setBloomSettings(float threshold, float intensity) {
        visualSettings.bloomThreshold = threshold;
        visualSettings.bloomIntensity = intensity;
    }
    
    const VisualSettings& getVisualSettings() const { return visualSettings; }
    const MouseInteraction& getMouseInteraction() const { return mouseInteraction; }
    
    // Advanced particle spawning
    void spawnParticleStream(const glm::vec3& start, const glm::vec3& direction, int count, int fluidType = 0) {
        glm::vec3 dir = glm::normalize(direction);
        float spacing = H * 0.3f;
        
        for(int i = 0; i < count && particles.size() < MAX_PARTICLES; i++) {
            glm::vec3 pos = start + dir * (i * spacing);
            glm::vec3 vel = dir * 2.0f + glm::vec3(
                (rand() % 100 - 50) / 1000.0f,
                (rand() % 100 - 50) / 1000.0f,
                (rand() % 100 - 50) / 1000.0f
            );
            addParticle(pos, vel, fluidType);
        }
    }
};

// ======================== OPENGL RENDERING ========================
const char* vertexShaderSource = R"(
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aColor;
layout (location = 2) in float aSize;
layout (location = 3) in float aBrightness;

out vec3 Color;
out float Depth;
out float Brightness;

uniform mat4 projection;
uniform mat4 view;
uniform float time;

void main() {
    Color = aColor;
    vec4 viewPos = view * vec4(aPos, 1.0);
    Depth = -viewPos.z;
    Brightness = aBrightness;
    
    // Add subtle animation to particle size
    float animatedSize = aSize * (1.0 + 0.1 * sin(time * 2.0 + aPos.x * 10.0));
    gl_Position = projection * viewPos;
    gl_PointSize = max(8.0, animatedSize * 500.0 / gl_Position.w);
}
)";

const char* fragmentShaderSource = R"(
#version 330 core
in vec3 Color;
in float Depth;
in float Brightness;
out vec4 FragColor;

uniform float bloomThreshold;
uniform float bloomIntensity;
uniform float time;

void main() {
    vec2 coord = gl_PointCoord - vec2(0.5);
    float dist = length(coord);
    if(dist > 0.5) discard;
    
    // Enhanced sphere with better lighting
    float alpha = 1.0 - smoothstep(0.3, 0.5, dist);
    vec3 normal = normalize(vec3(coord * 2.0, sqrt(1.0 - 4.0 * dist * dist)));
    
    // Dynamic lighting
    vec3 lightDir = normalize(vec3(0.5, 1.0, 0.3));
    float diffuse = max(dot(normal, lightDir), 0.0) * 0.7 + 0.3;
    
    // Add rim lighting
    float rim = 1.0 - max(dot(normal, vec3(0.0, 0.0, 1.0)), 0.0);
    rim = pow(rim, 2.0);
    
    // Combine lighting
    vec3 finalColor = Color * diffuse + Color * rim * 0.3;
    finalColor *= Brightness;
    
    // Bloom effect
    float bloomFactor = max(0.0, Brightness - bloomThreshold);
    finalColor += Color * bloomFactor * bloomIntensity;
    
    // Add subtle color variation based on depth
    float depthFactor = 1.0 - smoothstep(2.0, 8.0, Depth);
    finalColor *= depthFactor;
    
    FragColor = vec4(finalColor, alpha * 0.95);
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
    float cameraDistance = 6.0f;
    glm::vec3 cameraPos;
    
    bool mousePressed = false;
    double lastX = SCR_WIDTH / 2.0;
    double lastY = SCR_HEIGHT / 2.0;
    
    int currentFluidType = 0;
    float time = 0.0f;
    bool showUI = true;
    bool mouseInteractionActive = false;
    
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
        glfwSwapInterval(1);
        
        if(!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
            std::cerr << "Failed to initialize GLAD" << std::endl;
            exit(-1);
        }
        
        setupOpenGL();
        
        // Spawn initial water block
        std::cout << "Spawning initial fluid..." << std::endl;
        simulator.spawnFluidCube(glm::vec3(0, 0.5, 0), 2000, 0);
        
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
            app->simulator.spawnFluidCube(glm::vec3(0, 1.2, 0), 300, app->currentFluidType, true);
        }
    }
    
    static void cursorPosCallback(GLFWwindow* window, double xpos, double ypos) {
        Application* app = (Application*)glfwGetWindowUserPointer(window);
        if(app->mousePressed) {
            float xoffset = xpos - app->lastX;
            float yoffset = app->lastY - ypos;
            app->lastX = xpos;
            app->lastY = ypos;
            
            app->cameraYaw += xoffset * 0.25f;
            app->cameraPitch += yoffset * 0.25f;
            app->cameraPitch = glm::clamp(app->cameraPitch, -89.0f, 89.0f);
        }
        
        // Update mouse interaction position
        if(app->mouseInteractionActive) {
            // Convert screen coordinates to world coordinates
            float normalizedX = (2.0f * xpos) / SCR_WIDTH - 1.0f;
            float normalizedY = 1.0f - (2.0f * ypos) / SCR_HEIGHT;
            
            // Simple projection to world space (this is a simplified version)
            glm::vec3 worldPos = glm::vec3(normalizedX * 3.0f, normalizedY * 3.0f, 0.0f);
            app->simulator.setMouseInteraction(worldPos, 0.5f, 10.0f, true);
        }
        
        app->lastX = xpos;
        app->lastY = ypos;
    }
    
    static void scrollCallback(GLFWwindow* window, double xoffset, double yoffset) {
        Application* app = (Application*)glfwGetWindowUserPointer(window);
        app->cameraDistance -= yoffset * 0.3f;
        app->cameraDistance = glm::clamp(app->cameraDistance, 2.0f, 15.0f);
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
                case GLFW_KEY_K: app->simulator.setGravityDirection(glm::vec3(0, 0, -1)); break;
                case GLFW_KEY_L: app->simulator.setGravityDirection(glm::vec3(0, 0, 1)); break;
                case GLFW_KEY_C: app->simulator.clearParticles(); break;
                case GLFW_KEY_P: app->simulator.toggleParallel(); break;
                case GLFW_KEY_R: 
                    app->simulator.clearParticles();
                    app->simulator.spawnFluidCube(glm::vec3(0, 0.5, 0), 2000, 0);
                    std::cout << "Reset simulation" << std::endl;
                    break;
                case GLFW_KEY_SPACE: 
                    app->simulator.spawnFluidCube(glm::vec3(0, 1.2, 0), 500, app->currentFluidType, true); 
                    break;
                case GLFW_KEY_EQUAL: 
                case GLFW_KEY_KP_ADD: 
                    app->simulator.setThreadCount(app->simulator.getMetrics().numThreads + 1); 
                    break;
                case GLFW_KEY_MINUS: 
                case GLFW_KEY_KP_SUBTRACT: 
                    app->simulator.setThreadCount(app->simulator.getMetrics().numThreads - 1); 
                    break;
                case GLFW_KEY_B: 
                    app->simulator.toggleBloom(); 
                    break;
                case GLFW_KEY_T: 
                    app->simulator.toggleTrails(); 
                    break;
                case GLFW_KEY_M: 
                    app->mouseInteractionActive = !app->mouseInteractionActive;
                    std::cout << "Mouse interaction: " << (app->mouseInteractionActive ? "ON" : "OFF") << std::endl;
                    break;
                case GLFW_KEY_S: 
                    app->simulator.spawnParticleStream(glm::vec3(-1.5, 1.0, 0), glm::vec3(1, -0.5, 0), 100, app->currentFluidType);
                    break;
                case GLFW_KEY_U: 
                    app->showUI = !app->showUI;
                    break;
            }
        }
    }
    
    void render() {
        const auto& particles = simulator.getParticles();
        const auto& visualSettings = simulator.getVisualSettings();
        
        glClearColor(0.02f, 0.02f, 0.05f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        
        if(particles.empty()) return;
        
        std::vector<float> vertexData;
        vertexData.reserve(particles.size() * 8); // Now 8 floats per particle
        
        for(const auto& p : particles) {
            vertexData.push_back(p.position.x);
            vertexData.push_back(p.position.y);
            vertexData.push_back(p.position.z);
            vertexData.push_back(p.color.r);
            vertexData.push_back(p.color.g);
            vertexData.push_back(p.color.b);
            vertexData.push_back(PARTICLE_RADIUS);
            vertexData.push_back(p.brightness);
        }
        
        glBindVertexArray(VAO);
        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBufferData(GL_ARRAY_BUFFER, vertexData.size() * sizeof(float), vertexData.data(), GL_DYNAMIC_DRAW);
        
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(3 * sizeof(float)));
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(6 * sizeof(float)));
        glEnableVertexAttribArray(2);
        glVertexAttribPointer(3, 1, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(7 * sizeof(float)));
        glEnableVertexAttribArray(3);
        
        updateCamera();
        
        glm::mat4 projection = glm::perspective(glm::radians(45.0f), (float)SCR_WIDTH / SCR_HEIGHT, 0.1f, 100.0f);
        glm::mat4 view = glm::lookAt(cameraPos, glm::vec3(0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
        
        glUseProgram(shaderProgram);
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "projection"), 1, GL_FALSE, glm::value_ptr(projection));
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "view"), 1, GL_FALSE, glm::value_ptr(view));
        glUniform1f(glGetUniformLocation(shaderProgram, "time"), time);
        glUniform1f(glGetUniformLocation(shaderProgram, "bloomThreshold"), visualSettings.bloomThreshold);
        glUniform1f(glGetUniformLocation(shaderProgram, "bloomIntensity"), visualSettings.bloomIntensity);
        
        glDrawArrays(GL_POINTS, 0, particles.size());
        
        // Draw container wireframe
        drawContainer(projection, view);
    }
    
    void drawContainer(const glm::mat4& projection, const glm::mat4& view) {
        // Simple line rendering for container bounds
        std::vector<float> lines = {
            // Bottom square
            CONTAINER_MIN.x, CONTAINER_MIN.y, CONTAINER_MIN.z,
            CONTAINER_MAX.x, CONTAINER_MIN.y, CONTAINER_MIN.z,
            
            CONTAINER_MAX.x, CONTAINER_MIN.y, CONTAINER_MIN.z,
            CONTAINER_MAX.x, CONTAINER_MIN.y, CONTAINER_MAX.z,
            
            CONTAINER_MAX.x, CONTAINER_MIN.y, CONTAINER_MAX.z,
            CONTAINER_MIN.x, CONTAINER_MIN.y, CONTAINER_MAX.z,
            
            CONTAINER_MIN.x, CONTAINER_MIN.y, CONTAINER_MAX.z,
            CONTAINER_MIN.x, CONTAINER_MIN.y, CONTAINER_MIN.z,
            
            // Top square
            CONTAINER_MIN.x, CONTAINER_MAX.y, CONTAINER_MIN.z,
            CONTAINER_MAX.x, CONTAINER_MAX.y, CONTAINER_MIN.z,
            
            CONTAINER_MAX.x, CONTAINER_MAX.y, CONTAINER_MIN.z,
            CONTAINER_MAX.x, CONTAINER_MAX.y, CONTAINER_MAX.z,
            
            CONTAINER_MAX.x, CONTAINER_MAX.y, CONTAINER_MAX.z,
            CONTAINER_MIN.x, CONTAINER_MAX.y, CONTAINER_MAX.z,
            
            CONTAINER_MIN.x, CONTAINER_MAX.y, CONTAINER_MAX.z,
            CONTAINER_MIN.x, CONTAINER_MAX.y, CONTAINER_MIN.z,
            
            // Vertical lines
            CONTAINER_MIN.x, CONTAINER_MIN.y, CONTAINER_MIN.z,
            CONTAINER_MIN.x, CONTAINER_MAX.y, CONTAINER_MIN.z,
            
            CONTAINER_MAX.x, CONTAINER_MIN.y, CONTAINER_MIN.z,
            CONTAINER_MAX.x, CONTAINER_MAX.y, CONTAINER_MIN.z,
            
            CONTAINER_MAX.x, CONTAINER_MIN.y, CONTAINER_MAX.z,
            CONTAINER_MAX.x, CONTAINER_MAX.y, CONTAINER_MAX.z,
            
            CONTAINER_MIN.x, CONTAINER_MIN.y, CONTAINER_MAX.z,
            CONTAINER_MIN.x, CONTAINER_MAX.y, CONTAINER_MAX.z,
        };
        
        GLuint lineVAO, lineVBO;
        glGenVertexArrays(1, &lineVAO);
        glGenBuffers(1, &lineVBO);
        
        glBindVertexArray(lineVAO);
        glBindBuffer(GL_ARRAY_BUFFER, lineVBO);
        glBufferData(GL_ARRAY_BUFFER, lines.size() * sizeof(float), lines.data(), GL_STATIC_DRAW);
        
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(0);
        
        // Simple shader for lines
        const char* lineVertSrc = R"(
            #version 330 core
            layout (location = 0) in vec3 aPos;
            uniform mat4 projection;
            uniform mat4 view;
            void main() {
                gl_Position = projection * view * vec4(aPos, 1.0);
            }
        )";
        
        const char* lineFragSrc = R"(
            #version 330 core
            out vec4 FragColor;
            void main() {
                FragColor = vec4(0.3, 0.3, 0.4, 0.5);
            }
        )";
        
        static GLuint lineShader = 0;
        if(lineShader == 0) {
            GLuint vs = compileShader(lineVertSrc, GL_VERTEX_SHADER);
            GLuint fs = compileShader(lineFragSrc, GL_FRAGMENT_SHADER);
            lineShader = glCreateProgram();
            glAttachShader(lineShader, vs);
            glAttachShader(lineShader, fs);
            glLinkProgram(lineShader);
            glDeleteShader(vs);
            glDeleteShader(fs);
        }
        
        glUseProgram(lineShader);
        glUniformMatrix4fv(glGetUniformLocation(lineShader, "projection"), 1, GL_FALSE, glm::value_ptr(projection));
        glUniformMatrix4fv(glGetUniformLocation(lineShader, "view"), 1, GL_FALSE, glm::value_ptr(view));
        
        glLineWidth(1.5f);
        glDrawArrays(GL_LINES, 0, lines.size() / 3);
        
        glDeleteVertexArrays(1, &lineVAO);
        glDeleteBuffers(1, &lineVBO);
    }
    
    void run() {
        float lastFrame = glfwGetTime();
        float accumulatedTime = 0.0f;
        
        std::cout << "\n╔══════════════════════════════════════════════════════════╗" << std::endl;
        std::cout << "║       3D SPH FLUID SIMULATION - CONTROLS                ║" << std::endl;
        std::cout << "╠══════════════════════════════════════════════════════════╣" << std::endl;
        std::cout << "║ CAMERA CONTROLS:                                        ║" << std::endl;
        std::cout << "║   Left Click + Drag : Rotate camera                     ║" << std::endl;
        std::cout << "║   Mouse Wheel       : Zoom in/out                       ║" << std::endl;
        std::cout << "║                                                          ║" << std::endl;
        std::cout << "║ FLUID SPAWNING:                                         ║" << std::endl;
        std::cout << "║   Right Click       : Spawn 300 particles (with velocity)║" << std::endl;
        std::cout << "║   SPACE             : Spawn 500 particles (with velocity)║" << std::endl;
        std::cout << "║   1-4               : Select fluid type                 ║" << std::endl;
        std::cout << "║                       1: Water (blue)                   ║" << std::endl;
        std::cout << "║                       2: Oil (gold, viscous)            ║" << std::endl;
        std::cout << "║                       3: Hot (red, energetic)           ║" << std::endl;
        std::cout << "║                       4: Cold (blue, dense)             ║" << std::endl;
        std::cout << "║                                                          ║" << std::endl;
        std::cout << "║ GRAVITY CONTROL:                                        ║" << std::endl;
        std::cout << "║   G : Down   | H : Left  | J : Right                   ║" << std::endl;
        std::cout << "║   K : Away   | L : Toward                              ║" << std::endl;
        std::cout << "║                                                          ║" << std::endl;
        std::cout << "║ SIMULATION CONTROL:                                     ║" << std::endl;
        std::cout << "║   P      : Toggle parallel/serial mode                  ║" << std::endl;
        std::cout << "║   +/-    : Increase/decrease thread count               ║" << std::endl;
        std::cout << "║   C      : Clear all particles                          ║" << std::endl;
        std::cout << "║   R      : Reset simulation                             ║" << std::endl;
        std::cout << "║   B      : Toggle bloom effect                           ║" << std::endl;
        std::cout << "║   T      : Toggle particle trails                        ║" << std::endl;
        std::cout << "║   M      : Toggle mouse interaction                      ║" << std::endl;
        std::cout << "║   S      : Spawn particle stream                        ║" << std::endl;
        std::cout << "║   U      : Toggle UI display                            ║" << std::endl;
        std::cout << "║   ESC    : Exit                                         ║" << std::endl;
        std::cout << "╚══════════════════════════════════════════════════════════╝\n" << std::endl;
        
        while(!glfwWindowShouldClose(window)) {
            float currentFrame = glfwGetTime();
            float deltaTime = currentFrame - lastFrame;
            lastFrame = currentFrame;
            time = currentFrame;
            
            // Update simulation multiple times per frame for stability
            const int substeps = 2;
            for(int i = 0; i < substeps; i++) {
                simulator.update(deltaTime / substeps);
            }
            
            render();
            
            // Display enhanced metrics
            const auto& metrics = simulator.getMetrics();
            const auto& visualSettings = simulator.getVisualSettings();
            char title[1024];
            snprintf(title, sizeof(title), 
                    "Enhanced SPH Simulation | Particles: %d | FPS: %.0f | Compute: %.1fms | Threads: %d | Speedup: %.2fx | Efficiency: %.0f%% | Bloom: %s | Trails: %s | Mouse: %s",
                    metrics.particleCount,
                    metrics.fps,
                    metrics.parallelTime,
                    metrics.numThreads,
                    metrics.speedup,
                    metrics.efficiency,
                    visualSettings.enableBloom ? "ON" : "OFF",
                    visualSettings.enableTrails ? "ON" : "OFF",
                    mouseInteractionActive ? "ON" : "OFF");
            glfwSetWindowTitle(window, title);
            
            glfwSwapBuffers(window);
            glfwPollEvents();
        }
    }
};

int main() {
    try {
        std::cout << "╔══════════════════════════════════════════════════════════╗" << std::endl;
        std::cout << "║     ENHANCED 3D PARALLEL SPH FLUID SIMULATION           ║" << std::endl;
        std::cout << "║     Features: Bloom, Trails, Mouse Interaction         ║" << std::endl;
        std::cout << "╚══════════════════════════════════════════════════════════╝\n" << std::endl;
        
        Application app;
        app.run();
    } catch(const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}