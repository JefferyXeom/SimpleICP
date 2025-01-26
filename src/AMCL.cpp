#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#define _USE_MATH_DEFINES
#include <cmath>
#include <random>
#include <cassert>

struct Pose {
    double x, y, theta;

    Pose(double x_ = 0, double y_ = 0, double theta_ = 0) : x(x_), y(y_), theta(theta_) {}
};

struct Particle {
    Pose pose;
    double weight;

    Particle(Pose p = Pose(), double w = 1.0) : pose(p), weight(w) {}
};

typedef std::vector<std::vector<int>> GridMap;

class AMCL {
public:
    AMCL(int num_particles, const GridMap& map, const GridMap& loss_map);
    void initializeParticles();
    void update(const std::vector<std::vector<double>>& laser_scan);
    void resample();
    Pose getEstimatedPose() const;

private:
    double calculateWeight(const Particle& particle, const std::vector<std::vector<double>>& laser_scan);
    double laserScanLikelihood(const Particle& particle, const std::vector<std::vector<double>>& laser_scan);
    Pose generateRandomPose();
    double getLossAtPosition(int x, int y);

    std::vector<Particle> particles_;
    GridMap map_;
    GridMap loss_map_;
    int num_particles_;
    std::default_random_engine generator_;
    std::normal_distribution<double> normal_distribution_;
};

// Constructor
AMCL::AMCL(int num_particles, const GridMap& map, const GridMap& loss_map)
    : num_particles_(num_particles), map_(map), loss_map_(loss_map), normal_distribution_(0.0, 1.0) {
    initializeParticles();
}

// 初始化粒子
void AMCL::initializeParticles() {
    particles_.clear();
    for (int i = 0; i < num_particles_; ++i) {
        particles_.push_back(Particle(generateRandomPose(), 1.0)); // 初始权重为1
    }
}

Pose AMCL::generateRandomPose() {
    std::uniform_real_distribution<double> x_dist(0.0, static_cast<double>(map_[0].size() - 1));
    std::uniform_real_distribution<double> y_dist(0.0, static_cast<double>(map_.size() - 1));
    std::uniform_real_distribution<double> theta_dist(0.0, 2 * M_PI);

    double x = x_dist(generator_);
    double y = y_dist(generator_);
    double theta = theta_dist(generator_);

    return Pose(x, y, theta);
}

// 获取损失地图中特定位置的损失值
double AMCL::getLossAtPosition(int x, int y) {
    if (x < 0 || y < 0 || x >= map_[0].size() || y >= map_.size()) {
        return 0.0; // 超出边界，返回0损失
    }
    return loss_map_[y][x]; // 获取对应位置的损失值
}

// 计算粒子的权重
double AMCL::calculateWeight(const Particle& particle, const std::vector<std::vector<double>>& laser_scan) {
    return laserScanLikelihood(particle, laser_scan);
}

// 计算激光扫描的相似度（基于全局损失地图）
double AMCL::laserScanLikelihood(const Particle& particle, const std::vector<std::vector<double>>& laser_scan) {
    double weight = 1.0;

    // 将激光扫描数据投影到粒子的位置
    for (size_t i = 0; i < laser_scan.size(); ++i) {
        int x = static_cast<int>(particle.pose.x + laser_scan[i][0] * cos(particle.pose.theta));
        int y = static_cast<int>(particle.pose.y + laser_scan[i][0] * sin(particle.pose.theta));

        // 获取该位置在损失地图中的值
        double loss = getLossAtPosition(x, y);

        // 根据损失值调整粒子的权重
        weight *= exp(-loss); // 假设损失越大，相似度越低
    }

    return weight;
}

// 更新粒子集
void AMCL::update(const std::vector<std::vector<double>>& laser_scan) {
    double total_weight = 0.0;

    // 更新每个粒子的权重
    for (auto& particle : particles_) {
        particle.weight = calculateWeight(particle, laser_scan);
        total_weight += particle.weight;
    }

    // 归一化权重
    for (auto& particle : particles_) {
        particle.weight /= total_weight;
    }
}

// 低方差重采样
void AMCL::resample() {
    std::vector<Particle> resampled_particles;
    double beta = 0.0;
    double max_weight = 0.0;

    // 找到最大权重
    for (const auto& particle : particles_) {
        max_weight = std::max(max_weight, particle.weight);
    }

    std::uniform_real_distribution<double> dist(0.0, max_weight);
    int index = std::rand() % num_particles_;

    // 低方差重采样
    for (int i = 0; i < num_particles_; ++i) {
        beta += dist(generator_) * 2.0;
        while (beta > particles_[index].weight) {
            beta -= particles_[index].weight;
            index = (index + 1) % num_particles_;
        }
        resampled_particles.push_back(particles_[index]);
    }

    particles_ = resampled_particles;
}

// 获取估计的位姿
Pose AMCL::getEstimatedPose() const {
    double x = 0.0, y = 0.0, theta = 0.0;
    double total_weight = 0.0;

    // 根据粒子的加权平均位置估计机器人的位置
    for (const auto& particle : particles_) {
        x += particle.pose.x * particle.weight;
        y += particle.pose.y * particle.weight;
        theta += particle.pose.theta * particle.weight;
        total_weight += particle.weight;
    }

    x /= total_weight;
    y /= total_weight;
    theta /= total_weight;

    return Pose(x, y, theta);
}

// 读取PNG文件并转换为栅格地图
GridMap loadMapFromPNG(const std::string& file_path) {
    cv::Mat img = cv::imread(file_path, cv::IMREAD_GRAYSCALE);
    assert(!img.empty() && "Map image not found");

    GridMap map(img.rows, std::vector<int>(img.cols));
    for (int y = 0; y < img.rows; ++y) {
        for (int x = 0; x < img.cols; ++x) {
            map[y][x] = img.at<uchar>(y, x);
        }
    }
    return map;
}

int main() {
    // 载入地图和损失地图
    GridMap map = loadMapFromPNG("map.png");
    GridMap loss_map = loadMapFromPNG("loss_map.png");

    AMCL amcl(100, map, loss_map);

    // 模拟一次激光扫描数据（简化为3个点）
    std::vector<std::vector<double>> laser_scan = {{1.0, 0.0}, {1.0, 1.0}, {1.0, 2.0}}; // {距离,角度}

    // 更新粒子集
    amcl.update(laser_scan);

    // 重采样
    amcl.resample();

    // 获取定位估计
    Pose estimated_pose = amcl.getEstimatedPose();
    std::cout << "Estimated Position: (" << estimated_pose.x << ", " << estimated_pose.y << ", " << estimated_pose.theta << ")\n";

    return 0;
}
