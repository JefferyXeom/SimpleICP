#ifndef P2PICP_H
#define P2PICP_H

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <Eigen/Dense>

class P2PICP {
public:
    // 构造函数，接受点云文件路径
    P2PICP(const std::string& source_pcd_path, const std::string& target_pcd_path);

    // 执行 ICP 配准
    void align(double tolerance = 1e-6, int max_iterations = 50);

    void alignWithLossFunction(double tolerance = 1e-6, int max_iterations = 50);
    
    // 获取最终的变换矩阵
    Eigen::Matrix4d getTransformation() const;

    // 可视化点云
    void visualize();

private:
    // 数据成员
    pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud; // 源点云
    pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud; // 目标点云
    Eigen::Matrix4d transformation; // 当前的变换矩阵

    // 私有成员函数
    pcl::PointCloud<pcl::PointXYZ>::Ptr findClosestPoints(); // 查找最近邻点
    double computeError(const pcl::PointCloud<pcl::PointXYZ>::Ptr& closest_cloud); // 计算误差
    Eigen::Matrix4d computeTransformation(const pcl::PointCloud<pcl::PointXYZ>::Ptr& closest_cloud); // 计算刚体变换
    void applyTransformation(const Eigen::Matrix4d& transform); // 应用变换到源点云
};

#endif // P2PICP_H
