#include "P2PICP.hpp"
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/transforms.h>
#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <thread>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <algorithm>
#include <cmath>
#include <Eigen/Dense>

// 构造函数
P2PICP::P2PICP(const std::string &source_pcd_path, const std::string &target_pcd_path)
    : source_cloud(new pcl::PointCloud<pcl::PointXYZ>()),
      target_cloud(new pcl::PointCloud<pcl::PointXYZ>()),
      transformation(Eigen::Matrix4d::Identity())
{
    // 加载源点云
    if (pcl::io::loadPCDFile<pcl::PointXYZ>(source_pcd_path, *source_cloud) == -1)
    {
        throw std::runtime_error("Failed to load source PCD file.");
    }

    // 加载目标点云
    if (pcl::io::loadPCDFile<pcl::PointXYZ>(target_pcd_path, *target_cloud) == -1)
    {
        throw std::runtime_error("Failed to load target PCD file.");
    }
}

// 执行 ICP 配准
void P2PICP::align(double tolerance, int max_iterations)
{
    int iteration = 0;
    double prev_error = std::numeric_limits<double>::max();

    while (iteration < max_iterations)
    {
        // 查找最近邻点
        auto closest_points = findClosestPoints();

        // 计算误差
        double current_error = computeError(closest_points);
        if (std::abs(prev_error - current_error) < tolerance)
        {
            break;
        }

        // 计算刚体变换
        Eigen::Matrix4d current_transform = computeTransformation(closest_points);

        // 更新全局变换
        transformation = current_transform * transformation;

        // 应用当前变换到源点云
        applyTransformation(current_transform);

        prev_error = current_error;
        iteration++;
    }
    std::cout << "ICP converged after " << iteration << " iterations." << std::endl;
}

// 获取最终的变换矩阵
Eigen::Matrix4d P2PICP::getTransformation() const
{
    return transformation;
}

// 查找最近邻点
pcl::PointCloud<pcl::PointXYZ>::Ptr P2PICP::findClosestPoints()
{
    auto closest_points = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>());

    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    kdtree.setInputCloud(target_cloud);

    for (const auto &point : source_cloud->points)
    {
        std::vector<int> point_idx(1);
        std::vector<float> point_distance(1);

        if (kdtree.nearestKSearch(point, 1, point_idx, point_distance) > 0)
        {
            closest_points->points.push_back(target_cloud->points[point_idx[0]]);
        }
    }

    return closest_points;
}

// 计算误差
double P2PICP::computeError(const pcl::PointCloud<pcl::PointXYZ>::Ptr &closest_points)
{
    double error = 0.0;
    for (size_t i = 0; i < source_cloud->points.size(); ++i)
    {
        error += (source_cloud->points[i].getVector3fMap() - closest_points->points[i].getVector3fMap()).squaredNorm();
    }
    return error / source_cloud->points.size();
}

// 计算刚体变换
Eigen::Matrix4d P2PICP::computeTransformation(const pcl::PointCloud<pcl::PointXYZ>::Ptr &closest_points)
{
    Eigen::Matrix4d transformation = Eigen::Matrix4d::Identity();

    // 转换点云为 Eigen 矩阵
    Eigen::MatrixXd source(3, source_cloud->points.size());
    Eigen::MatrixXd target(3, closest_points->points.size());

    for (size_t i = 0; i < source_cloud->points.size(); ++i)
    {
        source.col(i) = source_cloud->points[i].getVector3fMap().cast<double>();
        target.col(i) = closest_points->points[i].getVector3fMap().cast<double>();
    }

    // 计算质心
    Eigen::Vector3d source_mean = source.rowwise().mean();
    Eigen::Vector3d target_mean = target.rowwise().mean();

    // 减去质心
    source.colwise() -= source_mean;
    target.colwise() -= target_mean;

    // 计算协方差矩阵
    Eigen::Matrix3d covariance = source * target.transpose();

    // SVD 分解
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(covariance, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();

    // 计算旋转矩阵
    Eigen::Matrix3d R = V * U.transpose();
    if (R.determinant() < 0)
    {
        V.col(2) *= -1;
        R = V * U.transpose();
    }

    // 计算平移向量
    Eigen::Vector3d t = target_mean - R * source_mean;

    // 组合变换矩阵
    transformation.block<3, 3>(0, 0) = R;
    transformation.block<3, 1>(0, 3) = t;

    return transformation;
}

// 应用变换到源点云
void P2PICP::applyTransformation(const Eigen::Matrix4d &transform)
{
    pcl::transformPointCloud(*source_cloud, *source_cloud, transform);
}

/////////////////////////////////////// 优化部分

// 损失函数：点到点 ICP 配准
struct P2PCostFunctor
{
public:
    P2PCostFunctor(const pcl::PointXYZ &source_point, const pcl::PointXYZ &target_point)
        : source_point(source_point), target_point(target_point) {}

    template <typename T>
    bool operator()(const T *const quaternion, const T *const translation, T *residuals) const
    {
        // 四元数旋转
        T transformed_point[3];
        T source_point_array[3] = {T(source_point.x), T(source_point.y), T(source_point.z)};
        T quaternion_com[4] = {ceres::sqrt(T(1) - ceres::pow(quaternion[0], 2) - ceres::pow(quaternion[1], 2) - ceres::pow(quaternion[2], 2)), 
                            quaternion[0], quaternion[1], quaternion[2]};
        ceres::QuaternionRotatePoint(quaternion_com, source_point_array, transformed_point);

        // 应用平移
        transformed_point[0] += translation[0];
        transformed_point[1] += translation[1];
        transformed_point[2] += translation[2];

        // 计算目标点与变换后的源点之间的差距
        residuals[0] = ceres::pow(transformed_point[0] - T(target_point.x), 2) +
                       ceres::pow(transformed_point[1] - T(target_point.y), 2) +
                       ceres::pow(transformed_point[2] - T(target_point.z), 2);

        return true;
    }

    const pcl::PointXYZ target_point;
    const pcl::PointXYZ source_point;
};

// 计算点云配准的优化问题
void P2PICP::alignWithLossFunction(double tolerance, int max_iterations)
{
    ceres::Problem problem;

    // 初始化旋转和平移
    double quaternion[3] = {0.0, 0.0, 0.0};  // 四元数 [w, x, y, z]，单位四元数表示没有旋转，省略w
    double translation[3] = {0.0, 0.0, 0.0}; // 默认平移为零

    // 输出点云大小
    std::cout << "Source cloud size: " << source_cloud->points.size() << std::endl;
    std::cout << "Target cloud size: " << target_cloud->points.size() << std::endl;

    // 为每一对源点和目标点添加损失函数
    for (size_t i = 0; i < std::min(source_cloud->points.size(), target_cloud->points.size()); ++i)
    {
        pcl::PointXYZ target_point = target_cloud->points[i];
        pcl::PointXYZ source_point = source_cloud->points[i];

        // 创建损失函数
        P2PCostFunctor *cost_function = new P2PCostFunctor(source_point, target_point);

        // 将损失函数添加到问题中
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<P2PCostFunctor, 1, 3, 3>(
                cost_function),
            nullptr, quaternion, translation);
    }

    // 设置优化求解器参数
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.max_num_iterations = max_iterations;
    options.function_tolerance = tolerance;
    options.parameter_tolerance = tolerance;
    options.minimizer_progress_to_stdout = true;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    // 输出优化过程信息
    std::cout << summary.BriefReport() << std::endl; // or FullReport

    // 更新变换
    Eigen::Quaterniond q(quaternion[0], quaternion[1], quaternion[2], quaternion[3]);
    Eigen::Matrix3d rotation_matrix = q.toRotationMatrix();
    transformation.block<3, 3>(0, 0) = rotation_matrix;
    transformation.block<3, 1>(0, 3) = Eigen::Vector3d(translation[0], translation[1], translation[2]);

    // 应用当前变换到源点云
    applyTransformation(transformation);

    // // 可视化点云
    // pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZ>());
    // pcl::transformPointCloud(*source_cloud, *transformed_cloud, transformation); // 变换源点云

    // visualize(source_cloud, target_cloud, transformed_cloud);  // 可视化变换后的点云

    return;
}

/////////////////////////////////////////

void P2PICP::visualize()
{
    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
    viewer->setBackgroundColor(0, 0, 0);

    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> source_color(source_cloud, 255, 0, 0);
    viewer->addPointCloud<pcl::PointXYZ>(source_cloud, source_color, "source cloud");

    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> target_color(target_cloud, 0, 255, 0);
    viewer->addPointCloud<pcl::PointXYZ>(target_cloud, target_color, "target cloud");

    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "source cloud");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "target cloud");

    viewer->addCoordinateSystem(1.0);
    viewer->initCameraParameters();

    while (!viewer->wasStopped())
    {
        viewer->spinOnce(100);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}
