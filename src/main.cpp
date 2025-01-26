#include "P2PICP.hpp"
// #include "test.hpp"


int main() {
    try {
        // 初始化 P2PICP 类
        P2PICP icp("../data/1.pcd", "../data/2.pcd");

        // 运行 ICP
        // icp.align();
        icp.alignWithLossFunction();

        // 打印最终的变换矩阵
        Eigen::Matrix4d transform = icp.getTransformation();
        std::cout << "Final Transformation:\n" << transform << std::endl;

        // 可视化初始和对齐后的点云
        icp.visualize();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    return 0;
}
