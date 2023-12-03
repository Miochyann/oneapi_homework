#include <CL/sycl.hpp>
#include <iostream>

constexpr int MATRIX_SIZE = 1024; // 可根据需要调整矩阵大小
constexpr int BLOCK_SIZE = 16;    // 可根据需要调整块大小
// 矩阵乘法核函数s
class MatrixMultiply;
int main() {
    // 分配主机端内存
    std::vector<float> inputMatrixA(MATRIX_SIZE * MATRIX_SIZE, 2.0);
    std::vector<float> inputMatrixB(MATRIX_SIZE * MATRIX_SIZE, 3.0);
    std::vector<float> outputMatrix(MATRIX_SIZE * MATRIX_SIZE, 0.0);
    // 初始化设备
    cl::sycl::device device = cl::sycl::default_selector().select_device();
    // 在GPU端分配内存
    cl::sycl::range<2> globalSize(MATRIX_SIZE, MATRIX_SIZE);
    cl::sycl::buffer<float, 2> bufferA(inputMatrixA.data(), globalSize);
    cl::sycl::buffer<float, 2> bufferB(inputMatrixB.data(), globalSize);
    cl::sycl::buffer<float, 2> bufferC(outputMatrix.data(), globalSize);
    // 创建队列和执行上下文
    cl::sycl::queue queue(device);
    cl::sycl::context context(queue.get_context());
    // 数据传输：将输入矩阵数据从主机端传输到GPU端
    queue.submit([&](cl::sycl::handler& cgh) {
        auto accessorA = bufferA.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessorB = bufferB.get_access<cl::sycl::access::mode::read>(cgh);
        auto accessorC = bufferC.get_access<cl::sycl::access::mode::write>(cgh);
        cgh.parallel_for<class MatrixMultiply>(globalSize, [=](cl::sycl::id<2> idx) {
            int row = idx[0];
            int col = idx[1];
            float result = 0.0;
            for (int k = 0; k < MATRIX_SIZE; k += BLOCK_SIZE) {
                cl::sycl::group<2> tileGroup = cgh.parallel_for_work_group();
                cl::sycl::group_barrier(tileGroup);
                for (int i = 0; i < BLOCK_SIZE; ++i) {
                    result += accessorA[{row, k + i}] * accessorB[{k + i, col}];
                }
                cl::sycl::group_barrier(tileGroup);
            }
            accessorC[idx] = result;
        });
    }).wait();
    // 数据传输：将输出矩阵数据从GPU端传输回主机端
    queue.submit([&](cl::sycl::handler& cgh) {
        auto accessorC = bufferC.get_access<cl::sycl::access::mode::read>(cgh);
        cgh.copy(accessorC, outputMatrix.data());
    }).wait();
    // 输出结果
    std::cout << "Matrix multiplication result:" << std::endl;
    for (int i = 0; i < std::min(5, MATRIX_SIZE); ++i) {
        for (int j = 0; j < std::min(5, MATRIX_SIZE); ++j) {
            std::cout << outputMatrix[i * MATRIX_SIZE + j] << " ";
        }
        std::cout << std::endl;
    }
    return 0;
}
