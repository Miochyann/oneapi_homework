# oneapi_homework
OneAPI是Intel开发的一个跨架构的编程模型，旨在简化在不同类型的处理器上进行并行计算的开发。这包括CPU、GPU、FPGA等多种处理器。OneAPI的目标是提供一个统一的编程接口，使开发人员能够更轻松地利用各种硬件加速器的性能。

OneAPI包括一系列的工具、库和编程语言扩展，以支持高性能计算、数据并行、机器学习等各种工作负载。其中，Data Parallel C++（DPC++）是OneAPI中的一种编程语言扩展，用于实现数据并行性。此外，OneAPI还提供了一些性能分析工具和库，以帮助开发人员优化其代码并充分利用硬件加速。

通过OneAPI，开发人员可以更容易地编写可在不同处理器上执行的代码，从而提高跨架构应用程序的可移植性。

### 作业1:并⾏矩阵乘法
#### 描述
编写⼀个基于oneAPI的C++/SYCL程序来执行矩阵乘法操作。需要考虑大尺寸矩阵的乘法操作以及不同线程之间的数据依赖关系。通常在实现矩阵乘法时，可以使用块矩阵乘法以及共享内存来提高计算效率。
#### 分析
利用基于SYCL的编程模型在GPU上实现矩阵乘法的计算，步骤如下：
1. 分配内存：在主机端分配内存空间用于存储输⼊矩阵和输出矩阵，同时在GPU端分配内存空间用于存储相应的输入和输出数据。
2. 数据传输：将输入矩阵数据从主机端内存传输到GPU端内存中。
3. 核函数调用：在SYCL中，矩阵乘法的计算通常会在GPU上使用核函数来实现并行计算。核函数会分配线程块和线程来处理不同的数据块。
4. 并行计算：在核函数中，每个线程负责计算输出矩阵的⼀个单独的元素。为了最大限度地利用GPU的并行计算能力，通常会使用⼆维线程块和线程网格的方式来处理矩阵的乘法计算。
5. 数据传输：计算完成后，将输出矩阵数据从GPU端内存传输回主机端内存中，以便进⼀步处理或分析。在并行计算矩阵乘法时，可以利用线程块和线程的层次结构来优化计算。通过合理划分矩阵数据并利用共享内存来减少全局内存访问的次数，可以⼤幅提高计算效率。此外，还可以利用GPU上的多个计算单元并执行行矩阵乘法，进⼀步提高计算速度。

代码：
``` cpp
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
```
### 收获
这次是我第一次了解到Oneapi，Oneapi可以提供底层的硬件抽象和并行计算功能。因此，开发人员可以更轻松地编写可移植、高性能的代码，而无需深入了解底层硬件的细节。使用下来感觉相较于其他的API上手简单，环境配置方便，相关文档也十分详细，使用者能够很快上手，同时其对于底层硬件和算法的优化，使得其具有很高的性能优势，以后在涉及到高性能数学计算时会优先考虑使用。