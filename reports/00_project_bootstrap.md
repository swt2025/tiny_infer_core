# 00 Project Bootstrap

## 今日完成内容

- 创建 C++17 CMake 工程骨架
- 实现 Matrix 类
- 使用 std::vector<float> 存储 row-major 连续内存
- 支持 rows / cols / size / data / fill / operator()
- 添加 Matrix 基础测试

## 当前 Matrix 内存布局

二维访问：

```cpp
m(row, col)
