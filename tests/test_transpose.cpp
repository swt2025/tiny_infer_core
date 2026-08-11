#include <cassert>
#include <iostream>

#include "tinyinfer/ops/transpose.h"

void testZeroRowOrCol()
{
    tinyinfer::Matrix a(0, 1);
    tinyinfer::Matrix b = tinyinfer::transpose(a);

    assert(b.rows() == a.cols() && b.rows() == 1);

    assert(b.cols() == a.rows() && b.cols() == 0);
}

void testNormal()
{
    tinyinfer::Matrix a(2, 3, 0.0f);
    a(0, 0) = 1.0f;
    a(0, 1) = 2.0f;
    a(0, 2) = 3.0f;
    a(1, 0) = 4.0f;
    a(1, 1) = 5.0f;
    a(1, 2) = 6.0f;

    tinyinfer::Matrix b = tinyinfer::transpose(a);

    assert(b.cols() == a.rows() && b.rows() == a.cols());

    assert(
        b(0, 0) == a(0, 0) &&
        b(0, 1) == a(1, 0) &&
        b(1, 0) == a(0, 1) &&
        b(1, 1) == a(1, 1) &&
        b(2, 0) == a(0, 2) &&
        b(2, 1) == a(1, 2)
    );
}

int main() {
    testZeroRowOrCol();
    testNormal();
    std::cout << "All transpose passed.";
}