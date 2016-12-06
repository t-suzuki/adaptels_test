#ifndef __ADAPTELS_H__
#define __ADAPTELS_H__
#include <chrono>
#include <opencv2/opencv.hpp>

namespace adaptel {

struct timer {
    timer() { start(); }
    void start() { t = std::chrono::system_clock::now(); }
    double stop_us() { 
        auto us = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - t).count();
        start();
        return double(us);
    }
    std::chrono::time_point<std::chrono::system_clock> t;
};

cv::Mat AdaptelSuperPixel(cv::Mat input_image, double information_threshold);

cv::Mat ShuffleAndVisualizeLabel(cv::Mat label_32sc1);

cv::Mat DrawLabelBorder(cv::Mat input_image, cv::Mat label_32sc1);

}

#endif // __ADAPTELS_H__
