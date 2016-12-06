#include "adaptels.h"
#include <queue>
#include <vector>
#include <random>

namespace adaptel {

template <typename Pixel>
struct information_calculator_type {
    double info = 0.0;
    Pixel sum = 0;
    Pixel mu = 0;
    int count = 0;

    void add_pixel(Pixel value) {
        sum += value;
        ++count;
        mu = Pixel(sum / double(count));
        info = probe_info(value);
    }

    // Laplacian distribution on euclidean distance.
    // p(v) = exp(-sqrt(<(x-u)', (x-u)> / sigma^2))
    // info(v) = - log(p(v))
    //         = sqrt(<(x-u)', (x-u)> / sigma^2))
    // here, we assume mu does not significantly change over time and calculate the APPROXIMATE additional information caused by a new pixel.
    // i.e. mu is not re-calculated for previous values.
    template <typename U>
    typename std::enable_if<!std::is_scalar<U>::value, double>::type probe_info(U value) {
        return info + std::sqrt((value - mu).dot(value - mu)) * 2.0;
    }
    template <typename U>
    typename std::enable_if<std::is_scalar<U>::value, double>::type probe_info(U value) {
        return info + std::abs(value - mu) * 2.0;
    }
};

struct item_type {
    float info;
    cv::Point pos;
    bool operator<(const item_type& rhs) const {
        return info > rhs.info; // reversed order of info.
    }
};

template <typename Pixel>
inline void AdaptelGrow(
    const cv::Mat input_image, cv::Mat mask, cv::Mat least_information, double information_threshold, cv::Point seed, size_t* candidate_count, size_t* adaptel_count) {
    information_calculator_type<Pixel> information_calculator;

    std::priority_queue<item_type> candidates;

    candidates.push({0.0, seed});
    while (!candidates.empty()) {
        item_type item = candidates.top();
        candidates.pop();
        if (mask.at<uint8_t>(item.pos) == 0) {
            // add the pixel to this adaptel.
            ++*adaptel_count;
            mask.at<uint8_t>(item.pos) = 1;
            information_calculator.add_pixel(input_image.at<Pixel>(item.pos));
            least_information.at<float>(item.pos) = float(information_calculator.info);
            // add nearby pixels to the candidates.
            {
                const int dy[] = {-1, 0, 0, 1};
                const int dx[] = { 0,-1, 1, 0};
                const int h = input_image.rows;
                const int w = input_image.cols;
                for (int k = 0; k < 4; ++k) {
                    const int py = item.pos.y + dy[k];
                    const int px = item.pos.x + dx[k];
                    if (0 <= py && py < h && 0 <= px && px < w && mask.at<uint8_t>(py, px) == 0) {
                        const float probe = float(information_calculator.probe_info(input_image.at<Pixel>(py, px)));
                        if (probe < information_threshold && probe < least_information.at<float>(py, px)) {
                            candidates.push({probe, {px, py}});
                            ++*candidate_count;
                        }
                    }
                }
            }
        }
    }
}

struct SNextSeed {
    SNextSeed(int rows, int cols)
        : mask(cv::Mat::zeros(rows, cols, CV_8UC1))
        , shell(cv::Mat::zeros(rows, cols, CV_8UC1))
        , candidates()
        , rng(1)
    {
        candidates.reserve(rows * cols);
    }

    // find a boundary pixel in (label != 0) image.
    bool NextSeed(const cv::Mat& label, cv::Point* seed, bool random_select) {
        mask = (label != 0);
        cv::dilate(mask, shell, cv::Mat());
        shell -= mask;

        candidates.clear();
        for (int y = 0; y < label.rows; ++y) {
            for (int x = 0; x < label.cols; ++x) {
                if (shell.at<uint8_t>(y, x) != 0) {
                    if (random_select) {
                        candidates.emplace_back(x, y);
                    } else {
                        *seed = {x, y};
                        return true;
                    }
                }
            }
        }
        if (candidates.empty()) {
            return false;
        }
        std::shuffle(candidates.begin(), candidates.end(), rng);
        *seed = candidates.front();
        return true;
    }

private:
    cv::Mat mask;
    cv::Mat shell;
    std::vector<cv::Point> candidates;
    std::mt19937_64 rng;
};


cv::Mat AdaptelSuperPixel(const cv::Mat input_image, double information_threshold) {
    constexpr bool verbose = false;
    const int h = input_image.rows;
    const int w = input_image.cols;
    cv::Point seed {h / 2, w / 2};

    cv::Mat label = cv::Mat::zeros(h, w, CV_32SC1);
    cv::Mat mask = cv::Mat::zeros(h, w, CV_8UC1);
    cv::Mat least_information = cv::Mat::zeros(h, w, CV_32FC1) + std::numeric_limits<float>::max();
    SNextSeed find_next_seed(h, w);

    timer t_total;
    size_t total_candidate_count = 0;
    size_t total_adaptel_count = 0;
    for (int ilabel = 1; ; ++ilabel) {
        timer t_iter;
        size_t adaptel_count = 0;
        size_t candidate_count = 0;
        mask.setTo(0);
        if (input_image.type() == CV_8UC1) {
            AdaptelGrow<uint8_t>(input_image, mask, least_information, information_threshold, seed, &candidate_count, &adaptel_count);
        } else if (input_image.type() == CV_8UC3) {
            AdaptelGrow<cv::Vec3b>(input_image, mask, least_information, information_threshold, seed, &candidate_count, &adaptel_count);
        } else if (input_image.type() == CV_32FC1) {
            AdaptelGrow<float>(input_image, mask, least_information, information_threshold, seed, &candidate_count, &adaptel_count);
        } else if (input_image.type() == CV_32FC3) {
            AdaptelGrow<cv::Vec3f>(input_image, mask, least_information, information_threshold, seed, &candidate_count, &adaptel_count);
        } else {
            break;
        }
        if (verbose) {
            std::printf("label=%4d, count=%5I64d, final=%5I64d (%lf us)\n", ilabel, candidate_count, adaptel_count, t_iter.stop_us());
        }
        total_adaptel_count += adaptel_count;
        total_candidate_count += candidate_count;
        label.setTo(ilabel, mask);

        timer t_seed;
        if (!find_next_seed.NextSeed(label, &seed, true)) {
            break;
        }
        if (verbose) {
            std::printf("seed (%lf us)\n", t_seed.stop_us());
        }
    }
    if (verbose) {
        std::printf("total count=%5I64d, final=%5I64d (%lf us)\n", total_candidate_count, total_adaptel_count, t_total.stop_us());
    }

    return label;
}


cv::Mat ShuffleAndVisualizeLabel(cv::Mat label_32sc1) {
    cv::Mat visualize(label_32sc1.rows, label_32sc1.cols, CV_8UC3);
    for (int y = 0; y < label_32sc1.rows; ++y) {
        for (int x = 0; x < label_32sc1.cols; ++x) {
            auto& p = visualize.at<cv::Vec3b>(y, x);
            int z = label_32sc1.at<int32_t>(y, x);
            z = (925 * z + 711) % 256;
            p[0] = z;
            z = (925 * z + 711) % 256;
            p[1] = z;
            z = (925 * z + 711) % 256;
            p[2] = z;
        }
    }
    return visualize;
}

cv::Mat DrawLabelBorder(cv::Mat input_image, cv::Mat label_32sc1) {
    cv::Mat visualize = input_image.clone();
    for (int y = 0; y < label_32sc1.rows; ++y) {
        for (int x = 0; x < label_32sc1.cols; ++x) {
            bool border = false;
            if (x + 1 < label_32sc1.cols && label_32sc1.at<int32_t>(y, x) != label_32sc1.at<int32_t>(y, x + 1)) {
                border = true;
            }
            if (y + 1 < label_32sc1.cols && label_32sc1.at<int32_t>(y, x) != label_32sc1.at<int32_t>(y + 1, x)) {
                border = true;
            }
            if (border) {
                auto& p = visualize.at<cv::Vec3f>(y, x);
                p[0] = 1.0f - p[0];
                p[1] = 1.0f - p[1];
                p[2] = 1.0f - p[2];
            }

        }
    }
    return visualize;
}

}
