#include "adaptels.h"

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::printf("[usage] AdaptelsMain.exe <image path> <information threshold> <color=1>\n");
        return -1;
    }
    const bool color = std::atoi(argv[3]) != 0;
    cv::Mat img = cv::imread(argv[1], color ? 1 : 0);

    cv::Mat img_32fc1;
    img.convertTo(img_32fc1, CV_32F, 1.0/255.0, 0.0);

    cv::Mat img_32fc1_input; // gray or L*ab
    cv::Mat img_32fc1_rgb; // RGB color
    if (color) {
        img_32fc1_rgb = img_32fc1.clone();
        cv::cvtColor(img_32fc1, img_32fc1_input, CV_BGR2Lab);
        img_32fc1_input /= 128.0;
    } else {
        cv::cvtColor(img_32fc1, img_32fc1_rgb, CV_GRAY2BGR);
        img_32fc1_input = img_32fc1;
    }

    const double T = std::atof(argv[2]);
    adaptel::timer t;
    cv::Mat label = adaptel::AdaptelSuperPixel(img_32fc1_input, T);
    std::printf("Adaptel %f ms\n", t.stop_us() * 1.0e-3);

    cv::imshow("original", img_32fc1);
    cv::imshow("label", adaptel::ShuffleAndVisualizeLabel(label));
    cv::imshow("boundary", adaptel::DrawLabelBorder(img_32fc1_rgb, label));
    cv::waitKey(0);

    return 0;
}
