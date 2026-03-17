#pragma once

#include <opencv2/opencv.hpp>
#include <thread>
#include <mutex>
#include <atomic>
#include <chrono>

class Tracker {
public:
    Tracker();
    ~Tracker();

    void start(int cameraIndex = 0);
    void stop();

private:
    void captureLoop();
    void processLoop();

    cv::Mat sharedFrame;
    std::mutex frameMutex;

    cv::Ptr<cv::BackgroundSubtractorMOG2> pMOG2;

    std::thread captureThread;

    std::atomic<bool> running;

    double fps;
    std::chrono::steady_clock::time_point lastTime;
};