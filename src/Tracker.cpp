#include "Tracker.hpp"
#include <iostream>

Tracker::Tracker()
    : running(false), fps(0.0)
{
    pMOG2 = cv::createBackgroundSubtractorMOG2(300, 25, false);
}

Tracker::~Tracker()
{
    stop();
}

void Tracker::start(int cameraIndex)
{
    running = true;
    captureThread = std::thread(&Tracker::captureLoop, this);
    processLoop(); // run process on main thread so waitKey works
    captureThread.join();
}

void Tracker::stop()
{
    running = false;
}

void Tracker::captureLoop()
{
    cv::VideoCapture cap(0);

    if (!cap.isOpened()) {
        std::cerr << "Error: cannot open camera" << std::endl;
        running = false;
        return;
    }

    while (running) {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) continue;

        {
            std::lock_guard<std::mutex> lock(frameMutex);
            sharedFrame = frame.clone();
        }
    }

    cap.release();
}

void Tracker::processLoop()
{
    // wait until first frame arrives
    while (running) {
        {
            std::lock_guard<std::mutex> lock(frameMutex);
            if (!sharedFrame.empty()) break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    lastTime = std::chrono::steady_clock::now();
    int objectIdCounter = 0;

    // track previous centroids to assign consistent IDs
    std::vector<cv::Point2f> prevCentroids;
    std::vector<int> objectIDs;

    while (running) {
        cv::Mat frame;

        {
            std::lock_guard<std::mutex> lock(frameMutex);
            if (sharedFrame.empty()) continue;
            frame = sharedFrame.clone();
        }

        // FPS
        auto now = std::chrono::steady_clock::now();
        double delta = std::chrono::duration<double>(now - lastTime).count();
        lastTime = now;
        if (delta > 0) fps = 0.9 * fps + 0.1 * (1.0 / delta); // smoothed FPS

        // MOG2 foreground mask
        cv::Mat fgMask;
        pMOG2->apply(frame, fgMask);

        // clean mask
        cv::GaussianBlur(fgMask, fgMask, cv::Size(5, 5), 0);
        cv::erode(fgMask,  fgMask, cv::Mat(), cv::Point(-1,-1), 1);
        cv::dilate(fgMask, fgMask, cv::Mat(), cv::Point(-1,-1), 3);

        // find contours
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(fgMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        // build current detections
        std::vector<cv::Rect>    boxes;
        std::vector<cv::Point2f> centroids;

       // step 1 — collect all valid contour points into one pool
std::vector<cv::Point> allPoints;
for (const auto& contour : contours) {
    if (cv::contourArea(contour) < 300) continue;
    allPoints.insert(allPoints.end(), contour.begin(), contour.end());
}

// step 2 — if we have points, merge them into ONE bounding box
if (!allPoints.empty()) {
    // group nearby contours using dilated mask
    cv::Mat merged = cv::Mat::zeros(fgMask.size(), CV_8UC1);
    cv::drawContours(merged,contours,-1,255,cv::FILLED);
    cv::dilate(merged, merged, cv::Mat(), cv::Point(-1,-1), 20);

    // find contours on merged mask
    std::vector<std::vector<cv::Point>> mergedContours;
    cv::findContours(merged, mergedContours,
                     cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    for (const auto& mc : mergedContours) {
        if (cv::contourArea(mc) < 2000) continue;

        cv::Rect bbox = cv::boundingRect(mc);

        // padding
        int pad = 12;
        bbox.x      = std::max(0, bbox.x - pad);
        bbox.y      = std::max(0, bbox.y - pad);
        bbox.width  = std::min(frame.cols - bbox.x, bbox.width  + 2 * pad);
        bbox.height = std::min(frame.rows - bbox.y, bbox.height + 2 * pad);

        boxes.push_back(bbox);
        centroids.push_back(cv::Point2f(
            bbox.x + bbox.width  / 2.0f,
            bbox.y + bbox.height / 2.0f
        ));
    }
}

        // assign IDs by matching centroids to previous frame
        std::vector<int> assignedIDs(centroids.size(), -1);

        for (int i = 0; i < (int)centroids.size(); i++) {
            float minDist = 80.0f; // max distance to consider same object
            int   bestJ   = -1;

            for (int j = 0; j < (int)prevCentroids.size(); j++) {
                float d = cv::norm(centroids[i] - prevCentroids[j]);
                if (d < minDist) {
                    minDist = d;
                    bestJ   = j;
                }
            }

            if (bestJ >= 0) {
                assignedIDs[i] = objectIDs[bestJ];
            } else {
                assignedIDs[i] = objectIdCounter++;
            }
        }

        prevCentroids = centroids;
        objectIDs     = assignedIDs;

        // draw — dark overlay panel at top for FPS
        cv::Mat overlay = frame.clone();
        cv::rectangle(overlay, cv::Rect(0, 0, frame.cols, 52),
                      cv::Scalar(0, 0, 0), cv::FILLED);
        cv::addWeighted(overlay, 0.5, frame, 0.5, 0, frame);

        // FPS text
        std::string fpsText = "FPS: " + std::to_string((int)fps);
        cv::putText(frame, fpsText, cv::Point(16, 34),
                    cv::FONT_HERSHEY_SIMPLEX, 0.9,
                    cv::Scalar(0, 255, 120), 2, cv::LINE_AA);

        // object count
        std::string countText = "Objects: " + std::to_string(boxes.size());
        cv::putText(frame, countText, cv::Point(160, 34),
                    cv::FONT_HERSHEY_SIMPLEX, 0.9,
                    cv::Scalar(0, 220, 255), 2, cv::LINE_AA);

        // draw each object box with ID label
        for (int i = 0; i < (int)boxes.size(); i++) {
            cv::Scalar color(0, 255, 80); // green

            // draw corner brackets instead of full rectangle — looks cleaner
            int x = boxes[i].x, y = boxes[i].y;
            int w = boxes[i].width, h = boxes[i].height;
            int cLen = std::min(20, std::min(w, h) / 3); // corner length

            // top-left
            cv::line(frame, {x, y},        {x + cLen, y},        color, 2, cv::LINE_AA);
            cv::line(frame, {x, y},        {x, y + cLen},        color, 2, cv::LINE_AA);
            // top-right
            cv::line(frame, {x+w, y},      {x+w-cLen, y},        color, 2, cv::LINE_AA);
            cv::line(frame, {x+w, y},      {x+w, y+cLen},        color, 2, cv::LINE_AA);
            // bottom-left
            cv::line(frame, {x, y+h},      {x+cLen, y+h},        color, 2, cv::LINE_AA);
            cv::line(frame, {x, y+h},      {x, y+h-cLen},        color, 2, cv::LINE_AA);
            // bottom-right
            cv::line(frame, {x+w, y+h},    {x+w-cLen, y+h},      color, 2, cv::LINE_AA);
            cv::line(frame, {x+w, y+h},    {x+w, y+h-cLen},      color, 2, cv::LINE_AA);

            // ID label with background pill
            std::string label = "ID " + std::to_string(assignedIDs[i]);
            int baseline = 0;
            cv::Size tSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX,
                                             0.55, 1, &baseline);
            cv::Rect labelBg(x, y - tSize.height - 10,
                             tSize.width + 10, tSize.height + 8);
            if (labelBg.y >= 0)
                cv::rectangle(frame, labelBg, color, cv::FILLED);

            cv::putText(frame, label,
                        cv::Point(x + 5, y - 5),
                        cv::FONT_HERSHEY_SIMPLEX, 0.55,
                        cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
        }

        cv::imshow("Real-Time Object Tracker", frame);

        //// Q or ESC to quit — works on main thread
        //int key = cv::waitKey(1) & 0xFF;
        //if (key == 'q' || key == 'Q' || key == 27) {
        //    stop();
        //    break;
        //}

        int key = cv::waitKey(30) & 0xFF;
        if (key == 'q' || key == 'Q' || key == 27) {
            stop();
            break;
        }
        // also quit if window is closed with X button
        if (cv::getWindowProperty("Real-Time Object Tracker",
            cv::WND_PROP_VISIBLE) < 1) {
            stop();
            break;
        }
    }

    cv::destroyAllWindows();
}