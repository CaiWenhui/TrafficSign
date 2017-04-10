#pragma once

#include "opencv.hpp"
using namespace cv;

class TrafficSingDetector
{
public:
	TrafficSingDetector();
	~TrafficSingDetector();
private:
	Size img_size;

public:
	int saturation_detect(Mat *img, vector<Rect> *signs, int thresh_value);

};

