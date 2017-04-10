// TrafficSign.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include <string>
#include "TrafficSingDetector.h"
#include "HogSvmClassifier.h"
#include <iostream>
#include <time.h>
#include <fstream>

//////////////////////////////////////////////////////////////////////////
//要载入的分类器名称，TS_CLASSIFIER_NUM定义见TS_define.h
string trafficSignName[TS_CLASSIFIER_NUM] = { "注意行人","禁止停车","施工" };

//////////////////////////////////////////////////////////////////////////
//路径

struct sign
{
	Rect rect;
	uchar type;
};


/*************************************************************************/
/*用于检测交通标志*/
/*************************************************************************/
int main()
{
	TrafficSingDetector detector;
	HogSvmClassifier classifiers[TS_CLASSIFIER_NUM];

	//////////////////////////////////////////////////////////////////////////
	//读取SVM分类器
	for (unsigned int i = 0; i < TS_CLASSIFIER_NUM; ++i)
	{
		string load_path = trafficSignName[i] + "_32.xml";
		int err = classifiers[i].svm_load(load_path);
		cout << load_path << ": " << err << endl;
	}

	//////////////////////////////////////////////////////////////////////////
	//读取图片
	Mat img = imread("test.jpg", 1);
	resize(img, img, Size(960, 640), CV_INTER_CUBIC);

	//////////////////////////////////////////////////////////////////////////
	//检测
	vector<Rect> candidates;
	detector.saturation_detect(&img, &candidates, 120);

	//////////////////////////////////////////////////////////////////////////
	//识别
	vector<sign> traffic_signs;
	float result = 0.0;
	for (unsigned int i = 0; i < candidates.size(); ++i)
	{
		Mat dete_img = img(candidates[i]);
		for (unsigned int j = 0; j < TS_CLASSIFIER_NUM; ++j)
		{
			result = 0.0;
			classifiers[j].predict(&dete_img, &result);
			if (result == 1.0)
			{
				sign s;
				s.rect = candidates[i];
				s.type = (uchar)j;
				traffic_signs.push_back(s);
			}
		}
	}

	//////////////////////////////////////////////////////////////////////////
	//框选及显示
	for (unsigned int i = 0; i < traffic_signs.size(); ++i)
	{
		rectangle(img, traffic_signs[i].rect, Scalar(0, 255, 0), 2);
		cout << trafficSignName[traffic_signs[i].type] << ": " << traffic_signs[i].rect.x << " " 
			<< traffic_signs[i].rect.y << " " << traffic_signs[i].rect.width << " " << traffic_signs[i].rect.height << endl;
	}	
	imshow("result", img);
	waitKey(0);
	destroyAllWindows();

	return 0;
}