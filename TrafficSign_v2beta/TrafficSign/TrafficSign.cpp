// TrafficSign.cpp : �������̨Ӧ�ó������ڵ㡣
//

#include "stdafx.h"
#include <string>
#include "TrafficSingDetector.h"
#include "HogSvmClassifier.h"
#include <iostream>
#include <time.h>
#include <fstream>

//////////////////////////////////////////////////////////////////////////
//Ҫ����ķ��������ƣ�TS_CLASSIFIER_NUM�����TS_define.h
string trafficSignName[TS_CLASSIFIER_NUM] = { "ע������","��ֹͣ��","ʩ��" };

//////////////////////////////////////////////////////////////////////////
//·��

struct sign
{
	Rect rect;
	uchar type;
};


/*************************************************************************/
/*���ڼ�⽻ͨ��־*/
/*************************************************************************/
int main()
{
	TrafficSingDetector detector;
	HogSvmClassifier classifiers[TS_CLASSIFIER_NUM];

	//////////////////////////////////////////////////////////////////////////
	//��ȡSVM������
	for (unsigned int i = 0; i < TS_CLASSIFIER_NUM; ++i)
	{
		string load_path = trafficSignName[i] + "_32.xml";
		int err = classifiers[i].svm_load(load_path);
		cout << load_path << ": " << err << endl;
	}

	//////////////////////////////////////////////////////////////////////////
	//��ȡͼƬ
	Mat img = imread("test.jpg", 1);
	resize(img, img, Size(960, 640), CV_INTER_CUBIC);

	//////////////////////////////////////////////////////////////////////////
	//���
	vector<Rect> candidates;
	detector.saturation_detect(&img, &candidates, 120);

	//////////////////////////////////////////////////////////////////////////
	//ʶ��
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
	//��ѡ����ʾ
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