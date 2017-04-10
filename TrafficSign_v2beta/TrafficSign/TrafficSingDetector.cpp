#include "stdafx.h"
#include "TrafficSingDetector.h"
#include <iostream>

TrafficSingDetector::TrafficSingDetector()
{
	img_size.width = 0;
	img_size.height = 0;
}


TrafficSingDetector::~TrafficSingDetector()
{
}

/**************************************************************************************/
/* 功能：使用饱和度进行交通标志检测 */
/* 返回值：
			0：执行成功  */
/**************************************************************************************/
int TrafficSingDetector::saturation_detect(Mat *img, vector<Rect> *signs, int  thresh_value)
{
	//////////////////////////////////////////////////////////////////////////
	//检测并确定图像大小
	if (img_size.width == 0 || img_size.height == 0)
	{
		img_size.width = img->cols;
		img_size.height = img->rows;
	}
	else if (img_size.width != img->cols || img_size.height != img->rows)
	{
		img_size.width = img->cols;
		img_size.height = img->rows;
	}
		//resize(*img, *img, img_size, INTER_CUBIC);

	//////////////////////////////////////////////////////////////////////////
	//清空vector
	if (!signs->empty())
		signs->clear();

	//////////////////////////////////////////////////////////////////////////
	//色彩空间转换及通道分离
	Mat hsv_img = img->clone();
	Mat hsv[3];
	cvtColor(hsv_img, hsv_img, CV_RGB2HSV);
	split(hsv_img, hsv);

	//////////////////////////////////////////////////////////////////////////
	//阈值化、中值滤波与区域检测
	Mat s_th;
	threshold(hsv[1], s_th, thresh_value, 255, CV_THRESH_BINARY);

	int kernel_size = (int)(((img_size.width + img_size.height) / 320));
	if (kernel_size % 2 != 1)
		kernel_size++;
	medianBlur(s_th, s_th, kernel_size);

	//imshow("sh", s_th);
	//waitKey(0);
	//destroyAllWindows();
	//////////////////////////////////////////////////////////////////////////
	//区域检测
	CvMat con = s_th;
	CvMemStorage* storage = cvCreateMemStorage(0);
	CvSeq * pcontour = 0;
	cvFindContours(&con, storage, &pcontour, sizeof(CvContour), CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE, cvPoint(0, 0));
	for (; pcontour != 0; pcontour = pcontour->h_next)
	{
		Rect r = ((CvContour*)pcontour)->rect;
		Mat v_roi = hsv[2](r);
		Scalar v_avg = mean(v_roi);
		int extend_pix = 0;
		if (r.height > 10 && r.width > 10 
			&& v_avg[0] > 50
			)
		{
			extend_pix = (r.width + r.height) / 40;
			(r.x - extend_pix) < 0 ? r.x = 0 : r.x -= extend_pix;
			(r.y - extend_pix) < 0 ? r.y = 0 : r.y -= extend_pix;
			(r.width + r.x + extend_pix * 2 > img_size.width) ? r.width = img_size.width - r.x : r.width += extend_pix * 2;
			(r.height + r.y + extend_pix * 2 > img_size.height) ? r.height = img_size.height - r.y : r.height += extend_pix * 2;
			signs->push_back(r);
			//rectangle(*img, r, Scalar(0, 255, 0), 2);
		}
	}

	cvReleaseMemStorage(&storage);
	return 0;
}