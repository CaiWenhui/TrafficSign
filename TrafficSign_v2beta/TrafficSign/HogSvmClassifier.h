#pragma once

//#include "opencv_header.h"
#include "opencv.hpp"
#include "TS_define.h"
#include <string>
using namespace std;
using namespace cv;
class HogSvmClassifier
{

public:
	HogSvmClassifier();
	~HogSvmClassifier();

//////////////////////////////////////////////////////////////////////////
//图像&数据参数
private:
	unsigned int sample_num;  //样本数量
	unsigned int imgHeight;   //图像高度
	unsigned int imgWidth;    //图像宽度
	CvMat *feature_mat;         //特征矩阵
	CvMat *label_mat;           //标签矩阵
	//CvMat *pred_mat;
//////////////////////////////////////////////////////////////////////////
//特征&分类器参数
private:
	HOGDescriptor *hog;       //HOG特征描述子
	size_t hog_size;    //HOG特征维数
	CvSVM svm;                //SVM对象
	CvSVMParams param;        //SVM参数对象
//////////////////////////////////////////////////////////////////////////
//标志位
private:
	int featureDataPreparedFlag;    //特征数据完成标志
	int labelDataPreparedFlag;      //标签数据完成标志
	int svmTraindFlag;              //SVM训练完成标志


public:
	int train(string txt_path, unsigned int *ErrNum);
	int train(string img_txt_paths, string label_txt_path, unsigned int *ErrNum);
	int predict(Mat *img, float *class_type);
	int svm_save(string path);
	int svm_load(string path);
	int print_info();
	int change_info();

private:
	int load_labels(string path);
	int hog_featere(Mat *img, int j);
	int hog_featere_pred(Mat *img, Mat *pred_mat);
	int hog_featere_with_path(string txt_path, unsigned int *ErrNum, int LoadType = HOG_LOAD_TYPE_SEP);
	int svm_train();
};