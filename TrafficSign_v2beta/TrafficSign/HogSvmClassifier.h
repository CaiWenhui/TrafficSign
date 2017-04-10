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
//ͼ��&���ݲ���
private:
	unsigned int sample_num;  //��������
	unsigned int imgHeight;   //ͼ��߶�
	unsigned int imgWidth;    //ͼ����
	CvMat *feature_mat;         //��������
	CvMat *label_mat;           //��ǩ����
	//CvMat *pred_mat;
//////////////////////////////////////////////////////////////////////////
//����&����������
private:
	HOGDescriptor *hog;       //HOG����������
	size_t hog_size;    //HOG����ά��
	CvSVM svm;                //SVM����
	CvSVMParams param;        //SVM��������
//////////////////////////////////////////////////////////////////////////
//��־λ
private:
	int featureDataPreparedFlag;    //����������ɱ�־
	int labelDataPreparedFlag;      //��ǩ������ɱ�־
	int svmTraindFlag;              //SVMѵ����ɱ�־


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