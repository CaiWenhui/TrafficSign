#include "stdafx.h"
#include "HogSvmClassifier.h"
#include <assert.h>
#include <vector>
#include <fstream>

#include <iostream>

HogSvmClassifier::HogSvmClassifier()
{
	sample_num = 0;
	imgHeight = 32;
	imgWidth = 32;
	featureDataPreparedFlag = 0;
	labelDataPreparedFlag = 0;
	svmTraindFlag = 0;
	hog = new HOGDescriptor(Size(imgWidth, imgHeight), Size(16, 16), Size(8, 8), Size(8, 8), 9);
	hog_size = hog->getDescriptorSize();
	param.kernel_type = CvSVM::RBF;
	param.svm_type = CvSVM::C_SVC;
	param.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 100, 0.000001);
	//pred_mat = cvCreateMat(1, (int)hog_size, CV_32FC1);

}



HogSvmClassifier::~HogSvmClassifier()
{
	delete hog;
	//cvReleaseMat(&feature_mat);
	//cvReleaseMat(&label_mat);
	//cvReleaseMat(&pred_mat);
	//if (!feature_mat->empty())
		//cvSetZero(feature_mat);
	//feature_mat->release();

	//if (!label_mat->empty())
		//cvSetZero(feature_mat);
	//label_mat->release();
}



/**************************************************************************************/
/* 功能：通过TXT读取图像标签，若标签已读取则更新 */
/* TXT中：第一行为处理图片的个数，第二个开始，每行为对应序列图片的标签  */
/* 返回值：
		  0：执行成功
		  -1: 读取标签个数失败
		  -2：标签个数与图片个数不匹配
*/
/**************************************************************************************/
int HogSvmClassifier::load_labels(string path)
{
	//////////////////////////////////////////////////////////////////////////
	//读取标签个数，若已读取图片，判断图片个数是否与标签匹配
	ifstream txtFile(path.c_str());
	unsigned int num = 0;
	try
	{
		txtFile >> num;
	}
	catch (const std::exception&)
	{
		txtFile.close();
		return -1;
	}
	if (featureDataPreparedFlag)
		if (sample_num != num)
		{
			txtFile.close();
			return -2;
		}		
	//////////////////////////////////////////////////////////////////////////
	//判断是否为第一次读取，并确定是否设置样本数量
	if (!labelDataPreparedFlag)
	{
		if (!featureDataPreparedFlag)
			sample_num = num;
		label_mat = cvCreateMat(sample_num, 1, CV_32FC1);
	}
	//////////////////////////////////////////////////////////////////////////
	//读取标签
	cout << "Read  Labels..." << endl;
	for (unsigned int i = 0; i < num; ++i)
	{
		cout << i << endl;
		//txtFile >> label_mat->data[i];
		float k;
		txtFile >> k;
		cvmSet(label_mat, i, 0, k);
	}
		
	//////////////////////////////////////////////////////////////////////////
	//设置标记位并关闭文件流
	labelDataPreparedFlag = 1;
	txtFile.close();

	return 0;
}



/**************************************************************************************/
/* 功能：输入图片，提取其HOG特征  */
/* 返回值：
		0：执行成功  */
/**************************************************************************************/
int HogSvmClassifier::hog_featere(Mat *img, int j)
{
	//////////////////////////////////////////////////////////////////////////
	//检查图像大小是否匹配
	if (img->cols != imgWidth || img->rows != imgHeight)
		resize(*img, *img, Size(imgWidth, imgHeight), INTER_CUBIC);
	//////////////////////////////////////////////////////////////////////////
	//获取Hog特征
	vector<float> descriptors((int)hog_size);
	IplImage *IplImg = &IplImage(*img);
	hog->compute(IplImg, descriptors, Size(1, 1), Size(0, 0));
	//////////////////////////////////////////////////////////////////////////
	//将特征值保存入矩阵
	for (vector<float>::size_type i = 0; i < descriptors.size(); ++i)
		cvmSet(feature_mat, j, (int)i, descriptors[i]);

	return 0;
}



/**************************************************************************************/
/* 功能：用于检测的HOG特征提取  */
/* 返回值：
		  0：执行成功  */
/**************************************************************************************/
int HogSvmClassifier::hog_featere_pred(Mat *img, Mat *pred_mat)
{
	//////////////////////////////////////////////////////////////////////////
	//检查图像大小是否匹配
	if (img->cols != imgWidth || img->rows != imgHeight)
		resize(*img, *img, Size(imgWidth, imgHeight), INTER_CUBIC);
	//////////////////////////////////////////////////////////////////////////
	//获取Hog特征
	vector<float> descriptors((int)hog_size);
	IplImage *IplImg = &IplImage(*img);
	hog->compute(IplImg, descriptors, Size(1, 1), Size(0, 0));
	//////////////////////////////////////////////////////////////////////////
	//将特征值保存入矩阵
	float *data = (float*)pred_mat->data;
	for (vector<float>::size_type i = 0; i < descriptors.size(); ++i)
		data[i] = descriptors[i];

	//descriptors.clear();
	//cvReleaseImage(&IplImg);
	return 0;
}



/*************************************************************************************/
/* 功能：根据txt保存的路径，批量提取图片HOG特征  */
/* 返回值：
		  0：执行成功
		  -1: 读取个数失败
		  -2: 图像个数与标签个数不匹配
		  -3: 图像读取失败
          -4：其他未知异常
LoadType：读取方式，详见TS_define.h
ErrNum：导致失败的路径序号  */
/*************************************************************************************/
int HogSvmClassifier::hog_featere_with_path(string txt_path, unsigned int *ErrNum, int LoadType)
{
	*ErrNum = 0;
	unsigned int num = 0;
	string img_path; 
	//////////////////////////////////////////////////////////////////////////
	//读取图片个数，若已读取标签，判断图片个数是否与标签匹配
	ifstream txtFile(txt_path.c_str());
	try
	{
		txtFile >> num;
	}
	catch (const std::exception& )
	{
		txtFile.close();
		return -1;
	}
	if (labelDataPreparedFlag)
		if (label_mat->rows != num)
		{
			txtFile.close();
			return -2;
		}
	//////////////////////////////////////////////////////////////////////////
	//为特征数据/标签申请空间并确定样本数量
	if (!featureDataPreparedFlag)
	{
		if (!labelDataPreparedFlag)
			sample_num = num;
		feature_mat = cvCreateMat(sample_num, (int)hog_size, CV_32FC1);
	}
	if (LoadType == HOG_LOAD_TYPE_TOG)
		if (!labelDataPreparedFlag)
		{
			label_mat = cvCreateMat(sample_num, 1, CV_32FC1);
		}
	//////////////////////////////////////////////////////////////////////////
	//读取图片并计算HOG值
	float *data = NULL;
	cout << "Start Compute HOG Features ..." << endl;
	try
	{
		for (unsigned int i = 0; i < num; ++i)
		{
			cout << i << endl;
			*ErrNum = i + 1;
			txtFile >> img_path;
			if (LoadType == HOG_LOAD_TYPE_TOG)
			{
				float k;
				txtFile >> k;
				cvmSet(label_mat, i, 0, k);
				//txtFile >> label_mat->data[i];
			}
			Mat img = imread(img_path.c_str(), 1);
			if (img.empty())
			{
				txtFile.close();
				return -3;
			}
			hog_featere(&img, i);
		}
	}
	catch (const std::exception&)
	{
		txtFile.close();
		return -4;
	}
	//////////////////////////////////////////////////////////////////////////
	//设置标志位并关闭文件流
	featureDataPreparedFlag = 1;
	if (LoadType == HOG_LOAD_TYPE_TOG)
		labelDataPreparedFlag = 1;
	txtFile.close();

	return 0;
}



/**************************************************************************************/
/* 功能：进行SVM训练  */
/* 返回值：
		0：执行成功
		-1: 已存在SVM模型
		-2: 参数优化网络不匹配 
		-3: 分类器训练失败  */
/**************************************************************************************/
int HogSvmClassifier::svm_train()
{
	if (svmTraindFlag)
		return -1;
	//////////////////////////////////////////////////////////////////////////
	//设置参数优化网络
	CvParamGrid CvParamGrid_C(pow(2.0, -5), pow(2.0, 15), pow(2.0, 2));
	CvParamGrid CvParamGrid_gamma(pow(2.0, -15), pow(2.0, 3), pow(2.0, 2));
	if (!CvParamGrid_C.check() || !CvParamGrid_gamma.check())
		return - 2;
	//////////////////////////////////////////////////////////////////////////
	//开始训练
	cout << "Start Train SVM Module ..." << endl;
	try
	{
		svm.train_auto(feature_mat, label_mat, Mat(), Mat(), param, 10, CvParamGrid_C, CvParamGrid_gamma,
			CvSVM::get_default_grid(CvSVM::P), CvSVM::get_default_grid(CvSVM::NU), CvSVM::get_default_grid(CvSVM::COEF), CvSVM::get_default_grid(CvSVM::DEGREE), true);
	}
	catch (const std::exception&)
	{
		return -3;
	}
	//////////////////////////////////////////////////////////////////////////
	//设置训练完成标志位
	cout << "SVM Module Training Completed ! " << endl;
	svmTraindFlag = 1;

	return 0;
}



/**************************************************************************************/
/* 功能：进行完整的HOG特征提取 + SVM训练  */
/* 图片路径和标签被分别写在两个txt文件中，文件开头第一行均为样本个数  */
/* 返回值：
		  0：执行成功
		  -1: 读取个数失败
		  -2: 图像个数与标签个数不匹配
		  -3: 图像读取失败
		  -4：其他未知异常
		  -5：已存在SVM模型
		  -6: SVM参数优化网络不匹配
		  -7: SVM分类器训练失败  */
/**************************************************************************************/
int HogSvmClassifier::train(string img_txt_paths, string label_txt_path, unsigned int *ErrNum)
{
	int res = 0;
	if (res = hog_featere_with_path(img_txt_paths, ErrNum, HOG_LOAD_TYPE_SEP) != 0)
		return res;
	if (res = load_labels(label_txt_path) != 0)
		return res;
	if (labelDataPreparedFlag && featureDataPreparedFlag)
		if (res = svm_train() != 0)
			return res - 4;

	return 0;
}



/**************************************************************************************/
/* 功能：进行完整的HOG特征提取 + SVM训练  */
/* 图片路径和标签写在一个txt文件中，文件开头第一行为样本个数，图片路径后跟图片标签  */
/* 返回值：
		  0：执行成功
		  -1: 读取个数失败
		  -2: 图像个数与标签个数不匹配
		  -3: 图像读取失败
		  -4：其他未知异常
		  -5：已存在SVM模型
		  -6: SVM参数优化网络不匹配
		  -7: SVM分类器训练失败  */
/**************************************************************************************/
int HogSvmClassifier::train(string txt_path, unsigned int *ErrNum)
{
	int res = 0;
	if (res = hog_featere_with_path(txt_path, ErrNum, HOG_LOAD_TYPE_TOG) != 0)
		return res;
	if (featureDataPreparedFlag&&labelDataPreparedFlag)
		if (res = svm_train() != 0)
			return res - 5;

	return 0;
}



/**************************************************************************************/
/* 功能：对输入图片进行分类  */
/* 返回值：
		  0：执行成功
		  -1：无SVM模型（未训练或未载入） */
/**************************************************************************************/
int HogSvmClassifier::predict(Mat *img, float *class_type)
{
	int res = 0;
	if (svmTraindFlag)
	{
		Mat *pred_mat = new Mat(1, (int)hog_size, CV_32FC1);
		res = hog_featere_pred(img, pred_mat);
		*class_type = svm.predict(*pred_mat);
		delete pred_mat;
	}
	else
		return -1;

	return 0;
}



/**************************************************************************************/
/* 功能：保存目前使用的SVM模型（XML文件）  */
/* 返回值：
		  0：执行成功
		  -1：无SVM模型（未训练或未载入） */
/**************************************************************************************/
int HogSvmClassifier::svm_save(string path)
{
	if (svmTraindFlag)
		svm.save(path.c_str());
	else
		return -1;

	return 0;
}



/**************************************************************************************/
/* 功能：载入SVM模型到类（XML文件）  */
/* 返回值：
		  0：执行成功  
		  -1: 已存在SVM模型  */
/**************************************************************************************/
int HogSvmClassifier::svm_load(string path)
{
	if (svmTraindFlag)
		return -1;
	svm.load(path.c_str());
	svmTraindFlag = 1;

	return 0;
}



int HogSvmClassifier::print_info()
{
	return 0;
}



int HogSvmClassifier::change_info()
{
	return 0;
}