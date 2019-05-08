#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ximgproc.hpp>
#include <vector>
#include <string>
#include <fstream>
using namespace std;
using namespace cv;
const string IMG_PATH = "img/3.png";
const string OUT_DARK_NAME = "img/3.dark.jpg";
const string OUT_TXT_NAME = "img/3.txt";
const string OUT_IMG_NAME = "img/3.out.jpg";
const int MAX_IMG_SIZE = 500;
const int COLOR_RANGE = 255;
const int RADIUS = 7;
const float T0 = 0.1;
const float OMEGA = 0.95;
#define FAST_MIN_FILTER

float rowMin[MAX_IMG_SIZE * 2][MAX_IMG_SIZE * 2], colMin[MAX_IMG_SIZE * 2][MAX_IMG_SIZE * 2];
Mat MinFilter(Mat input, int radius) {//最小值滤波器
	int height = input.rows, width = input.cols;
	int heightExp = height + radius * 2, widthExp = width + radius * 2;
	Mat gray(heightExp, widthExp, CV_32FC1);
	Mat output(height, width, CV_32FC1);
	const int patchSize = radius * 2 + 1;

	for (int i = 0; i < heightExp; ++i) {
		for (int j = 0; j < widthExp; ++j) {
			gray.at<float>(i, j) = 1;
			if (i >= radius && j >= radius && i < height + radius && j < width + radius) {
				for (int k = 0; k < 3; ++k) {
					gray.at<float>(i, j) = min(input.at<Vec3f>(i - radius, j - radius)[k], gray.at<float>(i, j));
				}
			}
		}
	}
	double start = static_cast<double>(getTickCount());
	
#ifdef FAST_MIN_FILTER
	deque<float> que;
	for (int i = 0; i < heightExp; ++i) {
		que.clear();
		for (int j = 0; j < widthExp; ++j) {
			float inValue = gray.at<float>(i, j);
			while (!que.empty() && que.back() > inValue) {
				que.pop_back();
			}
			que.push_back(inValue);
			
			if (j >= patchSize) {
				float outValue = gray.at<float>(i, j - patchSize);
				if (outValue == que.front()) {
					que.pop_front();
				}
			}
			if (j >= patchSize - 1) {
				rowMin[i][j - patchSize + 1] = que.front();
			}
		}
	}
	for (int i = 0; i < widthExp; ++i) {
		que.clear();
		for (int j = 0; j < heightExp; ++j) {
			float inValue = rowMin[j][i];
			while (!que.empty() && que.back() > inValue) {
				que.pop_back();
			}
			que.push_back(inValue);
			if (j >= patchSize) {
				float outValue = rowMin[j - patchSize][i];
				if (outValue == que.front()) {
					que.pop_front();
				}
			}
			if (j >= patchSize - 1) {
				colMin[j - patchSize + 1][i] = que.front();
			}
		}
	}
	int cnt1 = 0, cnt2 = 0, cnt3 = 0;
	for (int i = 0; i < height; ++i) {
		for (int j = 0; j < width; ++j) {
			output.at<float>(i, j) = colMin[i][j];
			if (colMin[i][j] <= 0.1) {
				++cnt1;
			}
			if (colMin[i][j] <= 0.3) {
				++cnt2;
			}
			if (colMin[i][j] <= 0.5) {
				++cnt3;
			}
		}
	}
	//ofstream fout(OUT_TXT_NAME);
	//fout << "Dark < 0.1 Ratio = " << cnt1 * 1.0 / (height * width) << endl;
	//fout << "Dark < 0.3 Ratio = " << cnt2 * 1.0 / (height * width) << endl;
	//fout << "Dark < 0.5 Ratio = " << cnt3 * 1.0 / (height * width) << endl;
#else
	for (int i = 0; i < height; ++i) {
		for (int j = 0; j < width; ++j) {
			output.at<float>(i, j) = 1;
			for (int k = 0; k < patchSize; ++k) {
				for (int l = 0; l < patchSize; ++l) {
					if (i + k < heightExp && j + l < widthExp) {
						output.at<float>(i, j) = min(gray.at<float>(i + k, j + l), output.at<float>(i, j));
					}
				}
			}
		}
	}
#endif
	double time = ((double)getTickCount() - start) / getTickFrequency();
	//fout << "Time Used：" << time << "s" << endl;
	imshow("dark", output);//暗通道图
	//output.convertTo(output, CV_8UC1, COLOR_RANGE * 1.0);
	//imwrite(OUT_DARK_NAME, output);
	//waitKey(0);
	return output;
}
void generateDepthMap(Mat trans[3]) {//生成深度图
	int height = trans[0].rows, width = trans[0].cols;
	Mat fineColorTrans(height, width, CV_32FC3);
	for (int i = 0; i < 3; ++i) {
		for (int j = 0; j < height; ++j) {
			for (int k = 0; k < width; ++k) {
				fineColorTrans.at<Vec3f>(j, k)[i] = trans[i].at<float>(j, k);
			}
		}
	}
	Mat fineGrayTrans;
	cvtColor(fineColorTrans, fineGrayTrans, COLOR_BGR2GRAY);
	float mini = FLT_MAX, maxi = -FLT_MAX;
	for (int i = 0; i < height; ++i) {
		for (int j = 0; j < width; ++j) {
			float x = fineGrayTrans.at<float>(i, j);
			float dis = -log(x);
			maxi = max(dis, maxi);
			mini = min(dis, mini);
		}
	}

	for (int i = 0; i < height; ++i) {
		for (int j = 0; j < width; ++j) {
			float x = -log(fineGrayTrans.at<float>(i, j));
			fineGrayTrans.at<float>(i, j) = 1 - (x - mini) / (maxi - mini);
		}
	}

	Mat depthGrayMap, depthColorMap;

	fineGrayTrans.convertTo(depthGrayMap, CV_8UC1, 255);
	applyColorMap(depthGrayMap, depthColorMap, cv::COLORMAP_JET);
	printf("farthest = %f, nearest = %f\n", maxi, mini);
	imshow("depth map", depthColorMap);//深度图
	//waitKey(0);
}
void getAtm(Mat img, Mat dark, float atm[3]) {//获得全局大气光线
	vector<Point> points;
	Point brightXY[3];
	int height = img.rows, width = img.cols;
	int num = height * width * 0.001;
	for (int i = 0; i < height; ++i) {
		for (int j = 0; j < width; ++j) {
			points.push_back(Point(j, i));
		}
	}
	sort(points.begin(), points.end(), [&dark](const Point &p1, const Point &p2) {
		return dark.at<float>(p1.y, p1.x) < dark.at<float>(p2.y, p2.x);
	});
	for (int i = points.size() - num; i < points.size(); ++i) {
		for (int j = 0; j < 3; ++j) {
			if (img.at<Vec3f>(points[i].y, points[i].x)[j] > atm[j]) {
				brightXY[j] = points[i];
				atm[j] = img.at<Vec3f>(points[i].y, points[i].x)[j];
			}
		}
	}
	printf("atm = (%f, %f, %f)\n", atm[0], atm[1], atm[2]);
	for (int i = points.size() - num; i < points.size(); ++i) {
		circle(img, points[i], 2, cv::Scalar(0, 1, 0), -1);
	}
	for (int i = 0; i < 3; ++i) {
		circle(img, brightXY[i], 2, cv::Scalar(0, 0, 1), -1);
	}
}
void transProc(Mat img, float atm[3], Mat dark, Mat trans[3]) {//透射图生成和精化
	for (int i = 0; i < 3; ++i) {
		trans[i] = 1 - OMEGA * dark / atm[i];
	}
	imshow("coarse transmission ", trans[0]);//粗制透射率图
	Mat gray;
	img.convertTo(gray, CV_32FC1);
	for (int i = 0; i < 3; ++i) {
		ximgproc::guidedFilter(gray, trans[i], trans[i], RADIUS * 8, 0.01);
	}
	imshow("fine transmission", trans[0]);//精细透射率图
	//waitKey(0);
}
Mat generateDehaze(Mat img, Mat trans[3], float atm[3]) {//图像去雾
	Mat dehaze(img.rows, img.cols, CV_32FC3);
	for (int i = 0; i < img.rows; ++i) {
		for (int j = 0; j < img.cols; ++j) {
			for (int k = 0; k < 3; ++k) {
				dehaze.at<Vec3f>(i, j)[k] = (img.at<Vec3f>(i, j)[k] - atm[k]) / max(trans[k].at<float>(i, j), T0) + atm[k];
			}
		}
	}
	return dehaze;
	
}
int main() {
	Mat img = imread(IMG_PATH);
	int height = img.rows, width = img.cols;
	if (max(height, width) > MAX_IMG_SIZE) {
		if (height > width) {
			resize(img, img, Size((float)width / height * MAX_IMG_SIZE, MAX_IMG_SIZE));
		}
		else {
			resize(img, img, Size(MAX_IMG_SIZE, (float)height / width * MAX_IMG_SIZE));
		}
	}
	img.convertTo(img, CV_32FC3, 1.0 / COLOR_RANGE);
	//imshow("src", img);
	//waitKey(0);
	Mat dark = MinFilter(img, RADIUS);
	float atm[3] = { 0 };
	Mat imgCopy;
	img.copyTo(imgCopy);
	getAtm(imgCopy, dark, atm);
	Mat trans[3];
	transProc(img, atm, dark, trans);
	generateDepthMap(trans);
	Mat dehaze = generateDehaze(img, trans, atm);
	imshow("src", imgCopy);
	imshow("dehaze", dehaze);
	dehaze.convertTo(dehaze, CV_8UC3, 255.0);
	imwrite(OUT_IMG_NAME, dehaze);
	waitKey(0);
	return 0;
}
