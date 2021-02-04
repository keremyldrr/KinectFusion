#pragma once

#include <vector>
#include <iostream>
#include <cstring>
#include <fstream>

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/base.hpp>

#include "Eigen.h"
#include "FreeImageHelper.h"

typedef unsigned char BYTE;
// #ifndef MINF

#define MINF -std::numeric_limits<float>::infinity()
// #endif
// reads sensor files according to https://vision.in.tum.de/data/datasets/rgbd-dataset/file_formats
class VirtualSensor
{
public:
	VirtualSensor() : m_currentIdx(-1), m_increment(1) {}

	~VirtualSensor()
	{
		SAFE_DELETE_ARRAY(m_depthFrame);
		SAFE_DELETE_ARRAY(m_depthFrame_filtered);
		SAFE_DELETE_ARRAY(m_colorFrame);
	}

	bool init(const std::string &datasetDir)
	{
		m_baseDir = datasetDir;

		// Read filename lists
		if (!readFileList(datasetDir + "depth.txt", m_filenameDepthImages, m_depthImagesTimeStamps))
			return false;
		if (!readFileList(datasetDir + "rgb.txt", m_filenameColorImages, m_colorImagesTimeStamps))
			return false;

		// Read tracking
		if (!readTrajectoryFile(datasetDir + "groundtruth.txt", m_trajectory, m_trajectoryTimeStamps))
			return false;

		if (m_filenameDepthImages.size() != m_filenameColorImages.size())
			return false;

		// Image resolutions
		m_colorImageWidth = 640;
		m_colorImageHeight = 480;
		m_depthImageWidth = 640;
		m_depthImageHeight = 480;

		// Intrinsics
		m_colorIntrinsics << 525.0f, 0.0f, 319.5f,
			0.0f, 525.0f, 239.5f,
			0.0f, 0.0f, 1.0f;

		m_depthIntrinsics = m_colorIntrinsics;

		m_colorExtrinsics.setIdentity();
		m_depthExtrinsics.setIdentity();

		m_depthFrame = new float[m_depthImageWidth * m_depthImageHeight];
		m_depthFrame_filtered = new float[m_depthImageWidth * m_depthImageHeight];
		for (unsigned int i = 0; i < m_depthImageWidth * m_depthImageHeight; ++i)
			m_depthFrame[i] = 0.5f;

		m_colorFrame = new BYTE[4 * m_colorImageWidth * m_colorImageHeight];
		for (unsigned int i = 0; i < 4 * m_colorImageWidth * m_colorImageHeight; ++i)
			m_colorFrame[i] = 255;

		m_currentIdx = -1;
		return true;
	}

	bool processNextFrame()
	{
		if (m_currentIdx == -1)
			m_currentIdx = 0;
		else
			m_currentIdx += m_increment;

		if ((unsigned int)m_currentIdx >= (unsigned int)m_filenameColorImages.size())
			return false;

		std::cout << "ProcessNextFrame [" << m_currentIdx << " | " << m_filenameColorImages.size() << "]" << std::endl;

		FreeImageB rgbImage;
		rgbImage.LoadImageFromFile(m_baseDir + m_filenameColorImages[m_currentIdx]);
		memcpy(m_colorFrame, rgbImage.data, 4 * 640 * 480);

		// depth images are scaled by 5000 (see https://vision.in.tum.de/data/datasets/rgbd-dataset/file_formats)
		FreeImageU16F dImage;
		dImage.LoadImageFromFile(m_baseDir + m_filenameDepthImages[m_currentIdx]);

		for (unsigned int i = 0; i < m_depthImageWidth * m_depthImageHeight; ++i)
		{
			if (dImage.data[i] == 0)
				m_depthFrame[i] = MINF;
			else
				m_depthFrame[i] = dImage.data[i] * 1.0f / 5000.0f;
		}

		// TODO filter and m_depthFrame_filtered
		cv::Mat filteredImage(m_depthImageHeight, m_depthImageWidth, CV_32FC1, m_depthFrame_filtered);
		;
		cv::Mat depthImage(m_depthImageHeight, m_depthImageWidth, CV_32FC1, m_depthFrame);

		// Filter Parameters can be modified further
		// ! TODO: Bilateral filter SEGFAULT in server
		cv::bilateralFilter(depthImage, filteredImage, 12, 0, 15, cv::BORDER_DEFAULT);
/*
		vertexMap = cv::Mat(m_depthImageHeight, m_depthImageWidth, CV_32FC3);
		normalMap = cv::Mat(m_depthImageHeight, m_depthImageWidth, CV_32FC3);
		vertexMap.setTo(MINF);
		normalMap.setTo(MINF); 

		
		//TODO test it with MINF also
		for (int i = 0; i < m_depthImageHeight; i++)
		{
			for (int j = 0; j < m_depthImageWidth; j++)
			{

				float depth = filteredImage.at<float>(i, j);
				if (depth != MINF)
				{
					Vector3f u_homo(i, j, 1);
					Vector3f vert = depth * m_depthIntrinsics.inverse() * u_homo;
					vertexMap.at<cv::Vec3f>(i, j)[0] = vert.x();
					vertexMap.at<cv::Vec3f>(i, j)[1] = vert.y();
					vertexMap.at<cv::Vec3f>(i, j)[2] = vert.z();
				}
			}
		}


		for (int i = 1; i < m_depthImageHeight - 1; i++)
		{
			for (int j = 1; j < m_depthImageWidth - 1; j++)
			{

				Vector3f left;
				left.x() = vertexMap.at<cv::Vec3f>(i, j - 1)[0];
				left.y() = vertexMap.at<cv::Vec3f>(i, j - 1)[1];
				left.z() = vertexMap.at<cv::Vec3f>(i, j - 1)[2];

				Vector3f right;
				right.x() = vertexMap.at<cv::Vec3f>(i, j + 1)[0];
				right.y() = vertexMap.at<cv::Vec3f>(i, j + 1)[1];
				right.z() = vertexMap.at<cv::Vec3f>(i, j + 1)[2];

				Vector3f up;
				up.x() = vertexMap.at<cv::Vec3f>(i + 1, j)[0];
				up.y() = vertexMap.at<cv::Vec3f>(i + 1, j)[1];
				up.z() = vertexMap.at<cv::Vec3f>(i + 1, j)[2];

				Vector3f down;
				down.x() = vertexMap.at<cv::Vec3f>(i - 1, j)[0];
				down.y() = vertexMap.at<cv::Vec3f>(i - 1, j)[1];
				down.z() = vertexMap.at<cv::Vec3f>(i - 1, j)[2];

				Vector3f diffX = right - left;
				Vector3f diffY = up - down;

				if (diffX.x() != MINF && diffY.x() != MINF)
				{

					Vector3f normalVector = diffY.cross(diffX).normalized();
					normalMap.at<cv::Vec3f>(i, j)[0] = normalVector.x();
					normalMap.at<cv::Vec3f>(i, j)[1] = normalVector.y();
					normalMap.at<cv::Vec3f>(i, j)[2] = normalVector.z();
				}
			}
		}


*/
		// cv::waitKey(0);
		// find transformation (simple nearest neighbor, linear search)
		double timestamp = m_depthImagesTimeStamps[m_currentIdx];
		double min = std::numeric_limits<double>::max();
		int idx = 0;
		for (unsigned int i = 0; i < m_trajectory.size(); ++i)
		{
			double d = abs(m_trajectoryTimeStamps[i] - timestamp);
			if (min > d)
			{
				min = d;
				idx = i;
			}
		}
		m_currentTrajectory = m_trajectory[idx];

		return true;
	}

	unsigned int getCurrentFrameCnt()
	{
		return (unsigned int)m_currentIdx;
	}

	// get current color data
	BYTE *getColorRGBX()
	{
		return m_colorFrame;
	}

	// get current depth data
	float *getDepth()
	{
		// return m_depthFrame_filtered;
		return m_depthFrame;
	}
	float *getDepthFiltered()
	{
		return m_depthFrame_filtered;
	}

	// color camera info
	Eigen::Matrix3f getColorIntrinsics()
	{
		return m_colorIntrinsics;
	}

	Eigen::Matrix4f getColorExtrinsics()
	{
		return m_colorExtrinsics;
	}

	unsigned int getColorImageWidth()
	{
		return m_colorImageWidth;
	}

	unsigned int getColorImageHeight()
	{
		return m_colorImageHeight;
	}

	// depth (ir) camera info
	Eigen::Matrix3f getDepthIntrinsics()
	{
		return m_depthIntrinsics;
	}

	Eigen::Matrix4f getDepthExtrinsics()
	{
		return m_depthExtrinsics;
	}

	unsigned int getDepthImageWidth()
	{
		return m_depthImageWidth;
	}

	unsigned int getDepthImageHeight()
	{
		return m_depthImageHeight;
	}

	// get current trajectory transformation
	Eigen::Matrix4f getTrajectory()
	{
		return m_currentTrajectory;
	}
/*
	cv::Mat getVertexMap()
	{

		
		return vertexMap;
	}
	cv::Mat getNormalMap(){

		return normalMap;
	}	
	
*/


private : bool
		  readFileList(const std::string &filename, std::vector<std::string> &result, std::vector<double> &timestamps)
{
	std::ifstream fileDepthList(filename, std::ios::in);
	if (!fileDepthList.is_open())
		return false;
	result.clear();
	timestamps.clear();
	std::string dump;
	std::getline(fileDepthList, dump);
	std::getline(fileDepthList, dump);
	std::getline(fileDepthList, dump);
	while (fileDepthList.good())
	{
		double timestamp;
		fileDepthList >> timestamp;
		std::string filename;
		fileDepthList >> filename;
		if (filename == "")
			break;
		timestamps.push_back(timestamp);
		result.push_back(filename);
	}
	fileDepthList.close();
	return true;
}

bool readTrajectoryFile(const std::string &filename, std::vector<Eigen::Matrix4f> &result,
						std::vector<double> &timestamps)
{
	std::ifstream file(filename, std::ios::in);
	if (!file.is_open())
		return false;
	result.clear();
	std::string dump;
	std::getline(file, dump);
	std::getline(file, dump);
	std::getline(file, dump);

	while (file.good())
	{
		double timestamp;
		file >> timestamp;
		Eigen::Vector3f translation;
		file >> translation.x() >> translation.y() >> translation.z();
		Eigen::Quaternionf rot;
		file >> rot;

		Eigen::Matrix4f transf;
		transf.setIdentity();
		transf.block<3, 3>(0, 0) = rot.toRotationMatrix();
		transf.block<3, 1>(0, 3) = translation;

		if (rot.norm() == 0)
			break;

		transf = transf.inverse().eval();

		timestamps.push_back(timestamp);
		result.push_back(transf);
	}
	file.close();
	return true;
}

EIGEN_MAKE_ALIGNED_OPERATOR_NEW
// cv::Mat vertexMap;
// cv::Mat normalMap;
// current frame index
int m_currentIdx;

int m_increment;

// frame data

float *m_depthFrame;
BYTE *m_colorFrame;
Eigen::Matrix4f m_currentTrajectory;

float *m_depthFrame_filtered;

// color camera info
Eigen::Matrix3f m_colorIntrinsics;
Eigen::Matrix4f m_colorExtrinsics;
unsigned int m_colorImageWidth;
unsigned int m_colorImageHeight;

// depth (ir) camera info
Eigen::Matrix3f m_depthIntrinsics;
Eigen::Matrix4f m_depthExtrinsics;
unsigned int m_depthImageWidth;
unsigned int m_depthImageHeight;

// base dir
std::string m_baseDir;
// filenamelist depth
std::vector<std::string> m_filenameDepthImages;
std::vector<double> m_depthImagesTimeStamps;
// filenamelist color
std::vector<std::string> m_filenameColorImages;
std::vector<double> m_colorImagesTimeStamps;

// trajectory
std::vector<Eigen::Matrix4f> m_trajectory;
std::vector<double> m_trajectoryTimeStamps;
};
