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

		// if (m_filenameDepthImages.size() != m_filenameColorImages.size())
		// 	return false;

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

			if(m_depthFrame[i] < 0.4f || m_depthFrame[i] > 8.0f){
				m_depthFrame[i] = MINF;
			}
		}

		// TODO filter and m_depthFrame_filtered
		vertexMaps.clear();
		normalMaps.clear();
		buildPyramids();

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

	cv::Mat getVertexMap(int i)
	{

		return vertexMaps[i];
	}
	cv::Mat getNormalMap(int i)
	{

		return normalMaps[i];
	}

private:
	void buildVertexAndNormalMaps(const cv::Mat &depthImage, int level)
	{

		vertexMaps.push_back(cv::Mat(depthImage.rows, depthImage.cols, CV_32FC3));
		vertexMaps[level].setTo(0);
		normalMaps.push_back(cv::Mat(depthImage.rows, depthImage.cols, CV_32FC3));
		normalMaps[level].setTo(0);

		//TODO test it with MINF also
		float scaleFactor = pow(0.5, level);
		float fovX = m_depthIntrinsics(0, 0) * scaleFactor;
		float fovY = m_depthIntrinsics(1, 1) * scaleFactor;
		float cX = (m_depthIntrinsics(0, 2) ) * scaleFactor ;
		float cY = (m_depthIntrinsics(1, 2) ) * scaleFactor ;
		for (int i = 0; i < depthImage.rows; i++)
		{
			for (int j = 0; j < depthImage.cols; j++)
			{

				float depth = depthImage.at<float>(i, j);
				if (depth != MINF && !std::isnan(depth))
				{


					Vector3f vert((j - cX) / fovX * depth, (i - cY) / fovY * depth, depth);
					vertexMaps[level].at<cv::Vec3f>(i, j)[0] = vert.x();
					vertexMaps[level].at<cv::Vec3f>(i, j)[1] = vert.y();
					vertexMaps[level].at<cv::Vec3f>(i, j)[2] = vert.z();
				}
				else
				{

					vertexMaps[level].at<cv::Vec3f>(i, j)[0] = MINF;
					vertexMaps[level].at<cv::Vec3f>(i, j)[1] = MINF;
					vertexMaps[level].at<cv::Vec3f>(i, j)[2] = MINF;

				}
			}
		}

		for (int i = 1; i < depthImage.rows - 1; i++)
		{
			for (int j = 1; j < depthImage.cols - 1; j++)
			{

				Vector3f left;
				left.x() = vertexMaps[level].at<cv::Vec3f>(i, j - 1)[0];
				left.y() = vertexMaps[level].at<cv::Vec3f>(i, j - 1)[1];
				left.z() = vertexMaps[level].at<cv::Vec3f>(i, j - 1)[2];

				Vector3f right;
				right.x() = vertexMaps[level].at<cv::Vec3f>(i, j + 1)[0];
				right.y() = vertexMaps[level].at<cv::Vec3f>(i, j + 1)[1];
				right.z() = vertexMaps[level].at<cv::Vec3f>(i, j + 1)[2];

				Vector3f up;
				up.x() = vertexMaps[level].at<cv::Vec3f>(i + 1, j)[0];
				up.y() = vertexMaps[level].at<cv::Vec3f>(i + 1, j)[1];
				up.z() = vertexMaps[level].at<cv::Vec3f>(i + 1, j)[2];

				Vector3f down;
				down.x() = vertexMaps[level].at<cv::Vec3f>(i - 1, j)[0];
				down.y() = vertexMaps[level].at<cv::Vec3f>(i - 1, j)[1];
				down.z() = vertexMaps[level].at<cv::Vec3f>(i - 1, j)[2];

				Vector3f diffX = right - left;
				Vector3f diffY = up - down;
				Vector3f vert;
				vert.x() = vertexMaps[level].at<cv::Vec3f>(i, j)[0];
				vert.y() = vertexMaps[level].at<cv::Vec3f>(i, j)[1];
				vert.z() = vertexMaps[level].at<cv::Vec3f>(i, j)[2];
				Vector3f normalVector = diffY.cross(diffX).normalized();
				const float du = 0.5f * (depthImage.at<float>(i, j + 1) - depthImage.at<float>(i, j - 1));
				const float dv = 0.5f * (depthImage.at<float>(i + 1, j) - depthImage.at<float>(i - 1, j));
				if (vert.allFinite() && normalVector.allFinite() && !(!std::isfinite(du) || !std::isfinite(dv) || abs(du) > 0.1f / 2 || abs(dv) > 0.1f / 2))
				{
					normalMaps[level].at<cv::Vec3f>(i, j)[0] = normalVector.x();
					normalMaps[level].at<cv::Vec3f>(i, j)[1] = normalVector.y();
					normalMaps[level].at<cv::Vec3f>(i, j)[2] = normalVector.z();
				}
				else
				{

					normalMaps[level].at<cv::Vec3f>(i, j)[0] = MINF;
					normalMaps[level].at<cv::Vec3f>(i, j)[1] = MINF;
					normalMaps[level].at<cv::Vec3f>(i, j)[2] = MINF;
				}
			}
		}
	}
	void buildPyramids()
	{

		cv::Mat depthImage(m_depthImageHeight, m_depthImageWidth, CV_32FC1, m_depthFrame);
		cv::Mat depthImageHalf(m_depthImageHeight / 2, m_depthImageWidth / 2, CV_32FC1);
		cv::Mat depthImageQuarter(m_depthImageHeight / 4, m_depthImageWidth / 4, CV_32FC1);
		cv::Mat smoothDepthImage(m_depthImageHeight, m_depthImageWidth, CV_32FC1);
		cv::Mat smoothDepthImageHalf(m_depthImageHeight / 2, m_depthImageWidth / 2, CV_32FC1);
		cv::Mat smoothDepthImageQuarter(m_depthImageHeight / 4, m_depthImageWidth / 4, CV_32FC1);
		// smoothDepthImage = depthImage;
		
	
		cv::pyrDown(depthImage, depthImageHalf);
		cv::pyrDown(depthImageHalf, depthImageQuarter);
	
		cv::bilateralFilter(depthImage, smoothDepthImage, 5, 400/5000, 400/5000, cv::BORDER_DEFAULT);
		cv::bilateralFilter(depthImageHalf, smoothDepthImageHalf, 5, 400/5000, 400/5000, cv::BORDER_DEFAULT);
		cv::bilateralFilter(depthImageQuarter, smoothDepthImageQuarter, 5, 400/5000, 400/5000, cv::BORDER_DEFAULT);

		buildVertexAndNormalMaps(smoothDepthImage, 0);
		buildVertexAndNormalMaps(smoothDepthImageHalf, 1);
		buildVertexAndNormalMaps(smoothDepthImageQuarter, 2);
	}
	bool
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

	std::vector<cv::Mat> vertexMaps;
	std::vector<cv::Mat> normalMaps;

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
