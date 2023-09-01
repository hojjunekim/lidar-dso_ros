/**
* This file is part of DSO.
* 
* Copyright 2016 Technical University of Munich and Intel.
* Developed by Jakob Engel <engelj at in dot tum dot de>,
* for more information see <http://vision.in.tum.de/dso>.
* If you use this code, please cite the respective publications as
* listed on the above website.
*
* DSO is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* DSO is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with DSO. If not, see <http://www.gnu.org/licenses/>.
*/





#include <locale.h>
#include <signal.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>

#include "util/settings.h"
#include "FullSystem/FullSystem.h"
#include "util/Undistort.h"
#include "IOWrapper/Pangolin/PangolinDSOViewer.h"
#include "IOWrapper/OutputWrapper/SampleOutputWrapper.h"


#include <ros/ros.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/PoseStamped.h>
#include "cv_bridge/cv_bridge.h"


#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <opencv2/opencv.hpp>
#include <pcl/PCLPointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>

std::string calib = "./src/dso_ros/config/STR_rgb_left.txt";
std::string vignetteFile = "";
std::string gammaFile = "";
std::string saveFile = "";
std::string imgTopic = "/kitti/left_image/";
std::string lidarTopic = "/kitti/velodyne_points/";
bool isImgColor = false;
bool useSampleOutput=false;
int img_min = 2900;
int img_max = 3400;
// morning: 3100-4800
// afternoon: 2900-4000
// evening: 2900-3400
ros::Time lastTime = ros::Time(0);

using namespace dso;
using namespace sensor_msgs;
using namespace message_filters;

void parseArgument(char* arg)
{
	int option;
	char buf[1000];
	if(1==sscanf(arg,"min=%d",&option))
	{
		img_min = option;
		printf("normalize min : %d\n", img_min);
		return;
	}
	if(1==sscanf(arg,"max=%d",&option))
	{
		img_max = option;
		printf("normalize max : %d\n", img_max);
		return;
	}
	if(1==sscanf(arg,"img=%s",buf))
	{
		imgTopic = buf;
		printf("image topic : %s\n", imgTopic.c_str());
		return;
	}
	if(1==sscanf(arg,"lidar=%s",buf))
	{
		lidarTopic = buf;
		printf("lidar topic : %s\n", lidarTopic.c_str());
		return;
	}
	if(1==sscanf(arg,"color=%dsss",&option))
	{
		if(option==1)
		{
			isImgColor = true;
			printf("Color Image Input\n");
		}
		return;
	}
	if(1==sscanf(arg,"savefile=%s",buf))
	{
		saveFile = buf;
		printf("saving to %s on finish!\n", saveFile.c_str());
		return;
	}

	if(1==sscanf(arg,"sampleoutput=%d",&option))
	{
		if(option==1)
		{
			useSampleOutput = true;
			printf("USING SAMPLE OUTPUT WRAPPER!\n");
		}
		return;
	}

	if(1==sscanf(arg,"quiet=%d",&option))
	{
		if(option==1)
		{
			setting_debugout_runquiet = true;
			printf("QUIET MODE, I'll shut up!\n");
		}
		return;
	}


	if(1==sscanf(arg,"nolog=%d",&option))
	{
		if(option==1)
		{
			setting_logStuff = false;
			printf("DISABLE LOGGING!\n");
		}
		return;
	}

	if(1==sscanf(arg,"nogui=%d",&option))
	{
		if(option==1)
		{
			disableAllDisplay = true;
			printf("NO GUI!\n");
		}
		return;
	}
	if(1==sscanf(arg,"nomt=%d",&option))
	{
		if(option==1)
		{
			multiThreading = false;
			printf("NO MultiThreading!\n");
		}
		return;
	}
	if(1==sscanf(arg,"calib=%s",buf))
	{
		calib = buf;
		printf("loading calibration from %s!\n", calib.c_str());
		return;
	}
	if(1==sscanf(arg,"vignette=%s",buf))
	{
		vignetteFile = buf;
		printf("loading vignette from %s!\n", vignetteFile.c_str());
		return;
	}

	if(1==sscanf(arg,"gamma=%s",buf))
	{
		gammaFile = buf;
		printf("loading gammaCalib from %s!\n", gammaFile.c_str());
		return;
	}
	
	if(1==sscanf(arg,"savefile=%s",buf))
	{
		saveFile = buf;
		printf("saving to %s on finish!\n", saveFile.c_str());
		return;
	}
	printf("could not parse argument \"%s\"!!\n", arg);
}

cv::Mat normalizeToRange(cv::Mat input, double minValue, double maxValue)
{
	cv::Mat output;
	double minVal = 0;
	double maxVal = 255;
	// double min, max;
	// cv::minMaxLoc(input, &min, &max);
	// printf("\nmin: %f max: %f\n", min, max);
	
	// Calculate the scaling factor
	double scale = (maxVal - minVal) / (maxValue - minValue);

	// Apply the normalization
	cv::subtract(input, minValue, output);
	cv::multiply(output, scale, output);
	cv::add(output, minVal, output);
	output.convertTo(output,CV_8UC1);
	return output;
}


FullSystem* fullSystem = 0;
Undistort* undistorter = 0;
int frameID = 0;

void imgLidCb(const sensor_msgs::ImageConstPtr& img, const sensor_msgs::PointCloud2ConstPtr& lidar_msg)
{
	cv_bridge::CvImagePtr cv_ptr;
	cv::Mat cv_image;
	if(isImgColor){
		cv_ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::MONO8);
		cv_image = cv_ptr->image.clone();
	}
	else{
		cv_ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::MONO16);
		cv_image = cv_ptr->image.clone();
		cv_image = normalizeToRange(cv_image, img_min, img_max);
	}
	
	// filtering(noise reduction)
	// cv::Mat smoothed;
	// double nonlocalh = 2; // Filtering strength.
	// cv::fastNlMeansDenoising(cv_image_norm, smoothed, nonlocalh);

	// cv::Mat sharpened;
	// cv::Mat sharp_kernel = (cv::Mat_<float>(3, 3) <<  0, -1,  0,
	// 										-1,  5, -1,
	// 											0, -1,  0);
	// cv::filter2D(cv_image_norm, sharpened, -1, sharp_kernel);


	// assert(cv_image.type() == CV_8U);
	assert(cv_image.channels() == 1);


	pcl::PCLPointCloud2 pcl_cloud;
	pcl_conversions::toPCL(*lidar_msg,pcl_cloud);

	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::fromPCLPointCloud2(pcl_cloud, *cloud);

	// STheReo rgb left
	float proj_rgbL2therL[] = {1.0, 0.0, 0.0, 0.0554,
		0.0, 1.0, 0.0, -0.0031,
		0.0, 0.0, 1.0, -0.0069,
		0.0, 0.0, 0.0, 1.0};
	float intrinsic[] = {788.41328, 0., 633.40907, 0.,
       790.92597, 237.46688, 0., 0., 1.};

	// STheReo thermal left
	float proj_lidar2therL[] = {-0.0260, -0.9995, 0.0181, 0.3125,
		-0.0095, -0.0179, -0.9997, -0.1815,
		0.9996, -0.0262, -0.0090, 0.0110,
		0.0, 0.0, 0.0, 1.0};
	
	// STheReo thermal left
	// float intrinsic[] = {447.7496, 0., 328.5247, 0.,
    //    447.4655, 260.0027, 0., 0., 1.};
	
	// dataset
	// float intrinsic[] = {429.43288714549999, 0., 311.11923634459998, 0.,
    //    429.5314275019, 266.12817575460002, 0., 0., 1.};
	
	/* handheld thermal right & lidar
	float projection[] = {-0.0765, -0.9927, 0.0928, 0.0006,
                            -0.1048, -0.0846, -0.9909, 0.0771,
                            0.9915, -0.0856, -0.0976, 0.0027};
	float intrinsic[] = {416.9527, 0.0, 313.3428,
                            0.0, 416.7449, 252.7190,
                            0.0, 0.0, 1.0};
	*/
	
	cv::Mat P_rgbL2therL(4,4,CV_32FC1,proj_rgbL2therL);
	cv::Mat P_lidar2therL(4,4,CV_32FC1,proj_lidar2therL);
	cv::Mat result = P_lidar2therL * P_rgbL2therL.inv();
	cv::Mat P = result.rowRange(0, 3);

	cv::Mat K(3,3,CV_32FC1,intrinsic);
	int ptnum = cloud->width;

	std::vector<std::vector<float>>* ptCloud = new std::vector<std::vector<float>>;
	int ptInnum = 0;

	int w = 1280;
	int h = 560;

	float* map_pt = new float[w*h];
	memset(map_pt,0,w*h*4);
	cv::Mat lidar_3d;
	cv::Mat img_3d;
	cv::Mat img_2d;

	for (const auto& point : cloud->points)
	{
		if(point.x<5) continue;
		lidar_3d = (cv::Mat_<float>(4,1) << point.x, point.y, point.z, 1);
		img_3d = P * lidar_3d;

		// if(img_3d.at<float>(2,0)<0) continue;
		float depth = sqrt(pow(img_3d.at<float>(0,0),2)+pow(img_3d.at<float>(1,0),2)+pow(img_3d.at<float>(2,0),2));
		if(depth<0.5) continue;
		img_3d = img_3d / img_3d.at<float>(2,0);
		img_2d = K * img_3d;
		float img_2dx = img_2d.at<float>(0,0);
		float img_2dy = img_2d.at<float>(1,0);

		if(img_2dx >= 0 && img_2dx <= cv_image.cols && img_2dy >= 0 && img_2dy <= cv_image.rows)
		{
			
			int xi = img_2d.at<float>(0,0);
			int yi = img_2d.at<float>(1,0);
			int idx = xi+yi*cv_image.cols;
			map_pt[idx] = 1;

			std::vector<float> point_row;
			point_row.push_back(xi);
			point_row.push_back(yi);
			point_row.push_back(idx);
			point_row.push_back(depth);
			ptCloud->push_back(point_row);
			
			ptInnum++;
		}
	}


	if(setting_fullResetRequested)
	{
		std::vector<IOWrap::Output3DWrapper*> wraps = fullSystem->outputWrapper;
		delete fullSystem;
		for(IOWrap::Output3DWrapper* ow : wraps) ow->reset();
		fullSystem = new FullSystem();
		fullSystem->linearizeOperation=false;
		fullSystem->outputWrapper = wraps;
	    if(undistorter->photometricUndist != 0)
	    	fullSystem->setGammaFunction(undistorter->photometricUndist->getG());
		setting_fullResetRequested=false;

	}

	MinimalImageB minImg((int)cv_image.cols, (int)cv_image.rows,(unsigned char*)cv_image.data);
	ImageAndExposure* undistImg = undistorter->undistort<unsigned char>(&minImg, 1,0, 1.0f);
	undistImg->timestamp=img->header.stamp.toSec(); // relay the timestamp to dso
	
	ros::Time currentTime = ros::Time::now();
	fullSystem->addActiveFrame(undistImg, frameID, ptCloud, map_pt);
	double addFrameHZ = 1/(currentTime - lastTime).toSec();
	printf("[hz] %.1f ", addFrameHZ);
	lastTime = currentTime;


	frameID++;
	delete undistImg;
	ptCloud = nullptr;
	map_pt = nullptr;
	delete ptCloud;
	delete[] map_pt;
}





int main( int argc, char** argv )
{
	ros::init(argc, argv, "dso_live");
    ros::NodeHandle nh;

	for(int ptInnum=1; ptInnum<argc;ptInnum++) parseArgument(argv[ptInnum]);

	setting_desiredImmatureDensity = 1000;
	setting_desiredPointDensity = 1200;
	setting_minFrames = 5;
	setting_maxFrames = 7;
	setting_maxOptIterations=4;
	setting_minOptIterations=1;
	setting_logStuff = false;
	setting_kfGlobalWeight = 1.3;


	printf("MODE WITH CALIBRATION, but without exposure times!\n");
	setting_photometricCalibration = 2;
	setting_affineOptModeA = 0;
	setting_affineOptModeB = 0;



    undistorter = Undistort::getUndistorterForFile(calib, gammaFile, vignetteFile);

    setGlobalCalib(
            (int)undistorter->getSize()[0],
            (int)undistorter->getSize()[1],
            undistorter->getK().cast<float>());


    fullSystem = new FullSystem();
    fullSystem->linearizeOperation=false;


    if(!disableAllDisplay)
	    fullSystem->outputWrapper.push_back(new IOWrap::PangolinDSOViewer(
	    		 (int)undistorter->getSize()[0],
	    		 (int)undistorter->getSize()[1]));


    if(useSampleOutput)
        fullSystem->outputWrapper.push_back(new IOWrap::SampleOutputWrapper());


    if(undistorter->photometricUndist != 0)
    	fullSystem->setGammaFunction(undistorter->photometricUndist->getG());

    message_filters::Subscriber<Image> imgSub(nh, imgTopic, 1);
    message_filters::Subscriber<PointCloud2> lidSub(nh, lidarTopic, 1);
    typedef sync_policies::ApproximateTime<Image, PointCloud2> SyncPolicy;
	static Synchronizer<SyncPolicy> sync(SyncPolicy(10), imgSub, lidSub);
    sync.registerCallback(boost::bind(&imgLidCb, _1, _2));

    ros::spin();
    fullSystem->printResult(saveFile); 
    for(IOWrap::Output3DWrapper* ow : fullSystem->outputWrapper)
    {
        ow->join();
        delete ow;
    }

    delete undistorter;
    delete fullSystem;

	return 0;
}

