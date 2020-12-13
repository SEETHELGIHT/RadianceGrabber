#include <vector>
#include <cmath>

#include "Image.h"

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;

namespace RadGrabber
{
#define NUM_LEN 256
#define PATH_LEN 2048

	int ImageWrite(const char* path, const char* ext, RadGrabber::ColorRGBA* rgba, int width, int height)
	{
		Mat pixelImage = Mat(height, width, CV_32FC4, rgba);
		cvtColor(pixelImage, pixelImage, COLOR_RGB2BGR);

		Mat newPixelImage = pixelImage.clone(), byteImage;
	
		newPixelImage *= 255;
		newPixelImage.convertTo(byteImage, CV_8UC4);

		flip(byteImage, byteImage, 0);

		vector<uchar> buff;//buffer for coding
		char buffer[PATH_LEN];
		sprintf(buffer, "%s%s", path, ext);
		//imencode(ext, pixelImage, buff);
		imwrite(buffer, byteImage);
		return 0;
	}

	int VideoWrite(const char* imagePathFormat, const char* imageExt, const char* videoPath, int width, int height, double fps, int frameCount)
	{
		char formatBuffer[NUM_LEN], imagePathFormatBuffer[NUM_LEN], imagePathBuffer[PATH_LEN];

		VideoWriter writer;
		writer.open(videoPath, VideoWriter::fourcc('X', 'V', 'I', 'D'), fps, Size(width, height), true);
		if (!writer.isOpened())
		{
			printf("VideoWrite Fucntion :: opening videopath(%s) fail..", videoPath);
			return 1;
		}

		int log10Value = (int)(log10((float)frameCount) + 1 - FLT_EPSILON);
		for (int i = 0; i < frameCount; i++)
		{
			//sprintf(formatBuffer, "%%0%dd", log10Value);
			sprintf(imagePathFormatBuffer, "%d%s", i, imageExt);
			sprintf(imagePathBuffer, imagePathFormat, imagePathFormatBuffer);

			Mat image = imread(imagePathBuffer, ImreadModes::IMREAD_COLOR);
			writer.write(image);
		}

		writer.release();

		return 0;
	}

	int VideoWrite2(const char* imagePathFormat, const char* videoPath, double fps, int frameCount, int startIndex, int endCount)
	{
		char formatBuffer[NUM_LEN], imagePathFormatBuffer[NUM_LEN], imagePathBuffer[PATH_LEN];

		sprintf(imagePathFormatBuffer, "%d", startIndex);
		sprintf(imagePathBuffer, imagePathFormat, imagePathFormatBuffer);
		Mat image = imread(imagePathBuffer, ImreadModes::IMREAD_COLOR);

		VideoWriter writer;
		writer.open(videoPath, VideoWriter::fourcc('X', 'V', 'I', 'D'), fps, Size(image.cols, image.rows), true);
		if (!writer.isOpened())
		{
			printf("VideoWrite Fucntion :: opening videopath(%s) fail..", videoPath);
			return 1;
		}

		int log10Value = (int)(log10((float)frameCount) + 1 - FLT_EPSILON);
		for (int i = startIndex; i < endCount; i++)
		{
			//sprintf(formatBuffer, "%%0%dd", log10Value);
			sprintf(imagePathFormatBuffer, "%d", i);
			sprintf(imagePathBuffer, imagePathFormat, imagePathFormatBuffer);

			Mat image = imread(imagePathBuffer, ImreadModes::IMREAD_COLOR);
			writer.write(image);
		}

		writer.release();

		return 0;
	}
}
