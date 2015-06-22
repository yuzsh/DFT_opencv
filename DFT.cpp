//
//  DFT.cpp
//  test00
//
//  Created by 湯澤拓矢 on 2015/06/14.
//  Copyright (c) 2015年 湯澤拓矢. All rights reserved.
//

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <iostream>
#include <string>

#define FILTER_R 70

//create complex image for DFT
void create_complex_dft_image(const cv::Mat& in, cv::Mat& out)
{
	using namespace cv;

	//real part
	Mat real_image;

	copyMakeBorder(in, real_image,
		0, getOptimalDFTSize(in.rows)-in.rows,
		0, getOptimalDFTSize(in.cols)-in.cols,
		BORDER_CONSTANT, Scalar::all(0));

	//make complex image
	Mat planes[] = { Mat_<float>(real_image), Mat::zeros(real_image.size(), CV_32F) };
	merge(planes, 2, out);
}

//create power spectrum
void create_fourier_magnitude_image_from_complex(const cv::Mat& in, cv::Mat& out)
{
	using namespace cv;

	//split real part and imaginary part
	Mat planes[2];
	split(in, planes);
    
    //calculate magnitude
	magnitude(planes[0], planes[1], out);

	//for display
	out += Scalar::all(1);
	log(out, out);

	out = out(Rect(0, 0, out.cols & -2, out.rows & -2));

	//swap quadrants
	const int half_width = out.cols / 2;
	const int half_height = out.rows / 2;

	Mat tmp;

	Mat q0(out, Rect(0, 0, half_width, half_height));
	Mat q1(out, Rect(half_width, 0, half_width, half_height));
	Mat q2(out, Rect(0, half_height, half_width, half_height));
	Mat q3(out, Rect(half_width, half_height, half_width, half_height));

	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);
	q1.copyTo(tmp);
	q2.copyTo(q1);
	tmp.copyTo(q2);

	//normalize
	normalize(out, out, 0, 1, CV_MINMAX);
}

//complex image -> real image
void create_inverse_fourier_image_from_complex(
	const cv::Mat& in, const::cv::Mat& origin, cv::Mat& out)
{
	using namespace cv;

	Mat splitted_image[2];
	split(in, splitted_image);

	//normalize
	splitted_image[0](cv::Rect(0, 0, origin.cols, origin.rows)).copyTo(out);
	cv::normalize(out, out, 0, 1, CV_MINMAX);
}


int main(int argc, char* argv[])
{
	if (argc < 2)
	{
		std::cout << "usage: " << argv[0] << " image " << std::endl;
		return 0;
	}


	auto image = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);


	if (image.empty())
	{
		std::cout << "can't read " << argv[1] << std::endl;
		return -1;
	}



	cv::Mat complex_image;
	create_complex_dft_image(image, complex_image);
	
	//DFT
	cv::dft(complex_image, complex_image);
	cv::Mat magnitude_image;
	create_fourier_magnitude_image_from_complex(complex_image, magnitude_image); //power spectrum



	//low pass filter
	cv::Mat filter_low = cv::Mat::zeros(512, 512, CV_8UC1);
	cv::circle(filter_low, cv::Point(0, 0), FILTER_R, cv::Scalar(255), -1);
	cv::circle(filter_low, cv::Point(0, filter_low.cols-1), FILTER_R, cv::Scalar(255), -1);
	cv::circle(filter_low, cv::Point(filter_low.rows-1, 0), FILTER_R, cv::Scalar(255), -1);
	cv::circle(filter_low, cv::Point(filter_low.rows-1, filter_low.cols-1), FILTER_R, cv::Scalar(255), -1);

	cv::Mat lowpass_out = cv::Mat::zeros(512, 512, CV_8UC1);
	complex_image.copyTo(lowpass_out, filter_low);
	cv::Mat magnitude_lpf;
	create_fourier_magnitude_image_from_complex(lowpass_out, magnitude_lpf); //power spectrum with LPF


	//high pass filter
	cv::Mat filter_high = ~filter_low;
	cv::Mat highpass_out = cv::Mat::zeros(512, 512, CV_8UC1);
	complex_image.copyTo(highpass_out, filter_high);
	cv::Mat magnitude_hpf;
	create_fourier_magnitude_image_from_complex(highpass_out, magnitude_hpf); //power spectrum with HPF
	


	//IDFT
	cv::idft(complex_image, complex_image);
	cv::Mat idft_image;
	create_inverse_fourier_image_from_complex(complex_image, image, idft_image); //for visualize of IDFT


	/*LPF -> IDFT*/
	cv::Mat idft_lpf;
	cv::idft(lowpass_out, idft_lpf);
	cv::Mat idft_image_lpf;
	create_inverse_fourier_image_from_complex(idft_lpf, image, idft_image_lpf);
	

	/*HPF -> IDFT*/
	cv::Mat idft_hpf;
	cv::idft(highpass_out, idft_hpf);
	cv::Mat idft_image_hpf;
	create_inverse_fourier_image_from_complex(idft_hpf, image, idft_image_hpf);
	

	cv::namedWindow("original");
	cv::imshow("original", image);

	cv::namedWindow("dft");
	cv::imshow("dft", magnitude_image);

	cv::namedWindow("idft");
	cv::imshow("idft", idft_image);

	cv::namedWindow("idft_lpf");
	cv::imshow("idft_lpf", idft_image_lpf);

	cv::namedWindow("idft_hpf");
	cv::imshow("idft_hpf", idft_image_hpf);

	cv::waitKey(0);

	cv::imwrite("LPF.jpg", filter_low);
	cv::imwrite("HPF.jpg", filter_high);
	cv::imwrite("LPF_spectrum.jpg", magnitude_lpf*255);
	cv::imwrite("HPF_spectrum.jpg", magnitude_hpf*255);

	cv::imwrite(std::string(argv[1]) + "_dft.jpg", magnitude_image * 255);
	cv::imwrite(std::string(argv[1]) + "_idft.jpg", idft_image * 255);
	cv::imwrite(std::string(argv[1]) + "_idft_lpf.jpg", idft_image_lpf * 255);
	cv::imwrite(std::string(argv[1]) + "_idft_hpf.jpg", idft_image_hpf * 255);

	return 0;
}

