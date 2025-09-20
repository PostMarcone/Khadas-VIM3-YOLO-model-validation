#include "utilities.h"

double time_diff(struct timespec start, struct timespec end) {
    return (end.tv_sec - start.tv_sec) + 
           (double)(end.tv_nsec - start.tv_nsec) / BILLION;
}

int minmax(int min, int v, int max)
{
	return (v < min) ? min : (max < v) ? max : v;
}

uint8_t* yuyv2rgb(uint8_t* yuyv, uint32_t width, uint32_t height)
{
  	uint8_t* rgb = (uint8_t *)calloc(width * height * 3, sizeof (uint8_t));
  	for (size_t i = 0; i < height; i++) {
		for (size_t j = 0; j < width; j += 2) {
	  		size_t index = i * width + j;
	  		int y0 = yuyv[index * 2 + 0] << 8;
	  		int u = yuyv[index * 2 + 1] - 128;
	  		int y1 = yuyv[index * 2 + 2] << 8;
	  		int v = yuyv[index * 2 + 3] - 128;
	  		rgb[index * 3 + 0] = minmax(0, (y0 + 359 * v) >> 8, 255);
	  		rgb[index * 3 + 1] = minmax(0, (y0 + 88 * v - 183 * u) >> 8, 255);
	  		rgb[index * 3 + 2] = minmax(0, (y0 + 454 * u) >> 8, 255);
	  		rgb[index * 3 + 3] = minmax(0, (y1 + 359 * v) >> 8, 255);
	  		rgb[index * 3 + 4] = minmax(0, (y1 + 88 * v - 183 * u) >> 8, 255);
	  		rgb[index * 3 + 5] = minmax(0, (y1 + 454 * u) >> 8, 255);
		}
  	}
  	return rgb;
}

cv::Scalar obj_id_to_color(int obj_id) {

	int const colors[6][3] = { { 1,0,1 },{ 0,0,1 },{ 0,1,1 },{ 0,1,0 },{ 1,1,0 },{ 1,0,0 } };
	int const offset = obj_id * 123457 % 6;
	int const color_scale = 150 + (obj_id * 123457) % 100;
	cv::Scalar color(colors[offset][0], colors[offset][1], colors[offset][2]);
	color *= color_scale;
	return color;
}

void print_progress_bar(size_t current, size_t total) {
	size_t bar_width = 50;
	
	float progress = static_cast<float>(current) / total;
    	size_t pos = static_cast<size_t>(bar_width * progress);

	std::cout << "\r[";
    	for (size_t i = 0; i < bar_width; ++i) {
        	if (i < pos) std::cout << "#";
        	//else if (i == pos) std::cout << ">";
        	else std::cout << ".";
    	}
    	std::cout << "] " << int(progress * 100.0) << "% completion!" << std::flush;
}

void image_debug_info(cv::Mat& image){
    std::cout << "----- Image Debug Info -----" << std::endl;
    std::cout << "Width: " << image.cols << std::endl;
    std::cout << "Height: " << image.rows << std::endl;
    std::cout << "Channels: " << image.channels() << std::endl;

    int type = image.type();
    std::string r;

    switch (type) {
        case CV_8UC1:  r = "CV_8UC1"; break;
        case CV_8UC3:  r = "CV_8UC3"; break;
        case CV_32FC1: r = "CV_32FC1"; break;
        case CV_32FC3: r = "CV_32FC3"; break;
        default:       r = "Unknown"; break;
    }

    std::cout << "Type: " << r << " (" << type << ")" << std::endl;
    
    double min_val, max_val;
    cv::minMaxLoc(image, &min_val, &max_val);
    std::cout << "Pixel Range: min=" << min_val << ", max=" << max_val << std::endl;
    
    // Optional: print a sample pixel
    cv::Vec3f sample = image.at<cv::Vec3f>(image.rows / 2, image.cols / 2);
    std::cout << "Sample pixel at center: R=" << sample[0] << " G=" << sample[1] << " B=" << sample[2] << std::endl;
    
    std::cout << "----------------------------" << std::endl;
}

void draw_results(cv::Mat& frame, DetectResult resultData, int img_width, int img_height){

	int i = 0;
	float left, right, top, bottom;

	for (i = 0; i < resultData.detect_num; i++) {
		left =  resultData.point[i].point.rectPoint.left*img_width;
        right = resultData.point[i].point.rectPoint.right*img_width;
        top = resultData.point[i].point.rectPoint.top*img_height;
        bottom = resultData.point[i].point.rectPoint.bottom*img_height;
		
	    //cout << "i:" <<resultData.detect_num <<" left:" << left <<" right:" << right << " top:" << top << " bottom:" << bottom <<endl;

		cv::Rect rect(left, top, right-left, bottom-top);
		cv::rectangle(frame,rect,obj_id_to_color(resultData.result_name[i].lable_id),1,8,0);
		int baseline;
		cv::Size text_size = cv::getTextSize(resultData.result_name[i].lable_name, cv::FONT_HERSHEY_COMPLEX,0.5,1,&baseline);
		cv::Rect rect1(left, top-20, text_size.width+10, 20);
		cv::rectangle(frame,rect1,obj_id_to_color(resultData.result_name[i].lable_id),-1);
		cv::putText(frame,resultData.result_name[i].lable_name,cvPoint(left+5,top-5),cv::FONT_HERSHEY_COMPLEX,0.5,cv::Scalar(0,0,0),1);
	}

	//cv::imshow("Image Window",frame);
	//cv::waitKey(0);
}
