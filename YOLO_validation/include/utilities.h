#ifndef _UTILITIES_H_
#define _UTILITIES_H_

#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/opencv.hpp>

#include <string>
#include <unistd.h>
#include <string>
#include <iostream>
#include <stdio.h>
#include <fstream>
#include <dirent.h>
#include <queue>
#include <sched.h>
#include <sys/resource.h>
#include <fcntl.h>
#include <linux/fb.h>
#include <linux/kd.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <pthread.h>
#include <VX/vx.h>
#include <VX/vxu.h>
#include <VX/vx_api.h>
#include <VX/vx_khr_cnn.h>
#include <semaphore.h>
#include <sys/time.h>
#include <sched.h>
#include <linux/videodev2.h>
#include <poll.h>
#include <semaphore.h>
#include <getopt.h>
#include <time.h>
#include <filesystem>

#include "nn_detect_utils.h"
#include "model_pre_post_processing.h"
#include "vnn_yolomodel.h"

#define BILLION 1000000000L

double time_diff(struct timespec start, struct timespec end);

int minmax(int min, int v, int max);

uint8_t* yuyv2rgb(uint8_t* yuyv, uint32_t width, uint32_t height);

cv::Scalar obj_id_to_color(int obj_id);

void print_progress_bar(size_t current, size_t total);

void image_debug_info(cv::Mat& image);

void draw_results(cv::Mat& frame, DetectResult resultData, int img_width, int img_height);

#endif
