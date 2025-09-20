#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/opencv.hpp>

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
#include "utilities.h"

#define IMAGE_DEBUG_INFO 0

using namespace std;
using namespace cv;
namespace fs = std::filesystem;

const char *model_path = "nb_model/yolo_model.nb";
std::string input_dir = "input_images";
std::string output_dir = "output_dirs/v8s_prune/prune20";

vsi_nn_graph_t * g_graph = NULL;

const static vsi_nn_postprocess_map_element_t* postprocess_map = NULL;
const static vsi_nn_preprocess_map_element_t* preprocess_map = NULL;

const vsi_nn_preprocess_map_element_t * vnn_GetPrePorcessMap()
{
	return preprocess_map;
}

uint32_t vnn_GetPrePorcessMapCount()
{
	if (preprocess_map == NULL)
		return 0;
	else
		return sizeof(preprocess_map) / sizeof(vsi_nn_preprocess_map_element_t);
}

const vsi_nn_postprocess_map_element_t * vnn_GetPostPorcessMap()
{
	return postprocess_map;
}

uint32_t vnn_GetPostPorcessMapCount()
{
	if (postprocess_map == NULL)
		return 0;
	else
		return sizeof(postprocess_map) / sizeof(vsi_nn_postprocess_map_element_t);
}

int run_detect_model(){
	fs::create_directories(output_dir);

	int g_nn_height, g_nn_width, g_nn_channel;

	double pre_processing_total_time = 0.0;
   	double inference_total_time = 0.0;
	double post_processing_total_time = 0.0;
	double img_processing_total_time = 0.0;

	// Load the model
	g_graph = vnn_CreateYoloModel(model_path, NULL,
	vnn_GetPrePorcessMap(), vnn_GetPrePorcessMapCount(),
	vnn_GetPostPorcessMap(), vnn_GetPostPorcessMapCount());

	vsi_nn_tensor_t *tensor = vsi_nn_GetTensor(g_graph, g_graph->input.tensors[0]);
	g_nn_width = tensor->attr.size[0];
	g_nn_height = tensor->attr.size[1];
	g_nn_channel = tensor->attr.size[2];

	size_t total_files = std::distance(std::filesystem::directory_iterator(input_dir), {});
	size_t count = 0;

	size_t update_every = total_files / 100;

	std::cout << "\nBeginning model validation...\n";

	for (const auto& entry : fs::directory_iterator(input_dir)) {
		if (entry.is_regular_file()) {
			const auto& path = entry.path();
			if (path.extension() == ".jpg") {
				std::string filename = path.stem().string();
				std::string output_filename = output_dir + "/" + filename + ".txt";

				cv::Mat tmp_image(g_nn_height, g_nn_width, CV_8UC3);

				cv::Mat img = cv::imread(path.string());

				if (img.empty()) {
					cout << "Image not loaded properly!" << endl;
					return -1;
				}
				
				// Converter a imagem para a resolucao que o modelo aceita
				cv::resize(img, tmp_image, tmp_image.size());
			
				// Alterar a ordem dos canais de cores para RGB (o que o yolo está treinada para aceitar)
				cv::cvtColor(tmp_image, tmp_image, cv::COLOR_BGR2RGB);
				tmp_image.convertTo(tmp_image, CV_32FC3);

				tmp_image = tmp_image / 255.0;

				input_image_t image;
				image.data      = tmp_image.data;
				image.width     = tmp_image.cols;
				image.height    = tmp_image.rows;
				image.channel   = tmp_image.channels();
				image.pixel_format = PIX_FMT_RGB888;

				if(IMAGE_DEBUG_INFO == true)
					image_debug_info(tmp_image);
	                
				DetectResult resultData;

				struct timespec total_start, total_end;
				struct timespec t1_start, t1_end;
				struct timespec t2_start, t2_end;
				struct timespec t3_start, t3_end;
				
				//cv::imshow("Image Window",tmp_image);
   			    //cv::waitKey(0);

				clock_gettime(CLOCK_MONOTONIC, &total_start);

				// Pre-Processamento
				clock_gettime(CLOCK_MONOTONIC, &t1_start);
				model_preprocess(image, g_graph, g_nn_width, g_nn_height, g_nn_channel, tensor);
				clock_gettime(CLOCK_MONOTONIC, &t1_end);
				//printf("Preprocess took: %.6f seconds\n", time_diff(t1_start, t1_end));
				pre_processing_total_time += time_diff(t1_start, t1_end);

				// Inferencia
				clock_gettime(CLOCK_MONOTONIC, &t2_start);
				vsi_nn_RunGraph(g_graph);
				clock_gettime(CLOCK_MONOTONIC, &t2_end);
				//printf("Inference took: %.6f seconds\n", time_diff(t2_start, t2_end));
				inference_total_time += time_diff(t2_start, t2_end);
                
				// Pos-Processamento
				clock_gettime(CLOCK_MONOTONIC, &t3_start);
				model_postprocess(g_graph, &resultData);
				clock_gettime(CLOCK_MONOTONIC, &t3_end);
				//printf("Postprocess took: %.6f seconds\n", time_diff(t3_start, t3_end));
				post_processing_total_time += time_diff(t3_start, t3_end);

				clock_gettime(CLOCK_MONOTONIC, &total_end);
				//printf("Total time: %.6f seconds\n", time_diff(total_start, total_end));
				img_processing_total_time += time_diff(total_start, total_end);

				// Draw and show results
				draw_results(img, resultData, img.cols, img.rows);
				//cv::imwrite("result.jpg", img); // Save output image
				//cv::waitKey(0);  // Keep window open

				std::ofstream outfile(output_filename);
				for (int i = 0; i < resultData.detect_num; i++) {
					int left = resultData.point[i].point.rectPoint.left*img.cols;
					int top = resultData.point[i].point.rectPoint.top*img.rows;
					int right = resultData.point[i].point.rectPoint.right*img.cols;
					int bottom = resultData.point[i].point.rectPoint.bottom*img.rows;	
								
					//cout << resultData.result_name[i].lable_name_only << " " << resultData.result_name[i].confidence << " " << left << " " << top << " " << right << " " << bottom << '\n';
					outfile << resultData.result_name[i].lable_name_only << " " << resultData.result_name[i].confidence << " " << left << " " << top << " " << right << " " << bottom<< '\n';
				}
			}
			++count;
    		if (count % update_every == 0 || count == total_files)
        		print_progress_bar(count, total_files);
		}
	}

	std::cout << "\nValidation run complete!\n";
	printf("\nAverage Preprocess Time: %.6f seconds.\n", pre_processing_total_time/total_files);
   	printf("Average Image Inference Time: %.6f seconds.\n", inference_total_time/total_files);
	printf("Average Postprocess Time: %.6f seconds.\n", post_processing_total_time/total_files);
	printf("\nAverage Total Time for Image Processing: %.6f seconds, resulting in %.0f fps.\n", img_processing_total_time/total_files, (1/(img_processing_total_time/total_files)));

	vnn_ReleaseYoloModel(g_graph, TRUE);
	g_graph = NULL;

	return 0;
}

int main(int argc, char** argv){

	run_detect_model();

	return 0;
}
