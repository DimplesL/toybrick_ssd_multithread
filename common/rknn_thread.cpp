#include <chrono>
#include "rknn_thread.h"

static const std::chrono::milliseconds con_var_time_out(100);

unsigned long rknn_opencv::get_time(void)
{
	struct timeval ts;
	gettimeofday(&ts, NULL);
	return (ts.tv_sec * 1000 + ts.tv_usec / 1000);
}

void *rknn_opencv::get_img_task(void *data)
{
	int ret;
	rknn_opencv *pd = (rknn_opencv *) data;
	printf("%s thread start...\n", __func__);

	while (pd->is_running) {
		cv::Mat tmp_origin;
		cv::Mat tmp_resize;

		ret = (*pd->get_img_func) (pd->user_ctx_ptr,  // 此处调用get_img，把图放进预先设计的内存中
					   tmp_origin, tmp_resize);
		if (ret < 0) {
			printf("get_img_func error[%d], stop demo...\n", ret);
			pd->is_running = false;
		}

		std::unique_lock < std::mutex > lock(pd->mtx_idle);

		if (pd->idle_queue.empty()) {  // idle是空的，则尚未处理完缓存图像。
			continue;
		}

		auto img = pd->idle_queue.front();  //idle不空了则取其第一个改变其内存存储的内容，更新为新图片
		pd->idle_queue.pop();
		lock.unlock();

		*(img->img_origin) = tmp_origin;
		*(img->img_resize) = tmp_resize;

		pd->mtx_input.lock();
		pd->input_queue.push(img);
		pd->cond_input_not_empty.notify_all();
		pd->mtx_input.unlock();
	}

	printf("%s thread terminate...\n", __func__);
	return NULL;
}

void *rknn_opencv::detect_img_task(void *data)
{
	int ret;
	rknn_opencv *pd = (rknn_opencv *) data;
	printf("%s thread start...\n", __func__);

	while (pd->is_running) {
		std::unique_lock < std::mutex > lock(pd->mtx_input);

		if (pd->input_queue.empty()) {
			pd->cond_input_not_empty.wait_for(lock,
							  con_var_time_out);
			continue;
		}

		auto img = pd->input_queue.front();  //img指向输入队列最前的图像内容
		pd->input_queue.pop();  //删除队列中最老那个，已经被img获取了
		lock.unlock();

		ret = (*pd->detect_img_func) (pd->user_ctx_ptr,
					      *(img->img_resize),
					      &img->out_data);
		if (ret < 0) {
			printf("detect_img_func error[%d], stop demo...\n",
			       ret);
			pd->is_running = false;
		}

		pd->mtx_output.lock();
		pd->output_queue.push(img);  // 与之前基本类似，把处理好的img存给output令其postprocess
		pd->cond_output_not_empty.notify_all();
		pd->mtx_output.unlock();
	}

	printf("%s thread terminate...\n", __func__);
	return NULL;
}

rknn_opencv::rknn_opencv()
{
	mat_count = 8;
	is_running = false;
	fps = 0.0;
	user_ctx_ptr = NULL;

	mats_queue = new rknn_queue_data[mat_count];
	mats_origin = new cv::Mat[mat_count];
	mats_resize = new cv::Mat[mat_count];

	for (int i = 0; i < mat_count; i++) {
		mats_queue[i].img_origin = &mats_origin[i];
		mats_queue[i].img_resize = &mats_resize[i];
		idle_queue.push(&mats_queue[i]);
	}

	frame_count = 0;
}

rknn_opencv::~rknn_opencv()
{
	delete[]mats_origin;
	delete[]mats_resize;
	delete[]mats_queue;
}

int rknn_opencv::start(int (*get_img) (void *, cv::Mat &, cv::Mat &),
		       int (*detect_img) (void *, cv::Mat &,
					  struct rknn_out_data *),
		       int (*show_img) (void *, cv::Mat &, float,
					struct rknn_out_data *), void *user_ctx)
{
	if (!get_img || !detect_img || !show_img)
		return -1;

	get_img_func = get_img;
	detect_img_func = detect_img;
	show_img_func = show_img;
	user_ctx_ptr = user_ctx;

	is_running = true;
	last_time = get_time();
	thread_get_img = std::thread(get_img_task, this);
	thread_detect_img = std::thread(detect_img_task, this);

	return 0;
}

int rknn_opencv::update_show(void)
{
	int ret;
	std::unique_lock < std::mutex > lock(mtx_output);

	if (output_queue.empty()) {
		if (is_running) {
			cond_output_not_empty.wait_for(lock, con_var_time_out);
			return 0;
		} else {
			return -1;
		}
	}

	auto img = output_queue.front();  // 与之前类似，取出处理完的第一张图，用来后处理
	output_queue.pop();
	lock.unlock();

	frame_count++;
	unsigned long cur_time = get_time();
	if (cur_time - last_time > 1000) {
		float sec_time = (cur_time - last_time) / 1000.0;
		fps = frame_count / sec_time;
		printf("%f, %5.2f\n", fps, fps);
		last_time = cur_time;
		frame_count = 0;
	}

	ret =
	    (*show_img_func) (user_ctx_ptr, *(img->img_origin), fps,  // 后处理直接到这里看即可，show之前有后处理逻辑
			      &img->out_data);

	mtx_idle.lock();
	idle_queue.push(img);  // 处理完的给空闲queue，表示可以继续给新图
	mtx_idle.unlock();

	return ret;
}

int rknn_opencv::stop(void)
{
	is_running = false;
	usleep(10000);

	cond_input_not_empty.notify_all();
	cond_output_not_empty.notify_all();

	thread_get_img.join();
	thread_detect_img.join();

	return 0;
}
