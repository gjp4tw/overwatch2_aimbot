#ifdef _WIN32
#define GPU
#define OPENCV
#endif

#include<iostream>
#include<windows.h>
#include "yolo_v2_class.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/dnn/dnn.hpp"
#include <opencv2/dnn/all_layers.hpp>
using namespace std;
using namespace cv;

HDC hdc, hdcMem;
HWND hwnd;
HBITMAP hBitmap;
void* ptrBitmapPixels;


int SCREENWIDTH = 1920;
int SCREENHEIGHT = 1080;
int Screenshot_W = 320;
int Screenshot_H = 320;

float confidenceThreshold = 0.85;
unique_ptr<Detector>detector;
dnn::dnn4_v20221220::Net net;
BITMAPINFO bmi;

int LEFT = int((SCREENWIDTH - Screenshot_W)/2.0), TOP = int((SCREENHEIGHT - Screenshot_H)/2.0);
int TYPE = 0;
bool SHOW = 0, AIM = 1;

string model = "ow_final";
int MAX_DIS = 55;
bool TRIGGER = true;
vector<bbox_t> DetectionObject;
int smooth = 30;
Mat frame;

void init_screenshot() {
	cout << "initing screenshot...\n";
	ZeroMemory(&bmi, sizeof(BITMAPINFO));
	bmi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
	bmi.bmiHeader.biWidth = Screenshot_W;
	bmi.bmiHeader.biHeight = -Screenshot_H;
	bmi.bmiHeader.biPlanes = 1;
	bmi.bmiHeader.biBitCount = 24;
	bmi.bmiHeader.biCompression = BI_RGB;
	bmi.bmiHeader.biSizeImage = Screenshot_H * Screenshot_H;
	while (hwnd == NULL) {
		hwnd = FindWindowA(NULL, "Windowed Projector (Source) - overwatch");
	}
	hdc = GetWindowDC(hwnd);
	hdcMem = CreateCompatibleDC(hdc);
	hBitmap = CreateDIBSection(hdc, &bmi, DIB_RGB_COLORS, (void**)&ptrBitmapPixels, NULL, 0);
	SelectObject(hdcMem, hBitmap);
	frame = Mat(Screenshot_W, Screenshot_H, CV_8UC3, ptrBitmapPixels);
}

void loadNet() {
	cout << "loading net...\n";
	switch (TYPE) {
	case 0:
		detector = make_unique<Detector>("./models/" + model + "/weight.cfg", "./models/" + model + "/weight.weights");
		break;
	case 1:
		net = dnn::readNetFromDarknet("./models/" + model + "/weight.cfg", "./models/" + model + "/weight.weights");
		net.setPreferableBackend(dnn::DNN_BACKEND_CUDA);
		net.setPreferableTarget(dnn::DNN_TARGET_CUDA_FP16);
		break;
	default:
		break;
	}
}
void move_mouse(int cx, int cy) {
	mouse_event(MOUSEEVENTF_MOVE, cx, cy, 0, 0);
}
void aimbot(){
	init_screenshot();
	loadNet();
	auto start = chrono::high_resolution_clock::now();
	auto end = chrono::high_resolution_clock::now();
	auto duration = chrono::duration_cast<chrono::microseconds>(end-start);
	while(true){
		if (GetAsyncKeyState(VK_DELETE) < 0)break;
		if (GetAsyncKeyState(VK_NUMPAD1) & 0x8000 != 0) {
			TYPE = 0;
			loadNet();
		}
		if (GetAsyncKeyState(VK_NUMPAD2) & 0x8000 != 0) {
			TYPE = 1;
			loadNet();
		}
		if (GetAsyncKeyState(VK_NUMPAD8 ) & 0x8000!= 0) {
			SHOW = !SHOW;
		}
		if (GetAsyncKeyState(VK_NUMPAD0) & 0x8000 != 0) {
			AIM = !AIM;
		}
		if (GetAsyncKeyState(VK_UP) & 0x8000 != 0 && AIM) {
			smooth += 1;
			cout << smooth << "\n";
		}
		if (GetAsyncKeyState(VK_DOWN) & 0x8000 != 0 && AIM) {
			smooth -= 1;
			smooth = max(1, smooth);
			cout << smooth<< "\n";
		}
		if(!TRIGGER || GetAsyncKeyState(VK_XBUTTON1)&0x8000){
			start = chrono::high_resolution_clock::now();

			BitBlt(hdcMem, 0, 0, Screenshot_W, Screenshot_H, hdc, LEFT, TOP, SRCCOPY);
			double min_dis = 1e9;
			vector<int> closest={};
			float f;
			switch (TYPE){
			case 0: {
				DetectionObject = detector->detect(frame, 0.85);
				for (int i = 0; i < DetectionObject.size(); i++) {
					double dx = DetectionObject[i].x + DetectionObject[i].w / 2.0 - Screenshot_W / 2.0;
					double dy = DetectionObject[i].y + DetectionObject[i].h /4.0 - Screenshot_H / 2.0;
					double dis = double(sqrtf(dx * dx + dy * dy));
					circle(frame, Point(int(DetectionObject[i].x + DetectionObject[i].w / 2.0), int(DetectionObject[i].y + DetectionObject[i].h / 4.0)), MAX_DIS, Scalar(0, 0, 255), 10);
					if (dis > MAX_DIS)continue;
					if (dis < min_dis) {
						min_dis = dis;
						if (dis > 30)f = 0.1;
						else f = 1.8;
						closest = { int(f*dx * abs(dx) / (MAX_DIS + smooth)),int(f*dy * abs(dy) / (MAX_DIS + smooth)) ,int(DetectionObject[i].x + DetectionObject[i].w / 2.0), int(DetectionObject[i].y + DetectionObject[i].h / 4.0)};
					}
				}
				break;
			}
			case 1: {
				Mat blob;
				dnn::blobFromImage(frame, blob, 1 / 255.0, Size(Screenshot_W, Screenshot_H), cv::Scalar(), true, false);
				net.setInput(blob);
				Mat output = net.forward();
				for (int i = 0; i < output.rows; i++) {
					float confidence = output.at<float>(i, 4);
					if (confidence > confidenceThreshold) {
						int x = static_cast<int>(output.at<float>(i, 0) * frame.cols);
						int y = static_cast<int>(output.at<float>(i, 1) * frame.rows - output.at<float>(1,3)*frame.rows*0.25);
						double dx = x  - Screenshot_W / 2.0;
						double dy = y  - Screenshot_H / 2.0;
						double dis = double(sqrtf(dx * dx + dy * dy));
						if (dis > MAX_DIS)continue;
						if (dis < min_dis) {
							min_dis = dis;
							if (dis > 30)f = 0.1;
							else f = 1.8;
							closest = { int(f * dx * abs(dx) / (MAX_DIS + smooth)),int(f * dy * abs(dy) / (MAX_DIS + smooth)) ,int(DetectionObject[i].x + DetectionObject[i].w / 2.0), int(DetectionObject[i].y + DetectionObject[i].h / 4.0) };
						}
					}
				}
			}
			default:
				break;
			}
			if(AIM&&closest.size()){
				thread(move_mouse, closest[0], closest[1]).detach();
			}
			end = chrono::high_resolution_clock::now();
			duration = chrono::duration_cast<chrono::microseconds>(end - start);
			putText(frame, to_string(1000000 / duration.count()), Point(10, 15), FONT_HERSHEY_COMPLEX_SMALL, 1, Scalar(0, 0, 255));
			if (SHOW) {
				imshow("frame", frame);
				waitKey(1);
			}
			else {
				destroyAllWindows();
			}
			
		}
	}
	destroyAllWindows();
	DeleteDC(hdcMem);
	ReleaseDC(hwnd, hdc);
	DeleteObject(hBitmap);
	frame.release();
}
int main(){
	aimbot();
}