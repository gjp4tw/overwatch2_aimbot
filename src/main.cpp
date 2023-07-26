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
BITMAPINFO bmi;

int SCREENWIDTH = 1920;
int SCREENHEIGHT = 1080;
int Screenshot_W = 320;
int Screenshot_H = 320;

float confidenceThreshold = 0.85;
unique_ptr<Detector>detector;

int LEFT = int((SCREENWIDTH - Screenshot_W)/2.0), TOP = int((SCREENHEIGHT - Screenshot_H)/2.0);
int TYPE = 0;
bool SHOW = 0, AIM = 1;

string model = "model1";
int MAX_TRACK_DIS = 55;
int MAX_FLICK_DIS = 70;
bool TRIGGER = true;
vector<bbox_t> DetectionObject;
int smooth = 225;
Mat frame;

int FlickInterval = 450000; //mccree 450000 ashee 650000
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
	detector = make_unique<Detector>("./models/" + model + "/weight.cfg", "./models/" + model + "/weight.weights");
}
void move_mouse(int cx, int cy, int c) {
	for(int i=0;i<c;i++)
		mouse_event(MOUSEEVENTF_MOVE, cx, cy, 0, 0);
	if (!TRIGGER) {
		mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0);
		Sleep(0.01);
		mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0);
	}
}
void aimbot(){
	init_screenshot();
	loadNet();
	chrono::high_resolution_clock::time_point start = chrono::high_resolution_clock::now();
	chrono::high_resolution_clock::time_point end = chrono::high_resolution_clock::now();
	chrono::microseconds duration = chrono::duration_cast<chrono::microseconds>(end-start);
	chrono::high_resolution_clock::time_point prev_flick = chrono::high_resolution_clock::now()- chrono::microseconds(FlickInterval);
	bool flick = false;
	while(true){
		if (GetAsyncKeyState(VK_DELETE) < 0)break;
		if (GetAsyncKeyState(VK_NUMPAD8 ) & 0x8000!= 0) {
			SHOW = !SHOW;
		}
		if (GetAsyncKeyState(VK_NUMPAD0) & 0x8000 != 0) {
			AIM = !AIM;
		}
		if (GetAsyncKeyState(VK_NUMPAD1) & 0x8000 != 0) {
			TRIGGER = !TRIGGER;
		}
		if (GetAsyncKeyState(VK_UP) & 0x8000 != 0 && AIM) {
			smooth += 1;
		}
		if (GetAsyncKeyState(VK_DOWN) & 0x8000 != 0 && AIM) {
			smooth -= 1;
			smooth = max(1, smooth);
		}
		if(GetAsyncKeyState(VK_XBUTTON1)&0x8000){
			start = chrono::high_resolution_clock::now();

			BitBlt(hdcMem, 0, 0, Screenshot_W, Screenshot_H, hdc, LEFT, TOP, SRCCOPY);
			double min_dis = 1e9;
			vector<double> closest={};
			float f;


			DetectionObject = detector->detect(frame, 0.85);
			for (int i = 0; i < DetectionObject.size(); i++) {
				double head_x = DetectionObject[i].x + DetectionObject[i].w / 2.0 ;
				double head_y = DetectionObject[i].y + DetectionObject[i].h / 4.0 ;
				double dx =  head_x - Screenshot_W / 2.0;
				double dy = head_y - Screenshot_H / 2.0;
				double dis = double(sqrtf(dx * dx + dy * dy));
				//circle(frame, Point(int(head_x), int(head_y)), MAX_DIS, Scalar(0, 0, 255), 10);
				if ((TRIGGER&&dis > MAX_TRACK_DIS)||(!TRIGGER&&dis>MAX_FLICK_DIS))continue;
				if (dis < min_dis) {
					min_dis = dis;
					closest = { dx, dy, dis };
					//closest = { int(f*dx * abs(dx) / (MAX_DIS + smooth)),int(f*dy * abs(dy) / (MAX_DIS + smooth)) ,int(DetectionObject[i].x + DetectionObject[i].w / 2.0), int(DetectionObject[i].y + DetectionObject[i].h / 4.0)};
				}
			}

			if (TRIGGER) {
				if (AIM && closest.size()) {
					if (closest[2] > 30)f = 0.5;
					else f = 1.8;
					int x = int(f * closest[0] * abs(closest[0]) / (MAX_TRACK_DIS + smooth));
					int y = int(f * closest[1] * abs(closest[1]) / (MAX_TRACK_DIS + smooth));
					thread(move_mouse, x, y, 4).detach();
				}
			}
			else {
				chrono::high_resolution_clock::time_point c = chrono::high_resolution_clock::now();
				if (AIM && chrono::duration_cast<chrono::microseconds>(c-prev_flick).count()>FlickInterval && closest.size()) {
					float mulx = 4.36 ;
					float muly = 4.35;

					int step = 30;
					thread(move_mouse, int(closest[0] *mulx / (2.66 / 2.5f) / step), int(closest[1]*muly / (2.66 / 2.5f) /step), step).detach();
					prev_flick = c;
				}
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