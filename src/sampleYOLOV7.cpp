#include <dirent.h>
#include <opencv2/opencv.hpp>
#include "AclLiteUtils.h"
#include "AclLiteImageProc.h"
#include "AclLiteResource.h"
#include "AclLiteError.h"
#include "AclLiteModel.h"
#include <chrono>
#include "acl/acl.h"

#include "mat_warper.h"
#ifndef DEMO_PYBIND11_SRC_DEMO_H_
#define DEMO_PYBIND11_SRC_DEMO_H_
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <Python.h> // Add the missing include for Python.h
namespace py = pybind11;
#endif // !DEMO_PYBIND11_SRC_DEMO_H_

using namespace std;
using namespace cv;
typedef enum Result {
    SUCCESS = 0,
    FAILED = 1
} Result;

typedef struct BoundBox {
    float x;
    float y;
    float width;
    float height;
    float score;
    size_t classIndex;
    size_t index;
} BoundBox;

bool sortScore(BoundBox box1, BoundBox box2)
{
    return box1.score > box2.score;
}




class sampleYOLOV7 {
    public:
    sampleYOLOV7(const char *modelPath,int model_size = 640, float confidenceThreshold = 0.25, float NMSThreshold = 0.45, int classNum = 80);
    Result InitResource();
    Result ProcessInput(cv::Mat srcImage);
    Result detect(std::vector<InferenceOutput>& inferOutputs);
    vector<BoundBox> GetResult(std::vector<InferenceOutput>& inferOutputs);
    vector<BoundBox> inference();
    py::list inference_py(py::array_t<unsigned char> input);
    ~sampleYOLOV7();
    void ReleaseResource();
    AclLiteResource aclResource_;
    AclLiteModel model_;
    aclrtRunMode runMode_;
    ImageData resizedImage_;
    
    const char *modelPath_;
    int32_t modelWidth_;
    int32_t modelHeight_;
    float confidenceThreshold_;
    float NMSThreshold_;
    int classNum_;
    int srcWidth_;
    int srcHeight_;
    cv::Mat ori_img;
    std::vector<InferenceOutput> inferOutputs_;
};

sampleYOLOV7::sampleYOLOV7(const char *modelPath,int model_size,float confidenceThreshold, float NMSThreshold, int classNum)          
{
    modelPath_ = modelPath;
    modelWidth_ = model_size;
    modelHeight_ = model_size;
    confidenceThreshold_ = confidenceThreshold;
    NMSThreshold_ = NMSThreshold;
    classNum_ = classNum;
    Result ret = this->InitResource();
    if (ret == FAILED) {
        ACLLITE_LOG_ERROR("InitResource failed, errorCode is %d", ret);
    }
}

sampleYOLOV7::~sampleYOLOV7()
{
    ReleaseResource();
}
Result sampleYOLOV7::InitResource()
{
    AclLiteError ret = aclResource_.Init();
    if (ret == FAILED) {
        ACLLITE_LOG_ERROR("resource init failed, errorCode is %d", ret);
        return FAILED;
    }
    ret = aclrtGetRunMode(&runMode_);
    if (ret == FAILED) {
        ACLLITE_LOG_ERROR("get runMode failed, errorCode is %d", ret);
        return FAILED;
    }
    // load model from file
    ret = model_.Init(modelPath_);
    if (ret == FAILED) {
        ACLLITE_LOG_ERROR("model init failed, errorCode is %d", ret);
        return FAILED;
    }
    return SUCCESS;
}

Result sampleYOLOV7::ProcessInput(cv::Mat srcImage)
{
    if (srcImage.cols <= 0 || srcImage.rows <= 0) {
        ACLLITE_LOG_ERROR("srcImage is empty");
        return FAILED;
    }
    srcWidth_ = srcImage.cols;
    srcHeight_ = srcImage.rows;
    cv::resize(srcImage, srcImage, cv::Size(modelWidth_, modelHeight_));
    // uint32_t bufLen = modelHeight_ * modelWidth_ * 3 / 2;
	// cv::Mat yuvImg(bufLen, modelWidth_, CV_8UC1);
	// cvtColor(srcImage, yuvImg, cv::COLOR_BGR2YUV_I420);
    int width = srcImage.cols;
    int height = srcImage.rows;

    cv::Mat img_nv12(height * 3 / 2, width, CV_8UC1);
    cv::Mat yuv_mat;
    cv::cvtColor(srcImage, yuv_mat, cv::COLOR_BGR2YUV_I420);
    uint8_t *yuv = yuv_mat.ptr<uint8_t>();
    
    uint8_t *ynv12 = img_nv12.ptr<uint8_t>();

    int32_t uv_height = height / 2;
    int32_t uv_width = width / 2;

    int32_t y_size = height * width;
    memcpy(ynv12, yuv, y_size);
    //printf("y_size:%d\n",y_size);

    uint8_t *nv12 = ynv12 + y_size;
    uint8_t *u_data = yuv + y_size;
    uint8_t *v_data = u_data + uv_height * uv_width;

    for (int32_t i = 0; i < uv_width * uv_height; i++) {
        *nv12++ = *u_data++;
        *nv12++ = *v_data++;
    }
    
    int32_t yuv_size = y_size + 2 * uv_height * uv_width;
    //printf("yuv_size:%d\n",yuv_size);
    
    uint32_t imageInfoSize = YUV420SP_SIZE(modelWidth_, modelHeight_);
    void *imageInfoBuf =  CopyDataToDevice(img_nv12.data, imageInfoSize,runMode_, MEMORY_DVPP);
    ImageData image;
    image.data = SHARED_PTR_DVPP_BUF(imageInfoBuf);
    image.size = imageInfoSize;
    image.width = modelWidth_;
    image.height = modelHeight_;
    image.format = PIXEL_FORMAT_YUV_SEMIPLANAR_420;
    resizedImage_ = image;
    return SUCCESS;
}

Result sampleYOLOV7::detect(std::vector<InferenceOutput>& inferOutputs)
{
    AclLiteError ret = model_.CreateInput(static_cast<void *>(resizedImage_.data.get()), resizedImage_.size);
    if (ret == FAILED) {
        ACLLITE_LOG_ERROR("CreateInput failed, errorCode is %d", ret);
        return FAILED;
    }

    ret = model_.Execute(inferOutputs);
    if (ret != ACL_SUCCESS) {
        ACLLITE_LOG_ERROR("execute model failed, errorCode is %d", ret);
        return FAILED;
    }
    return SUCCESS;
}

vector<BoundBox> sampleYOLOV7::GetResult(std::vector<InferenceOutput> &inferOutputs)
{
    uint32_t outputDataBufId = 0;
    float *classBuff = static_cast<float *>(inferOutputs[outputDataBufId].data.get());
    size_t offset = 4;
    // total number of boxs yolov8 [1,84,8400]
    size_t modelOutputBoxNum = (int)8400*(modelWidth_/640)*(modelHeight_/640);
    vector<BoundBox> boxes;
    size_t yIndex = 1;
    size_t widthIndex = 2;
    size_t heightIndex = 3;
    for (size_t i = 0; i < modelOutputBoxNum; ++i)
    {

        float maxValue = 0;
        size_t maxIndex = 0;
        for (size_t j = 0; j < (size_t)classNum_; ++j)
        {

            float value = classBuff[(offset + j) * modelOutputBoxNum + i];
            if (value > maxValue)
            {
                // index of class
                maxIndex = j;
                maxValue = value;
            }
        }

        if (maxValue > confidenceThreshold_)
        {
            BoundBox box;
            box.x = classBuff[i] * srcWidth_ / modelWidth_;
            box.y = classBuff[yIndex * modelOutputBoxNum + i] * srcHeight_ / modelHeight_;
            box.width = classBuff[widthIndex * modelOutputBoxNum + i] * srcWidth_ / modelWidth_;
            box.height = classBuff[heightIndex * modelOutputBoxNum + i] * srcHeight_ / modelHeight_;
            box.score = maxValue;
            box.classIndex = maxIndex;
            box.index = i;
            if (maxIndex < (size_t)classNum_)
            {
                boxes.push_back(box);
            }
        }
    }
    // filter boxes by NMS
    vector<BoundBox> result;
    result.clear();
    float NMSThreshold = NMSThreshold_;
    int32_t maxLength = modelWidth_ > modelHeight_ ? modelWidth_ : modelHeight_;
    std::sort(boxes.begin(), boxes.end(), sortScore);
    BoundBox boxMax;
    BoundBox boxCompare;
    while (boxes.size() != 0)
    {
        size_t index = 1;
        result.push_back(boxes[0]);
        while (boxes.size() > index)
        {
            boxMax.score = boxes[0].score;
            boxMax.classIndex = boxes[0].classIndex;
            boxMax.index = boxes[0].index;

            // translate point by maxLength * boxes[0].classIndex to
            // avoid bumping into two boxes of different classes
            boxMax.x = boxes[0].x + maxLength * boxes[0].classIndex;
            boxMax.y = boxes[0].y + maxLength * boxes[0].classIndex;
            boxMax.width = boxes[0].width;
            boxMax.height = boxes[0].height;

            boxCompare.score = boxes[index].score;
            boxCompare.classIndex = boxes[index].classIndex;
            boxCompare.index = boxes[index].index;

            // translate point by maxLength * boxes[0].classIndex to
            // avoid bumping into two boxes of different classes
            boxCompare.x = boxes[index].x + boxes[index].classIndex * maxLength;
            boxCompare.y = boxes[index].y + boxes[index].classIndex * maxLength;
            boxCompare.width = boxes[index].width;
            boxCompare.height = boxes[index].height;

            // the overlapping part of the two boxes
            float xLeft = max(boxMax.x, boxCompare.x);
            float yTop = max(boxMax.y, boxCompare.y);
            float xRight = min(boxMax.x + boxMax.width, boxCompare.x + boxCompare.width);
            float yBottom = min(boxMax.y + boxMax.height, boxCompare.y + boxCompare.height);
            float width = max(0.0f, xRight - xLeft);
            float hight = max(0.0f, yBottom - yTop);
            float area = width * hight;
            float iou = area / (boxMax.width * boxMax.height + boxCompare.width * boxCompare.height - area);

            // filter boxes by NMS threshold
            if (iou > NMSThreshold)
            {
                boxes.erase(boxes.begin() + index);
                continue;
            }
            ++index;
        }
        boxes.erase(boxes.begin());
    }
    //x,y,wight,height,score,classIndex
    return result;
}

//输入图片，输出推理结果
vector<BoundBox> sampleYOLOV7::inference()
{
    vector<BoundBox> boxes;
    AclLiteError ret =this->ProcessInput(this->ori_img);
    if (ret == FAILED) {
        ACLLITE_LOG_ERROR("ProcessInput image failed, errorCode is %d", ret);
        return boxes;
    }
    ret = this->detect(inferOutputs_);
    if (ret == FAILED) {
        ACLLITE_LOG_ERROR("Inference failed, errorCode is %d", ret);
        return boxes;
    }
    
    boxes = this->GetResult(inferOutputs_);
    return boxes;
}


void sampleYOLOV7::ReleaseResource()
{
    model_.DestroyResource();
    aclResource_.Release();
}

py::list sampleYOLOV7::inference_py(py::array_t<unsigned char> input)
{
    cv::Mat mat = numpy_uint8_3c_to_cv_mat(input);
    ori_img = mat;
    vector<BoundBox> boxes = this->inference();
    py::list result;
    for (size_t i = 0; i < boxes.size(); i++)
    {
        BoundBox box = boxes[i];
        py::list item;
        item.append(box.classIndex);
        item.append(box.x);
        item.append(box.y);
        item.append(box.width);
        item.append(box.height);
        item.append(box.score);
        result.append(item);
    }
    return result;
}


// PYBIND11_MODULE(yolov8_runner_pool, m)
// {
//     m.doc() = "yolov8 runner pool";
//     py::class_<yolov8_runner_pool>(m, "yolov8_runner_pool")
//         .def(py::init<std::string&,int,float,float,int>(), py::arg("model_path"),py::arg("thread_num"),py::arg("box_conf_threshold"),py::arg("nms_threshold"),py::arg("class_num"))
//         .def("inference", &yolov8_runner_pool::inference_py,py::return_value_policy::move)
//         .def("inference_all", &yolov8_runner_pool::inference_all_py,py::return_value_policy::move);
// }

PYBIND11_MODULE(sampleYOLOV7, m)
{
    m.doc() = "sampleYOLV7,return classIndex,x_center,y_center,width,height,score";
    py::class_<sampleYOLOV7>(m, "sampleYOLOV7")
        .def(py::init<const char*,int,float,float,int>(), py::arg("model_path"),py::arg("model_size"),py::arg("confidenceThreshold"),py::arg("NMSThreshold"),py::arg("classNum"))
        .def("inference", &sampleYOLOV7::inference_py,py::return_value_policy::move);
}
// int main()
// {
//     const char* modelPath = "/home/HwHiAiUser/Desktop/sampleYOLOV7/yolov8n_normal.om";
//     const string imagePath = "/home/HwHiAiUser/Desktop/sampleYOLOV7/dog1_1024_683.jpg";
//     sampleYOLOV7 sampleYOLO(modelPath, 640,0.25, 0.45, 80);
//     //循环100次并计算平均时间
//     cv::Mat srcImage = cv::imread(imagePath);
//     auto start = std::chrono::high_resolution_clock::now();
//     sampleYOLO.ori_img = srcImage;
//     for (int i = 0; i < 1000; i++) {
//         vector<BoundBox> boxes = sampleYOLO.inference();
//         for (size_t i = 0; i < boxes.size(); i++) {
//             BoundBox box = boxes[i];
//             //cout << box.x << " " << box.y << " " << box.width << " " << box.height << " " << box.score << " " << box.classIndex << endl;
//         }
//     }


//     auto end = std::chrono::high_resolution_clock::now();
//     std::chrono::duration<double, std::milli> elapsed = end - start;
//     std::cout << "Elapsed time: " << elapsed.count() / 1000 << " ms\n";

//     return SUCCESS;
// }