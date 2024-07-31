#include <dirent.h>
#include <opencv2/opencv.hpp>
#include "AclLiteUtils.h"
#include "AclLiteImageProc.h"
#include "AclLiteResource.h"
#include "AclLiteError.h"
#include "AclLiteModel.h"
#include <chrono>
#include "acl/acl.h"
#include "ThreadPool.hpp"

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

class SampleYOLOV7 {
    public:
    SampleYOLOV7(const char *modelPath,int model_size = 640, float confidenceThreshold = 0.25, float NMSThreshold = 0.45, int classNum = 80);
    Result InitResource();
    Result ProcessInput(cv::Mat srcImage);
    Result detect(std::vector<InferenceOutput>& inferOutputs);
    vector<BoundBox> GetResult(std::vector<InferenceOutput>& inferOutputs);
    vector<BoundBox> inference();
    ~SampleYOLOV7();
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

SampleYOLOV7::SampleYOLOV7(const char *modelPath,int model_size,float confidenceThreshold, float NMSThreshold, int classNum)          
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

SampleYOLOV7::~SampleYOLOV7()
{
    ReleaseResource();
}
Result SampleYOLOV7::InitResource()
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

Result SampleYOLOV7::ProcessInput(cv::Mat srcImage)
{
    if (srcImage.cols <= 0 || srcImage.rows <= 0) {
        ACLLITE_LOG_ERROR("srcImage is empty");
        return FAILED;
    }
    srcWidth_ = srcImage.cols;
    srcHeight_ = srcImage.rows;
    cv::resize(srcImage, srcImage, cv::Size(modelWidth_, modelHeight_));
    uint32_t bufLen = modelHeight_ * modelWidth_ * 3 / 2;
	cv::Mat yuvImg(bufLen, modelWidth_, CV_8UC1);
	cvtColor(srcImage, yuvImg, cv::COLOR_BGR2YUV_I420);
    
    uint32_t imageInfoSize = YUV420SP_SIZE(modelWidth_, modelHeight_);
    void *imageInfoBuf =  CopyDataToDevice(yuvImg.data, imageInfoSize,runMode_, MEMORY_DVPP);
    ImageData image;
    image.data = SHARED_PTR_DVPP_BUF(imageInfoBuf);
    image.size = imageInfoSize;
    image.width = modelWidth_;
    image.height = modelHeight_;
    image.format = PIXEL_FORMAT_YUV_SEMIPLANAR_420;
    resizedImage_ = image;
    return SUCCESS;
}

Result SampleYOLOV7::detect(std::vector<InferenceOutput>& inferOutputs)
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

vector<BoundBox> SampleYOLOV7::GetResult(std::vector<InferenceOutput> &inferOutputs)
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
vector<BoundBox> SampleYOLOV7::inference()
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


void SampleYOLOV7::ReleaseResource()
{
    model_.DestroyResource();
    aclResource_.Release();
}

//类比下面的rknn多线程代码写出ascend310的多线程代码
class yolov8_runner_pool
{
    private:
    vector<BoundBox> od_results;
    int class_num = 80;
    int put_count = 0;
    int if_init = 0;
    vector<SampleYOLOV7*> ascpool;
    dpool::ThreadPool* pool;
    queue<future<vector<BoundBox>>> futs;
    //model_path,thread_num
    const char* model_path;
    int thread_num;
    float box_conf_threshold;
    float nms_threshold;
    int model_size;
    public:
        yolov8_runner_pool(const char* model_p,int thread_num = 3,float box_conf_threshold = 0.45, float nms_threshold = 0.5,int class_num = 80,int model_size = 640)
        {
            this->model_path = model_p;
            this->thread_num = thread_num;
            this->pool = new dpool::ThreadPool(thread_num);
            this->box_conf_threshold = box_conf_threshold;
            this->nms_threshold = nms_threshold;
            this->class_num = class_num;
            this->model_size = model_size;
            just_init();
        }
        ~yolov8_runner_pool()
        {
            while (!futs.empty())
            {
                futs.pop();
            }
            for (int i = 0; i < (int)ascpool.size(); i++)
            {
                delete ascpool[i];
            }
        }
        void just_init()
        {
            for (int i = 0; i < thread_num; i++)
            {
                SampleYOLOV7 *ptr = new SampleYOLOV7(model_path,model_size,box_conf_threshold,nms_threshold,class_num);
                this->ascpool.push_back(ptr);
                printf("=============>>init %d<<=============\n",i);
            }
            printf("=============>>init finish<<=============\n");
        }
        void put_img(cv::Mat &input)
        {
            if (put_count >= thread_num)
            {
                if_init = 1;
                put_count %= thread_num;
            }
            ascpool[put_count]->ori_img = input;
            futs.push(pool->submit(&SampleYOLOV7::inference, ascpool[put_count++% thread_num]));
        }

        vector<BoundBox> return_error()
        {
            vector<BoundBox> od_results;
            return od_results;
        }
        vector<BoundBox> get_od_results()
        {
            if (futs.empty())
            {
                printf("get error no task to inference\n");
                return return_error();
            }
            od_results = futs.front().get();
            futs.pop();
            return od_results;
        }
        vector<BoundBox> inference(cv::Mat &input)
        {
            if (this->put_count < thread_num && if_init == 0)
            {
                put_img(input);
                return return_error();
            }
            else{
                od_results = get_od_results();
                put_img(input);
                return od_results;
            }
        }
        std::vector<vector<BoundBox>> inference_all(std::vector<cv::Mat> &inputs)
        {
            std::vector<vector<BoundBox>> od_results_list;
            for (int i = 0; i < (int)inputs.size(); i++)
            {
                if (this->put_count < thread_num)
                {
                    put_img(inputs[i]);
                }
                else
                {
                    od_results = get_od_results();
                    put_img(inputs[i]);
                    od_results_list.push_back(od_results);
                }
            }
            while (!futs.empty())
            {
                od_results = futs.front().get();
                futs.pop();
                od_results_list.push_back(od_results);
            }
            this->put_count = 0;
            return od_results_list;
        }
};


int main()
{
    const char* modelPath = "/home/HwHiAiUser/Desktop/sampleYOLOV7/yolov8n_normal.om";
    const string imagePath = "/home/HwHiAiUser/Desktop/sampleYOLOV7/dog1_1024_683.jpg";
    //SampleYOLOV7 sampleYOLO(modelPath, 640,0.25, 0.45, 80);
    yolov8_runner_pool sampleYOLO(modelPath, 1,0.25, 0.45, 80,640);
    //循环100次并计算平均时间
    cv::Mat srcImage = cv::imread(imagePath);
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 1000; i++) 
    {
        cv::Mat copyed = srcImage.clone();
        vector<BoundBox> boxes = sampleYOLO.inference(copyed);
        for (size_t i = 0; i < boxes.size(); i++) 
        {
            BoundBox box = boxes[i];
            cout << box.x << " " << box.y << " " << box.width << " " << box.height << " " << box.score << " " << box.classIndex << endl;
        }
        // vector<BoundBox> boxes = sampleYOLO.inference();
        // for (size_t i = 0; i < boxes.size(); i++) 
        // {
        //     BoundBox box = boxes[i];
        //     cout << box.x << " " << box.y << " " << box.width << " " << box.height << " " << box.score << " " << box.classIndex << endl;
        // }
    }


    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    std::cout << "Elapsed time: " << elapsed.count() / 1000 << " ms\n";

    return SUCCESS;
}