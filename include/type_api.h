#define RGBU8_IMAGE_SIZE(width, height) ((width) * (height) * 3)
#define RGBAU8_IMAGE_SIZE(width, height) ((width) * (height) * 4)

typedef enum  {
    SUCCESS = 0,
    FAILED = 1
}Result;
typedef struct  {
    float x;
    float y;
    float w;
    float h;
    float score;
    int classIndex;
    int index; // index of output buffer
} BBox;

typedef struct BoundBox {
    float x;
    float y;
    float width;
    float height;
    float score;
    size_t classIndex;
    size_t index;
} BoundBox;
