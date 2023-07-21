#include "../common/common.hpp"
#include "../common/RenderImage.hpp"
#include "../include/Mv3dRgbdApi.h"	
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;

void* handle = NULL;
RIFrameInfo depth = { 0 };
RIFrameInfo rgb = { 0 };
CascadeClassifier faceCascade;
float FACE_OVERLAP_RATE = 0.8;//�ص������ã���Χ0~1

//��ʼ��+�����豸
void HIK_initialization()
{
    MV3D_RGBD_VERSION_INFO stVersion;
    ASSERT_OK(MV3D_RGBD_GetSDKVersion(&stVersion));
    LOGD("dll version: %d.%d.%d", stVersion.nMajor, stVersion.nMinor, stVersion.nRevision);

    ASSERT_OK(MV3D_RGBD_Initialize());

    unsigned int nDevNum = 0;
    ASSERT_OK(MV3D_RGBD_GetDeviceNumber(DeviceType_Ethernet | DeviceType_USB, &nDevNum));
    LOGD("MV3D_RGBD_GetDeviceNumber success! nDevNum:%d.", nDevNum);
    ASSERT(nDevNum);

    // �����豸
    std::vector<MV3D_RGBD_DEVICE_INFO> devs(nDevNum);
    ASSERT_OK(MV3D_RGBD_GetDeviceList(DeviceType_Ethernet | DeviceType_USB, &devs[0], nDevNum, &nDevNum));
    for (unsigned int i = 0; i < nDevNum; i++)
    {
        LOG("Index[%d]. SerialNum[%s] IP[%s] name[%s].\r\n", i, devs[i].chSerialNumber, devs[i].SpecialInfo.stNetInfo.chCurrentIp, devs[i].chModelName);
    }

    //���豸
    unsigned int nIndex = 0;
    ASSERT_OK(MV3D_RGBD_OpenDevice(&handle, &devs[nIndex]));
    LOGD("OpenDevice success.");

    //�ı�ֱ��ʲ�����0x00010001Ϊ 1280��720�� 0x00020002Ϊ 640��360
    //MV3D_RGBD_PARAM stparam;
    //stparam.enParamType = ParamType_Enum;
    //stparam.ParamInfo.stEnumParam.nCurValue = 0x00010001;
    //ASSERT_OK(MV3D_RGBD_SetParam(handle, MV3D_RGBD_ENUM_RESOLUTION, &stparam));

      //�ı��ع����
      //MV3D_RGBD_PARAM stparam;
      //stparam.enParamType = ParamType_Float;;
      //stparam.ParamInfo.stFloatParam.fCurValue = 100.0000;
      //ASSERT_OK(MV3D_RGBD_SetParam(handle, MV3D_RGBD_FLOAT_EXPOSURETIME, &stparam));
      //LOGD("EXPOSURETIME: (%f)", stparam.ParamInfo.stFloatParam.fCurValue);

    // ��ʼ��������
    ASSERT_OK(MV3D_RGBD_Start(handle));
    LOGD("Start work success.");
}

//�ر��ͷ��豸
void HIK_stop()
{
    ASSERT_OK(MV3D_RGBD_Stop(handle));
    ASSERT_OK(MV3D_RGBD_CloseDevice(&handle));
    ASSERT_OK(MV3D_RGBD_Release());

    LOGD("Main done!");
}

//�ص���� ȫ�����ص�����pass
void isOverLap(std::vector<Rect> faces)
{
    //������faces[i].x,faces[i].y,faces[i].width,faces[i].height�������¹�ϵʱ�����������ص���Ҫ���������һ���������ص��� FACE_OVERLAP_RATE, ȡֵΪ0��1
    for (int i = 0; i < faces.size(); i++)
    {
        for (int j = i + 1; j < faces.size(); j++)
        {
            if ((faces[i].x + faces[i].width * FACE_OVERLAP_RATE > faces[j].x) &&
                (faces[j].x + faces[j].width * FACE_OVERLAP_RATE > faces[i].x) &&
                (faces[i].y + faces[i].height * FACE_OVERLAP_RATE > faces[j].y) &&
                (faces[j].y + faces[j].height * FACE_OVERLAP_RATE > faces[i].y)
                )
            {
                LOGD("A--L--A--R--M!!!");
                return;
            }
        }
    }
    LOGD("PASS");
}

int main(int argc, char** argv)
{

    HIK_initialization();

    MV3D_RGBD_FRAME_DATA stFrameData = { 0 };

    int t = 1;

    while (1)
    {
        // ��ȡͼ������
        int nRet = MV3D_RGBD_FetchFrame(handle, &stFrameData, 5000);
        if (MV3D_RGBD_OK == nRet)
        {
            LOGD("MV3D_RGBD_FetchFrame success.");

            //������ȡÿ֡����
            parseFrame(&stFrameData, &depth, &rgb);
            Mat rgb_frame(rgb.nHeight, rgb.nWidth, CV_8UC3, rgb.pData);


            LOGD("====================== Parsing(%d) ========================", t++);
            LOGD("rgb: FrameNum(%d), height(%d), width(%d)", rgb.nFrameNum, rgb.nHeight, rgb.nWidth);

            //B��Rͨ����������ʾ������ɫͼ��
            Mat frame, gray_frame;
            cvtColor(rgb_frame, frame, COLOR_BGR2RGB);
            cvtColor(rgb_frame, gray_frame, COLOR_BGR2GRAY);

            //ԭʼrgbͼ��
            //imshow("src_face", frame);

            //���½�������ʶ��
            faceCascade.load("haarcascade_frontalface_default.xml");

            if (faceCascade.empty())
            {
                LOGD("xml not found!");
            }

            //����ֱ��ͼ���⻯
            equalizeHist(gray_frame, gray_frame);

            //ִ�ж�߶�������⣬���Ե������ʶ��׼ȷ��
            vector<Rect> faces;
            faceCascade.detectMultiScale(gray_frame, faces, 1.1, 3, 0, Size(24, 24));

            if (faces.size() == 0) {
                LOGD("Face Recognition Failed");
            }

            for (int i = 0; i < faces.size(); i++)
            {
                rectangle(frame, faces[i].tl(), faces[i].br(), Scalar(255, 255, 255), 4);
                LOGD("catch_face: x(%d), y(%d), w(%d), h(%d)", faces[i].x, faces[i].y, faces[i].width, faces[i].height);

                //�ص����
                isOverLap(faces);

                //���ʶ�𵽵�����
                char text[10];
                sprintf_s(text, "%s%d", "face:", i);
                putText(frame, text, Point(faces[i].x, faces[i].y), FONT_HERSHEY_COMPLEX, 0.5, Scalar(0, 0, 255));

            }

            //��ʾ������Ŀ
            namedWindow("HIK_face", 0);
            resizeWindow("HIK_face", 640, 360);
            char text[10];
            sprintf_s(text, "%s%zd", "Nums:", faces.size());
            putText(frame, text, Point(10, 50), FONT_HERSHEY_SIMPLEX, 2, Scalar(255, 0, 0), 4);
            imshow("HIK_face", frame);
            waitKey(0);
        }
    }

    HIK_stop();

    return  0;
}