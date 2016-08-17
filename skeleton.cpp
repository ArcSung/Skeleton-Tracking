#include "skeleton.hpp"

#include <vector>
#include <string>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>

using namespace cv;
using namespace std;

Mat CalcuEDT(Mat DT, Point ref)
{
    int channels = DT.channels();
    DT.convertTo(DT, CV_8UC1, 255);
    Mat EDT = Mat(DT.size(), DT.type(), Scalar(0));
    int refValue = DT.at<unsigned char>(ref.y, ref.x);
    printf("DT.type:%d refValue %d x:%d y:%d\n",DT.type(), refValue, ref.x, ref.y);
    if(refValue != 0)
    {    
        for(int y = 0; y < DT.rows - 1; y++)
            for(int x = 0; x < DT.cols - 1; x++)
            {
                int Idp =  DT.at<unsigned char>(y, x);
                if(Idp != 0)
                {    
                    int diff = DT.at<unsigned char>(y, x)*(1 + abs(Idp - refValue)/refValue);
                    if(diff > 255)
                        diff = 255;
                    EDT.at<unsigned char>(y, x) = diff;
                }
            }   
    }
    imshow("EDT", EDT);
    return EDT;
}    

double CalcuDistance(Point P1, Point P2)
{
    return norm(P1 - P2);
}    

void detectAndDraw( Mat& img, CascadeClassifier& cascade,
                    double scale, Mat disp, Mat& mask)
{
    int i = 0;
    char str[30];
    vector<Rect> faces;
    const static Scalar colors[] =  { CV_RGB(0,0,255),
        CV_RGB(0,128,255),
        CV_RGB(0,255,255),
        CV_RGB(0,255,0),
        CV_RGB(255,128,0),
        CV_RGB(255,255,0),
        CV_RGB(255,0,0),
        CV_RGB(255,0,255)} ;
    Mat gray, smallImg( cvRound (img.rows/scale), cvRound(img.cols/scale), CV_8UC1 );

    cvtColor( img, gray, COLOR_BGR2GRAY );
    //threshold(disp, mask, 10, 255, THRESH_BINARY);
    //dilate(mask, mask, Mat());
    //gray &= mask;
    resize( gray, smallImg, smallImg.size(), 0, 0, INTER_LINEAR );
    equalizeHist( smallImg, smallImg );

    cascade.detectMultiScale( smallImg, faces,
        1.1, 2, 0
        //|CASCADE_FIND_BIGGEST_OBJECT
        //|CASCADE_DO_ROUGH_SEARCH
        |CASCADE_SCALE_IMAGE
        ,
        Size(30, 30) );
    for( vector<Rect>::const_iterator r = faces.begin(); r != faces.end(); r++, i++ )
    {
        BodySkeleton body_skeleton;
        Mat smallImgROI;
        Mat DT, EDT, Skin;
        Mat people = Mat::zeros(img.size(), CV_8UC3);
        vector<Rect> nestedObjects;
        Point center;
        Scalar color = colors[i%8];
        int radius = cvRound((r->width + r->height)*0.128*scale);
        double aspect_ratio = (double)r->width/r->height;

        center.x = cvRound((r->x + r->width*0.5)*scale);
        center.y = cvRound((r->y + r->height*0.5)*scale);
        //double Dist = GetDistance(center.x, center.y, disp, mask);
        //threshold(disp, mask, 10, 255,  THRESH_BINARY);
        fillContours(mask);
        findConnectComponent(mask, center.x, center.y);
        Skin = findSkinColor(img);
        //img.copyTo(people, mask);
        body_skeleton.head = Point(center.x, center.y);
        body_skeleton.neck = Point(center.x, center.y + r->height*0.6);
        body_skeleton.rShoulder = Point(center.x + r->width*0.9, center.y + r->height*0.9);
        body_skeleton.lShoulder = Point(center.x - r->width*0.9, center.y + r->height*0.9);
        findUpperBody( img, cascade2, scale, Rect(r->x, r->y, r->width, r->height), body_skeleton.rShoulder, body_skeleton.lShoulder);
        DT = findDistTran(mask);

        //find right arm
        EDT = CalcuEDT(DT, body_skeleton.rShoulder);
        body_skeleton.rElbow = findArm(EDT, body_skeleton.rShoulder, r->width*0.8, 0);
        body_skeleton.rHand = findHand(Skin, body_skeleton.rElbow, r->height*1.0);

        waitKey(0);
        //find left arm
        EDT = CalcuEDT(DT, body_skeleton.lShoulder);
        body_skeleton.lElbow = findArm(EDT, body_skeleton.lShoulder, r->width*0.8, 1);
        body_skeleton.lHand = findHand(Skin, body_skeleton.lElbow, r->height*1.0);

        rectangle( img, cvPoint(cvRound(r->x*scale), cvRound(r->y*scale)),
                    cvPoint(cvRound((r->x + r->width-1)*scale), cvRound((r->y + r->height-1)*scale)),
                    color, 3, 8, 0);
        line(img, body_skeleton.head,   body_skeleton.neck, color, 2, 1, 0);
        line(img, body_skeleton.neck,   body_skeleton.rShoulder, color, 2, 1, 0);
        line(img, body_skeleton.neck,   body_skeleton.lShoulder, color, 2, 1, 0);
        line(img, body_skeleton.rShoulder,   body_skeleton.rElbow, color, 2, 1, 0);
        line(img, body_skeleton.lShoulder,   body_skeleton.lElbow, color, 2, 1, 0);
        line(img, body_skeleton.rElbow,   body_skeleton.rHand, color, 2, 1, 0);
        line(img, body_skeleton.lElbow,   body_skeleton.lHand, color, 2, 1, 0);
        circle(img, body_skeleton.head, radius*0.2, Scalar(0, 255, 0), 2, 1, 0);
        circle(img, body_skeleton.neck, radius*0.2, Scalar(0, 255, 0), 2, 1, 0);
        circle(img, body_skeleton.rShoulder, radius*0.2, Scalar(255, 0, 0), 2, 1, 0);
        circle(img, body_skeleton.lShoulder, radius*0.2, Scalar(255, 0, 0), 2, 1, 0);
        circle(img, body_skeleton.rElbow, radius*0.2, Scalar(0, 255, 0), 2, 1, 0);
        circle(img, body_skeleton.lElbow, radius*0.2, Scalar(0, 255, 0), 2, 1, 0);
        circle(img, body_skeleton.rHand, radius*0.2, Scalar(0, 0, 255), 2, 1, 0);
        circle(img, body_skeleton.lHand, radius*0.2, Scalar(0, 0, 255), 2, 1, 0);
        //putText(img, str, cvPoint(cvRound(r->x*scale), cvRound(r->y*scale)), CV_FONT_HERSHEY_DUPLEX, 1, CV_RGB(0, 255, 0));
    }
}

int main(int argc, char* argv[])
{
    
    if( !cascade.load(cascadeName)){ printf("--(!)Error cascade\n"); return -1; };
    if( !cascade2.load(cascadeName2)){ printf("--(!)Error cascade2\n"); return -1; };

    Mat img, gray;
    Mat disp8, bin_mask;

    img = imread(argv[1]);
    cvtColor( img, gray, COLOR_BGR2GRAY );
    threshold(gray, bin_mask, 10, 255, THRESH_BINARY);
    //imshow("src", img);
    detectAndDraw(img, cascade, 1.0, disp8, bin_mask);
    imshow("after proc", img);
    waitKey(0);

    return 1;
}

Mat findDistTran(Mat bw)
{
    Mat dist;
    distanceTransform(bw, dist, CV_DIST_L2, 3);

    // Normalize the distance image for range = {0.0, 1.0}
    // so we can visualize and threshold it
    normalize(dist, dist, 0, 1, NORM_MINMAX);
    //namedWindow("Distance Transform Image", 0);
    //imshow("Distance Transform Image", dist);
    return dist;
}

void findUpperBody( Mat& img, CascadeClassifier& cascade,
                    double scale, Rect FaceRect, Point &lShoulder, Point &rShoulder)
{
    int i = 0;
    char str[30];
    vector<Rect> upbody;
    Mat gray, smallImg( cvRound (img.rows/scale), cvRound(img.cols/scale), CV_8UC1 );
    cvtColor( img, gray, COLOR_BGR2GRAY );
    resize( gray, smallImg, smallImg.size(), 0, 0, INTER_LINEAR );
    equalizeHist( smallImg, smallImg );

    cascade.detectMultiScale( smallImg, upbody,
        1.1, 2, 0
        |CASCADE_FIND_BIGGEST_OBJECT
        //|CASCADE_DO_ROUGH_SEARCH
        |CASCADE_SCALE_IMAGE
        ,
        Size(30, 30) );
    for( vector<Rect>::const_iterator r = upbody.begin(); r != upbody.end(); r++, i++ )
    {
        if(r->width > FaceRect.width && r->height > FaceRect.height)
        {
            printf("find upbody\n");
            rShoulder = Point(cvRound(r->x*scale + (r->width-1)*0.1), r->y*scale + (r->height-1)*0.9);
            lShoulder = Point(cvRound(r->x*scale + (r->width-1)*0.9), r->y*scale + (r->height-1)*0.9);
            /*rectangle( img, cvPoint(cvRound(r->x*scale), cvRound(r->y*scale)),
                        cvPoint(cvRound((r->x + r->width-1)*scale), cvRound((r->y + r->height-1)*scale)),
                        Scalar(0, 255, 255), 3, 8, 0);*/
        }    
    }
}

void fillContours(Mat &bw)
{
    // Another option is to use dilate/erode/dilate:
	int morph_operator = 1; // 0: opening, 1: closing, 2: gradient, 3: top hat, 4: black hat
	int morph_elem = 2; // 0: rect, 1: cross, 2: ellipse
	int morph_size = 10; // 2*n + 1
    int operation = morph_operator + 2;

    // Apply the specified morphology operation
    Mat element = getStructuringElement( morph_elem, Size( 2*morph_size + 1, 2*morph_size+1 ), Point( morph_size, morph_size ) );
    morphologyEx( bw, bw, operation, element );
    
    vector<vector<Point> > contours; // Vector for storing contour
    vector<Vec4i> hierarchy;
     
    findContours(bw, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE); // Find the contours in the image
 
    Scalar color(255);
    for(int i = 0; i < contours.size(); i++) // Iterate through each contour
    {
            drawContours(bw, contours, i, color, CV_FILLED, 8, hierarchy);
    }
}

void findConnectComponent(Mat &bw, int x, int y)
{
    Mat labelImage(bw.size(), CV_32S);
    int nLabels = connectedComponents(bw, labelImage, 8);
    int label = labelImage.at<int>(x, y);

    if(label > 0)
    {    
        inRange(labelImage, Scalar(label), Scalar(label), bw);
        threshold(bw, bw, 1, 255, THRESH_BINARY);
    }    

}

Mat findSkinColor(Mat src)
{
    Mat bgr2ycrcbImg, ycrcb2skinImg;
    cvtColor( src, bgr2ycrcbImg, cv::COLOR_BGR2YCrCb );
    inRange( bgr2ycrcbImg, cv::Scalar(80, 135, 85), cv::Scalar(255, 180, 135), ycrcb2skinImg );
    erode(ycrcb2skinImg, ycrcb2skinImg, Mat());
    dilate(ycrcb2skinImg, ycrcb2skinImg, Mat());
    //fillContours(ycrcb2skinImg);
    std::vector<std::vector<Point> > contours;
    std::vector<Vec4i> hierarchy;

    findContours(ycrcb2skinImg,
    contours,
    hierarchy,
    RETR_TREE,
    CHAIN_APPROX_SIMPLE);

    Mat drawing = Mat::zeros( src.size(), CV_8UC1 );

    for( int i = 0; i< contours.size(); i++ ) // iterate through each contour. 
    {
        double a=contourArea( contours[i],false);  //  Find the area of contour
        if(a>200)
        {
            drawContours(drawing, contours, i, Scalar(255), CV_FILLED, 8, hierarchy);
        }
    } 
    //imshow("skin color", drawing);

    return drawing;
}    

Point findArm(Mat EDT, Point lShoulder, int fheight, int findLeftelbow)
{
    float Slope = 0;
    float refValue = EDT.at<unsigned char>(lShoulder.x, lShoulder.y);
    Point elbow = lShoulder;

    Mat proc;
    inRange(EDT, Scalar(refValue - 3 > 0? refValue -3 : 2), Scalar(refValue + 3), proc);
    imshow("proc", proc);

    for(int i = 0; i < 6; i++)
    {
        bool find = false; 
        Point search;
        float TempSlope = 0;
        for(int y = elbow.y + fheight/5; y > elbow.y - fheight/5; y--)
        {    
           if(findLeftelbow == 1)
           {   
               for(int x = elbow.x - fheight/5; x < elbow.x + fheight/5; x++)
               {
                  if(proc.at<unsigned char>(y, x) != 0)
                  {
                       search = Point(x, y);
                       find = true;
                       break;
                  }    
               }
           }
           else
           {
               for(int x = elbow.x + fheight/5; x > elbow.x - fheight/5; x--)
               {
                  if(proc.at<unsigned char>(y, x) != 0)
                  {
                       search = Point(x, y);
                       find = true;
                       break;
                  }    
               }
            }   
               

           if(find == true)
           {
              if(search.y - elbow.y !=0) 
                TempSlope = (float)(search.x - elbow.x)/(search.y - elbow.y); 
              else
                TempSlope = search.x - elbow.x >= 0 ? 0.5 : -0.5;
              break;
           }    
        }

        printf("Slope %f, TempSlope %f\n", Slope, TempSlope);

        if(find == true)
        {
            Slope = TempSlope;
            elbow = search;
            if(abs(Slope - TempSlope) > 0.4)
                break;     
        }    
    }    

    printf("lelbow %d, %d\n", elbow.x, elbow.y);
    return elbow;

}    

Point findHand(Mat Skin, Point rElbow, int FWidth)
{
    Point rHand = rElbow;
    Mat labelImage(Skin.size(), CV_32S);
    int nLabels = connectedComponents(Skin, labelImage, 8);
    int label;
    int minD = FWidth;
    int maxD = 0;
    int procD = 0;

    //find the most close area
    for(int x = rElbow.x - FWidth; x < rElbow.x + FWidth; x++)
        for(int y = rElbow.y - FWidth; y < rElbow.y + FWidth; y++)
        {
            if(labelImage.at<int>(y,x) != 0)
            {    
                procD =CalcuDistance(rElbow, Point(x,y));
                if(procD < minD)
                {    
                    minD = procD;
                    label = labelImage.at<int>(y,x);
                }        
            }    
        }    

    //find the most far point of the most close area
    for(int x = rElbow.x - FWidth; x < rElbow.x + FWidth; x++)
        for(int y = rElbow.y - FWidth; y < rElbow.y + FWidth; y++)
        {
            if(labelImage.at<int>(y,x) == label)
            {    
                procD =CalcuDistance(rElbow, Point(x,y));
                if(procD > maxD)
                {    
                    maxD = procD;
                    rHand = Point(x,y);
                }        
            }    
        }    
  printf("hand %d, %d\n", rHand.x, rHand.y);  

  return rHand;
}    
