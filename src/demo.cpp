// GridMatch.cpp : Defines the entry point for the console application.

//#define USE_GPU 

#include "Header.h"
#include "gms_matcher.h"
#include<thread>
#include <opencv2/sfm.hpp>
#include <opencv2/viz.hpp>

using namespace cv::sfm;
void GmsMatch(Mat &img1, Mat &img2);

void runImagePair(int argc, char ** argv){
	
	Mat img1,img2;
	if(argc<3){
	img1 = imread("../data/Left04.jpg");
	img2 = imread("../data/Right04.jpg");
	}
	else
	{
		img1 = imread(argv[1]);
		img2 = imread(argv[2]);
	}
	imresize(img1, 480);
	imresize(img2, 480);

	GmsMatch(img1, img2);
}

void run(int a)
{
    cout<<"Oh mama!"<<endl;
    cout<<a<<endl;
}

int main(int argc, char ** argv)
{
#ifdef USE_GPU
	int flag = cuda::getCudaEnabledDeviceCount();
	if (flag != 0){ cuda::setDevice(0); }
#endif // USE_GPU

	runImagePair(argc, argv);

	return 0;
}

void triangulation(vector<KeyPoint>& kp1,vector<KeyPoint>& kp2,vector<DMatch> &inlier, vector<Vec3d>& points3d)
{
    double bf = 37.341142001156200, f = 622.11766490376556, cx = 330.22266006469727, cy =211.32878684997559;
    Vec3d points3d2;
    for(size_t i = 0; i < inlier.size(); i++)
    {
        Point2f left = kp1[inlier[i].queryIdx].pt;
        Point2f right = kp2[inlier[i].trainIdx].pt;
        float d=left.x-right.x;//+abs(pointsl(1,i)-pointsr(1,i));
        points3d2(2)=abs(bf/d);
        points3d2(1)=points3d2(2)*(left.x-cx)/f;//b*pointsl(0,i)/d;335
        points3d2(0)=points3d2(2)*(right.y-cy)/f;//b*pointsl(1,i)/d;235
        cout<<points3d2<<endl;
        if(points3d2.ddot(points3d2)>3)continue;
        points3d.push_back(points3d2);
    }
    cout<<"sucess"<<endl;
}

void visualize3Dpoints(vector<Vec3d> &point_cloud_est)
{
    /// Create 3D windows
    viz::Viz3d window_est("Estimation Coordinate Frame");
    window_est.setBackgroundColor(viz::Color::black()); // black by default
//    window_est.setWindowPosition(Point(150,150));

  /// Wait for key 'q' to close the window
    cout << endl << "Press:                       " << endl;
    cout <<         " 'q' to close the windows    " << endl;

    if (1)// path_est.size() > 0 )
    {

        while(!window_est.wasStopped())
        {
            int N=point_cloud_est.size();
            /// Render points as 3D cubes
            viz::WCloud cloud_widget(point_cloud_est, viz::Color::green());
            window_est.showWidget("point_cloud", cloud_widget);
             window_est.showWidget("coos", viz::WCoordinateSystem(0.05));
            // frame rate 1s
            window_est.spinOnce(1, true);
            window_est.removeAllWidgets();
        }
    }
}


void GmsMatch(Mat &img1, Mat &img2){
	vector<KeyPoint> kp1, kp2;
	Mat d1, d2;
	vector<DMatch> matches_all, matches_gms;

	Ptr<ORB> orb = ORB::create(10000);
	orb->setFastThreshold(0);
	orb->detectAndCompute(img1, Mat(), kp1, d1);
	orb->detectAndCompute(img2, Mat(), kp2, d2);

#ifdef USE_GPU
	GpuMat gd1(d1), gd2(d2);
	Ptr<cuda::DescriptorMatcher> matcher = cv::cuda::DescriptorMatcher::createBFMatcher(NORM_HAMMING);
	matcher->match(gd1, gd2, matches_all);
#else
	BFMatcher matcher(NORM_HAMMING);
	matcher.match(d1, d2, matches_all);
#endif

	// GMS filter
	int num_inliers = 0;
	std::vector<bool> vbInliers;
	gms_matcher gms(kp1,img1.size(), kp2,img2.size(), matches_all);
	num_inliers = gms.GetInlierMask(vbInliers, false, true);

	cout << "Get total " << num_inliers << " matches." << endl;

	// draw matches
	for (size_t i = 0; i < vbInliers.size(); ++i)
	{
		if (vbInliers[i] == true)
		{
			matches_gms.push_back(matches_all[i]);
		}
	}

	Mat show = DrawInlier(img1, img2, kp1, kp2, matches_gms, 1);
    imshow("show", show);
//    thread a(&imshow,"show", show);
    vector<Vec3d> points3d;

//    int fun = 10;
    triangulation(kp1,kp2,matches_gms,points3d);
//    thread th(&visualize3Dpoints,points3d);
//    thread r(&run,fun);
    visualize3Dpoints(points3d);
    waitKey();
}


