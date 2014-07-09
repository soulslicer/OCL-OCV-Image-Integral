/*
 * Adventures in OpenCL tutorial series
 * Part 1
 *
 * author: Ian Johnson
 * code based on advisor Gordon Erlebacher's work
 * NVIDIA's examples
 * as well as various blogs and resources on the internet
 */
#include <stdio.h>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/ocl/ocl.hpp"
#include <iostream>

#include "cll.h"

// Namespaces
using namespace std;
using namespace cv;

// Globals
string inputName;




void initKernels(){
    CL kernel_SumImgRows_cl;
    kernel_SumImgRows_cl.loadProgram(kernel_SumImgRows,"kernel_SumImgRows");
    kernel_SumImgRows_cl.popCorn();
    // size_t program_length = (size_t)(strlen(kernel_SumImgCols));
    // cout << program_length << endl;
    // program = clCreateProgramWithSource(context, 1,
    //                   (const char **) &cSourceCL, &program_length, &err);
    // printf("clCreateProgramWithSource: %s\n", oclErrorString(err));
}

// Main
int main( int argc, const char** argv )
{

    //initKernels();

    //return 0;

    //1. Help Files
    const char* keys =
        "{ h | help       | false       | print help message }"
        "{ i | input      |             | specify input image }"
        ;

    //2. Command line parser
    CommandLineParser cmd(argc, argv, keys);
    if (cmd.get<bool>("help"))
    {
        cout << "Usage : facedetect [options]" << endl;
        cout << "Available options:" << endl;
        cmd.printParams();
        return EXIT_SUCCESS;
    }

    //3. Get commands
    inputName = cmd.get<string>("i");

    //4. Create a capture object
    CvCapture* capture = 0;
    Mat frame, frameCopy, image;

    //5. Load the file, whether it's a camera, image or video
    if( inputName.empty() )
    {
        capture = cvCaptureFromCAM(0);
        if(!capture)
            cout << "Capture from CAM 0 didn't work" << endl;
    }
    else
    {
        capture = cvCaptureFromAVI( inputName.c_str() );
        if(!capture){
            cout << "Capture from AVI didn't work" << endl;
            return EXIT_FAILURE;
        }
    }

    //6. Capture Video or Image and display
    if( capture )
    {
        cout << "In capture ..." << endl;
        int i=0;
        for(;;)
        {
            IplImage* iplImg = cvQueryFrame( capture );
            frame = iplImg;
            vector<Rect> faces;
            if( frame.empty() )
                break;

            cout << "F" << endl;
            imshow("frame",frame);

            if( waitKey( 10 ) >= 0 )
                break;
        }
        cvReleaseCapture( &capture );
    }

    cvDestroyWindow("result");
    std::cout<< "Finished" <<std::endl;
    return 0;
}


//Viola and Jones


// int main(int argc, char** argv)
// {
//     printf("Hello, OpenCL\n");
//     //initialize our CL object, this sets up the context
//     CL example;

//     //load and build our CL program from the file
//     example.loadProgram("part1.cl");

//     //initialize the kernel and send data from the CPU to the GPU
//     example.popCorn();
//     //execute the kernel
//     example.runKernel();
// }
