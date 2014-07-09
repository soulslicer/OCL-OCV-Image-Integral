#ifndef ADVCL_CLL_H_INCLUDED
#define ADVCL_CLL_H_INCLUDED

#if defined __APPLE__ || defined(MACOSX)
    #include <OpenCL/opencl.h>
#else
    #include <CL/opencl.h>
#endif

#define MULTI_LINE_STRING(a) #a
static const char *kernel_SumImgRows = MULTI_LINE_STRING(
    __kernel void kernel_SumImgRows(__constant   int       * imgDim,
                             __read_only  image2d_t   bmp,
                             __write_only image2d_t   temp)

    {

       const sampler_t smp = CLK_NORMALIZED_COORDS_FALSE | //Natural coordinates
                          CLK_ADDRESS_CLAMP | //Clamp to zeros
                          CLK_FILTER_NEAREST; //Don't interpolate

       uint4 pix;
       int4 pixInt = (int4)(0,0,0,0);

       int2 coords;
       coords.y = get_global_id(0);

       for (int i = 0; i < imgDim[0]; i++)
       {
           coords.x = i;

           pix = read_imageui(bmp, smp, coords);

           pixInt += (int4)((int)pix.x, (int)pix.y, (int)pix.z, 0);

           write_imagei(temp, coords, pixInt);
       }
    }
);

static const char *kernel_SumImgCols = MULTI_LINE_STRING(
    __kernel void kernel_SumImgCols(__constant   int       * imgDim,
                             __read_only  image2d_t   temp,
                             __write_only image2d_t   imgInt)

    {

       const sampler_t smp = CLK_NORMALIZED_COORDS_FALSE | //Natural coordinates
                          CLK_ADDRESS_CLAMP | //Clamp to zeros
                          CLK_FILTER_NEAREST; //Don't interpolate

       int4 pix;
       int4 pixInt = (int4)(0,0,0,0);

       int2 coords;
       coords.x = get_global_id(0);

       for (int i = 0; i < imgDim[1]; i++)
       {
           coords.y = i;

           pix = read_imagei(temp, smp, coords);

           pixInt += pix;

           write_imagei(imgInt, coords, pixInt);
       }
    }
);

class CL {
    public:

        //These are arrays we will use in this tutorial
        cl_mem imgDim;
        cl_mem imgA;
        cl_mem imgB;
        char* kernel_name;

        size_t workGroupSize[1]; //N dimensional array of workgroup size we must pass to the kernel

        //default constructor initializes OpenCL context and automatically chooses platform and device
        CL();
        //default destructor releases OpenCL objects and frees device memory
        ~CL();

        //load an OpenCL program from a file
        //the path is relative to the CL_SOURCE_DIR set in CMakeLists.txt
        void loadProgram(const char* kernel, char* name);

        //setup the data for the kernel
        //these are implemented in part1.cpp (in the future we will make these more general)
        void popCorn();
        //execute the kernel
        void runKernel();

    private:

        //handles for creating an opencl context
        cl_platform_id platform;

        //device variables
        cl_device_id* devices;
        cl_uint numDevices;
        unsigned int deviceUsed;

        cl_context context;

        cl_command_queue command_queue;
        cl_program program;
        cl_kernel kernel;


        //debugging variables
        cl_int err;
        cl_event event;

        //buildExecutable is called by loadProgram
        //build runtime executable from a program
        void buildExecutable();

};

#endif
