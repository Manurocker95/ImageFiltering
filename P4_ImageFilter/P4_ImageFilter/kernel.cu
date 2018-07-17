//****************************************************************************
// Also note that we've supplied a helpful debugging function called checkCudaErrors.
// You should wrap your allocation and copying statements like we've done in the
// code we're supplying you. Here is an example of the unsafe way to allocate
// memory on the GPU:
//
// cudaMalloc(&d_red, sizeof(unsigned char) * numRows * numCols);
//
// Here is an example of the safe way to do the same thing:
//
// checkCudaErrors(cudaMalloc(&d_red, sizeof(unsigned char) * numRows * numCols));
//****************************************************************************

#include <iostream>
#include <iomanip>
#include <cuda.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"  //needed for threadId.x

#include <cuda_runtime_api.h>
//#include "opencv\cv.hpp"

#define checkCudaErrors(val) check( (val), #val, __FILE__, __LINE__)
#define clamp(x,a,b) (__min(__max((x), a), b))
#define BLOCK_SIZE 32
#define FILTER_WIDTH 5

template<typename T>
void check(T err, const char* const func, const char* const file, const int line) {
	if (err != cudaSuccess) {
		std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
		std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
		exit(1);
	}
}

__global__
void box_filter(const unsigned char* const inputChannel,
	unsigned char* const outputChannel,
	int numRows, int numCols,
	const float* const filter, const int filterWidth)
{
	// TODO: 
	// NOTA: Cuidado al acceder a memoria que esta fuera de los limites de la imagen
	//
	// if ( absolute_image_position_x >= numCols ||
	//      absolute_image_position_y >= numRows )
	// {
	//     return;
	// }
	// NOTA: Que un thread tenga una posición correcta en 2D no quiere decir que al aplicar el filtro
	// los valores de sus vecinos sean correctos, ya que pueden salirse de la imagen.

	const int2 thread_2D_pos = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
		blockIdx.y * blockDim.y + threadIdx.y);

	const int thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;

	//make sure we don't try and access memory outside the image
	//by having any threads mapped there return early
	if (thread_2D_pos.x >= numCols || thread_2D_pos.y >= numRows) {
		return;
	}


	const int2 maskCenter = make_int2(floor(filterWidth / 2.0), floor(filterWidth / 2.0));

	const int2 i_start = make_int2((thread_2D_pos.x >= maskCenter.x) ? 0 : maskCenter.x - thread_2D_pos.x,
		(thread_2D_pos.y >= maskCenter.y) ? 0 : maskCenter.y - thread_2D_pos.y);

	const int2 i_stop = make_int2((thread_2D_pos.x < numCols - maskCenter.x) ? filterWidth : (numCols - 1 - thread_2D_pos.x + maskCenter.x),
		(thread_2D_pos.y < numRows - maskCenter.y) ? filterWidth : (numCols - 1 - thread_2D_pos.x + maskCenter.x));

	//Loop over the mask 
	for (int i = i_start.x; i < i_stop.x; i++) { //every Row
		for (int j = i_start.y; j < i_stop.y; i++) { //everycolumn

													 //Since no other thread is accessing this pixel, i dont need a atomicAdd.    MAYBE????                 /<-------------------- x-----------> <-----------------------y--------------------> 
			outputChannel[thread_1D_pos] += filter[i*filterWidth + j] * inputChannel[(thread_2D_pos.x - maskCenter.x + i) + (thread_2D_pos.y - maskCenter.y + j)*numCols]; //we have to reconvert from 2d to 1d array

		}
	}






}

// Optimize for pointer aliasing using __restrict__ allows CUDA commpiler to use the read-only data cache and improves performance

__global__
void box_filter_Shared(const unsigned char* const inputChannel,
	unsigned char* const outputChannel,
	int numRows, int numCols,
	const float* const filter, const int filterWidth)
{

	//Copy Image to shared memory
	//__shared__ unsigned char copyImage[blockDim.y][blockDim.x];
	__shared__ unsigned char copyImage[BLOCK_SIZE][BLOCK_SIZE];  //block dimension
	__shared__ float copyFilter[5][5];//[filterWidth][filterWidth];
	int bx = blockIdx.x;
	int by = blockIdx.y;
	//local to block
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	const int2 maskCenter = make_int2(floor(filterWidth / 2.0), floor(filterWidth / 2.0));


	//Global thread id
	//	const int2 thread_2D_pos = make_int2(bx * blockDim.x + tx,
	//		by * blockDim.y + ty);

	int2 thread_2D_pos = make_int2(tx + bx * (blockDim.x - 2 * maskCenter.x),
		ty + by * (blockDim.y - 2 * maskCenter.y));

		//thread_2D_pos = make_int2(tx + bx * blockDim.x -  (bx+1) * maskCenter.x,    
		//	ty + by * blockDim.y - (by+1) * maskCenter.y);



	//const int thread_1D_pos = thread_2D_pos.y * (numCols - 2 * maskCenter.x) + thread_2D_pos.x;
	int thread_1D_pos = thread_2D_pos.y * (numCols)+thread_2D_pos.x;




	//leave if not in total area
	if (thread_2D_pos.x >= numCols + 2 * maskCenter.x || thread_2D_pos.y >= numRows + 2 * maskCenter.y) {
		//if (thread_2D_pos.x > numCols || thread_2D_pos.y > numRows ) {
		return;
	}




	//unsigned char value;
	float value;
	if (thread_2D_pos.x >= maskCenter.x && thread_2D_pos.y >= maskCenter.y  && thread_2D_pos.x < numCols + maskCenter.x  && thread_2D_pos.y < numRows + maskCenter.y) {
		value = inputChannel[thread_1D_pos];
	}
	else {
		//printf("thread_2D: %i,%i\n", thread_2D_pos.x, thread_2D_pos.y);
		value = 255.0;
	}

	//put color into shared memory
	copyImage[ty][tx] = value;

	//make a copy of the filter in local memory
	if (tx < filterWidth && ty < filterWidth) {
		copyFilter[ty][tx] = filter[ty*filterWidth + tx];
	}



	// CONVOLUTION MASK

	float value2;

	//Leave if not in image,  the global image pading boarder, Protects from bluring
	if (thread_2D_pos.x < maskCenter.x || thread_2D_pos.y < maskCenter.y || thread_2D_pos.x > numCols - maskCenter.x || thread_2D_pos.y > numRows - maskCenter.y) {
		return;
	}

	//in 30x30 box
	if (tx >= maskCenter.x && ty >= maskCenter.y && tx < blockDim.x - maskCenter.x && ty < blockDim.y - maskCenter.y) {
		//if (tx >= 2 && ty >=2 && tx < 30 && ty < 30) {


		for (int r = 0; r < filterWidth; r++) {
			for (int c = 0; c < filterWidth; c++) {
				//value2 += copyImage[ty - maskCenter.y + r][tx - maskCenter.x + c] * filter[r*filterWidth + c];
				value2 += (copyImage[ty - maskCenter.y + r][tx - maskCenter.x + c]) * copyFilter[r][c];

				//value2 += (copyImage[ty - maskCenter.y + r - 2 * maskCenter.y*blockIdx.y][tx - maskCenter.x + c - 2 * maskCenter.x*blockIdx.x]) * copyFilter[r][c];
			}
		}

		//CLAMP to not have over saturated values:
		if (value2 < 0.0) {
			value2 = 0.0;
		}
		if (value2 > 255.0) {
			value2 = 255.0;
		}


		//set value to output image
		outputChannel[thread_1D_pos] = value2;

	}



	//DIRECTLY COPY FROM localMemory to Image
	value2 = copyImage[ty][tx];
	//outputChannel[thread_1D_pos] = value2;

	__syncthreads(); 






}

__global__
void box_filter_Shared_2(const unsigned char* const inputChannel,
	unsigned char* const outputChannel,
	int numRows, int numCols,
	const float* const filter, const int filterWidth)
{
	int bx = blockIdx.x;
	int by = blockIdx.y;
	//local to block
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	const int2 maskCenter = make_int2(floor(filterWidth / 2.0), floor(filterWidth / 2.0));
	const int2 thread_2D_pos = make_int2(tx + blockIdx.x*blockDim.x, blockIdx.y*blockDim.y + ty);
	int thread_1D_pos = thread_2D_pos.y*numCols + thread_2D_pos.x;

	/// if thread is out the image
	if (thread_2D_pos.x >= numCols || thread_2D_pos.y >= numRows)
		return;

	float value = 0.0f;


	//put color into shared memory
	__shared__ unsigned char copyImage[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ float copyFilter[5][5];

	copyImage[ty][tx] = inputChannel[thread_1D_pos];

	//make a copy of the filter in local memory
	if (tx < filterWidth && ty < filterWidth) {
		copyFilter[ty][tx] = filter[ty*filterWidth + tx];
	}
	__syncthreads(); // SyncThreads to have all the share memory complete

					 // CONVOLUTION MASK
	if (tx >= 2 && ty >= 2 && tx < blockDim.x - 2 && ty < blockDim.y - 2) {
		for (int r = 0; r < filterWidth; r++) {
			for (int c = 0; c < filterWidth; c++) {
				//value2 += (copyImage[ty - maskCenter.y + r][tx - maskCenter.x + c]) * copyFilter[r][c];
				value += (copyImage[ty - maskCenter.y + r][tx - maskCenter.x + c]) * copyFilter[r][c];// filter[r*filterWidth + c];

			}
		}
		value = clamp(value, 0, 255);
	}
	else
	{
		for (int r = 0; r < filterWidth; r++) {
			for (int c = 0; c < filterWidth; c++) {
				//value2 += (copyImage[ty - maskCenter.y + r][tx - maskCenter.x + c]) * copyFilter[r][c];
				value += (copyImage[ty][tx]) * copyFilter[r][c];// filter[r*filterWidth + c];

			}
		}
		value = clamp(value, 0, 255);
	}



	outputChannel[thread_1D_pos] = value;
	//outputChannel[thread_1D_pos] = copyImage[ty][tx];
}


__global__
void box_filter_Shared_3(const unsigned char* const inputChannel,
	unsigned char* const outputChannel,
	int numRows, int numCols,
	const float* const filter, const int filterWidth)
{
	int bx = blockIdx.x;
	int by = blockIdx.y;
	//local to block
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	const int2 maskCenter = make_int2(floor(filterWidth / 2.0), floor(filterWidth / 2.0));
	const int2 thread_2D_pos = make_int2(tx + blockIdx.x*blockDim.x, blockIdx.y*blockDim.y + ty);
	int thread_1D_pos = thread_2D_pos.y*numCols + thread_2D_pos.x;

	/// if thread is out the image
	if (thread_2D_pos.x >= numCols || thread_2D_pos.y >= numRows)
		return;

	float value = 0.0f;


	//put color into shared memory
	__shared__ unsigned char copyImage[36][36];
	__shared__ float copyFilter[FILTER_WIDTH][FILTER_WIDTH];

	copyImage[ty][tx] = inputChannel[thread_1D_pos];

	//make a copy of the filter in local memory
	if (tx < filterWidth && ty < filterWidth) {
		copyFilter[ty][tx] = filter[ty*filterWidth + tx];
	}
	__syncthreads(); // SyncThreads to have all the share memory complete

	if ((thread_2D_pos.x < maskCenter.x || thread_2D_pos.y < maskCenter.y) || (thread_2D_pos.x >= numCols-maskCenter.x || thread_2D_pos.y >= numRows - maskCenter.y))
		return;

	// CONVOLUTION MASK
	for (int r = 0; r < filterWidth; r++) 
	{
		for (int c = 0; c < filterWidth; c++)
		{
			value += (copyImage[ty - maskCenter.y + r][tx - maskCenter.x + c]) * copyFilter[r][c];
		}
	}
	value = clamp(value, 0, 255);


	outputChannel[thread_1D_pos] = value;
	//outputChannel[thread_1D_pos] = copyImage[ty][tx];
}

__global__
void box_filter_GlobalMemory(const unsigned char* const inputChannel,
	unsigned char* const outputChannel,
	int numRows, int numCols,
	const float* const filter, const int filterWidth)
{
	
	const int2 maskCenter = make_int2(floor(filterWidth / 2.0), floor(filterWidth / 2.0));

	int r = blockIdx.y * blockDim.y + threadIdx.y;
	int c = blockIdx.x * blockDim.x + threadIdx.x;

	// COMPROBAR SIEMPRE QUE NO NOS ESTAMOS SALIENDO
		if ((r >= numRows) || (c >= numCols))
			return;

	float result = 0.f;
	for (int filter_r = 0; filter_r < filterWidth; filter_r++) {
		for (int filter_c = 0; filter_c < filterWidth; filter_c++) {
			int image_r = r - maskCenter.y + filter_r;
			int image_c = c -maskCenter.x + filter_c;
			// VER SI ESTAMOS DENTRO DE LA IMAGEN
			if ((image_c >= 0) && (image_c < numCols) && (image_r >= 0) && (image_r < numRows)) {
				float image_value = inputChannel[image_r * numCols + image_c];
				float filter_value = filter[filter_r*filterWidth + filter_c]; // OBTENER VALOR DE FILTRO
					result += image_value * filter_value;


			}
		}
	}

	result = clamp(result, 0, 255);
	outputChannel[r * numCols + c] = result;
}



//This kernel takes in an image represented as a uchar4 and splits
//it into three images consisting of only one color channel each
__global__
void separateChannels(const uchar4* const inputImageRGBA,
	int numRows,
	int numCols,
	unsigned char* const redChannel,
	unsigned char* const greenChannel,
	unsigned char* const blueChannel)
{
	// TODO: 
	// NOTA: Cuidado al acceder a memoria que esta fuera de los limites de la imagen
	//
	// if ( absolute_image_position_x >= numCols ||
	//      absolute_image_position_y >= numRows )
	// {
	//     return;
	// }


	const int2 thread_2D_pos = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
		blockIdx.y * blockDim.y + threadIdx.y);

	const int thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;

	//make sure we don't try and access memory outside the image
	//by having any threads mapped there return early
	if (thread_2D_pos.x >= numCols || thread_2D_pos.y >= numRows) {
		return;
	}


	//Get each Channel
	//redChannel[thread_1D_pos]=  inputImageRGBA[thread_1D_pos +0];  //position+0 is red
	//greenChannel[thread_1D_pos]= inputImageRGBA[thread_1D_pos +1]; //position+1 is green
	//blueChannel[thread_1D_pos]= inputImageRGBA[thread_1D_pos +2];

	uchar4 inPixel = inputImageRGBA[thread_1D_pos];
	redChannel[thread_1D_pos] = inPixel.x;  //position+0 is red
	greenChannel[thread_1D_pos] = inPixel.y; //position+1 is green
	blueChannel[thread_1D_pos] = inPixel.z;




}

//This kernel takes in three color channels and recombines them
//into one image. The alpha channel is set to 255 to represent
//that this image has no transparency.
__global__
void recombineChannels(const unsigned char* const redChannel,
	const unsigned char* const greenChannel,
	const unsigned char* const blueChannel,
	uchar4* const outputImageRGBA,
	int numRows,
	int numCols)
{
	const int2 thread_2D_pos = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
		blockIdx.y * blockDim.y + threadIdx.y);

	const int thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;

	//make sure we don't try and access memory outside the image
	//by having any threads mapped there return early
	if (thread_2D_pos.x >= numCols || thread_2D_pos.y >= numRows)
		return;

	unsigned char red = redChannel[thread_1D_pos];
	unsigned char green = greenChannel[thread_1D_pos];
	unsigned char blue = blueChannel[thread_1D_pos];

	//Alpha should be 255 for no transparency
	uchar4 outputPixel = make_uchar4(red, green, blue, 255);

	outputImageRGBA[thread_1D_pos] = outputPixel;
}

unsigned char *d_red, *d_green, *d_blue;
float         *d_filter;

void allocateMemoryAndCopyToGPU(const size_t numRowsImage, const size_t numColsImage,
	const float* const h_filter, const size_t filterWidth)
{

	//allocate memory for the three different channels
	checkCudaErrors(cudaMalloc(&d_red, sizeof(unsigned char) * numRowsImage * numColsImage));
	checkCudaErrors(cudaMalloc(&d_green, sizeof(unsigned char) * numRowsImage * numColsImage));
	checkCudaErrors(cudaMalloc(&d_blue, sizeof(unsigned char) * numRowsImage * numColsImage));

	//TODO:
	//Reservar memoria para el filtro en GPU: d_filter, la cual ya esta declarada
	int size = sizeof(float) * numRowsImage * numColsImage;
	checkCudaErrors(cudaMalloc(&d_filter, size));

	//TODO
	// Copiar el filtro  (h_filter) a memoria global de la GPU (d_filter)
	checkCudaErrors(cudaMemcpy(d_filter, h_filter, sizeof(float) *filterWidth*filterWidth, cudaMemcpyHostToDevice));


}


void create_filter(float **h_filter, int *filterWidth) {

	const int KernelWidth = 5; //OJO CON EL TAMAÑO DEL FILTRO//
	*filterWidth = KernelWidth;

	//create and fill the filter we will convolve with
	*h_filter = new float[KernelWidth * KernelWidth];

	/*
	//Filtro gaussiano: blur
	const float KernelSigma = 2.;

	float filterSum = 0.f; //for normalization

	for (int r = -KernelWidth/2; r <= KernelWidth/2; ++r) {
		for (int c = -KernelWidth/2; c <= KernelWidth/2; ++c) {
			float filterValue = expf( -(float)(c * c + r * r) / (2.f * KernelSigma * KernelSigma));
			(*h_filter)[(r + KernelWidth/2) * KernelWidth + c + KernelWidth/2] = filterValue;
			filterSum += filterValue;
		}
	}

	float normalizationFactor = 1.f / filterSum;

	for (int r = -KernelWidth/2; r <= KernelWidth/2; ++r) {
		for (int c = -KernelWidth/2; c <= KernelWidth/2; ++c) {
			(*h_filter)[(r + KernelWidth/2) * KernelWidth + c + KernelWidth/2] *= normalizationFactor;
		}
	}
			  
	*/
						   //Laplaciano 5x5
	
	(*h_filter)[0] = 0;   (*h_filter)[1] = 0;    (*h_filter)[2] = -1.;  (*h_filter)[3] = 0;    (*h_filter)[4] = 0;
	(*h_filter)[5] = 1.;  (*h_filter)[6] = -1.;  (*h_filter)[7] = -2.;  (*h_filter)[8] = -1.;  (*h_filter)[9] = 0;
	(*h_filter)[10] = -1.; (*h_filter)[11] = -2.; (*h_filter)[12] = 17.; (*h_filter)[13] = -2.; (*h_filter)[14] = -1.;
	(*h_filter)[15] = 1.; (*h_filter)[16] = -1.; (*h_filter)[17] = -2.; (*h_filter)[18] = -1.; (*h_filter)[19] = 0;
	(*h_filter)[20] = 1.;  (*h_filter)[21] = 0;   (*h_filter)[22] = -1.; (*h_filter)[23] = 0;   (*h_filter)[24] = 0;
	
	//TODO: crear los filtros segun necesidad
	//NOTA: cuidado al establecer el tamaño del filtro a utilizar

}


void convolution(const uchar4 * const h_inputImageRGBA, uchar4 * const d_inputImageRGBA,
	uchar4* const d_outputImageRGBA, const size_t numRows, const size_t numCols,
	unsigned char *d_redFiltered,
	unsigned char *d_greenFiltered,
	unsigned char *d_blueFiltered,
	const int filterWidth, float * h_filter)
{
	//TODO: Calcular tamaños de bloque
	const dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE, 1); //32 by 32 threads (or pixels) in each block
									 //const dim3 gridSize(ceil(numCols/blockSize.x), ceil(numRows / blockSize.y), 1);  //how many blocks fit in the image width and height
	const dim3 gridSize(1 + ((numCols) / blockSize.x), 1 + ((numRows) / blockSize.y), 1);  //1+ is faster than ceil

	const int2 maskCenter = make_int2(floor(filterWidth / 2.0), floor(filterWidth / 2.0));
	const dim3 gridSizeShared((numCols) / (blockSize.x - maskCenter.x * 2) + 1, (numRows) / (blockSize.y - maskCenter.y * 2) + 1, 1);  //1+ is faster than ceil


																																	   //TODO: Lanzar kernel para separar imagenes RGBA en diferentes colores
	separateChannels << <gridSize, blockSize >> > (d_inputImageRGBA,
		numRows,
		numCols,
		d_red,
		d_green,
		d_blue);
	checkCudaErrors(cudaGetLastError());



	//TODO: Ejecutar convolución. Una por canal
	//Launch kernal for red

	//box_filter_Shared
	box_filter_Shared_3 << <gridSize, blockSize >> > (d_red,
		d_redFiltered,
		numRows, numCols,
		d_filter, filterWidth);

	checkCudaErrors(cudaGetLastError());


	//FOR BLUE

	box_filter_Shared_3 << <gridSize, blockSize >> > (d_blue,
		d_blueFiltered,
		numRows, numCols,
		d_filter, filterWidth);


	//FOR GREEN
	box_filter_Shared_3 << <gridSize, blockSize >> > (d_green,
		d_greenFiltered,
		numRows, numCols,
		d_filter, filterWidth);



	// Recombining the results. 
	recombineChannels << <gridSize, blockSize >> >(d_redFiltered,
		d_greenFiltered,
		d_blueFiltered,
		d_outputImageRGBA,
		numRows,
		numCols);


	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());



}


//Free all the memory that we allocated
//TODO: make sure you free any arrays that you allocated
void cleanup() {
	checkCudaErrors(cudaFree(d_red));
	checkCudaErrors(cudaFree(d_green));
	checkCudaErrors(cudaFree(d_blue));

}
