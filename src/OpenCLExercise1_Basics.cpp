//////////////////////////////////////////////////////////////////////////////
// Radix Sort   
// 
// (based on OpenCL exercise 1)
//////////////////////////////////////////////////////////////////////////////

// includes
#include <stdio.h>
 


#include <Core/Assert.hpp>
#include <Core/Time.hpp>
#include <OpenCL/cl-patched.hpp>
#include <OpenCL/Program.hpp>
#include <OpenCL/Event.hpp>
#include <OpenCL/Device.hpp>

#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <vector>
#include <list>




//////////////////////////////////////////////////////////////////////////////
// CPU implementation
//////////////////////////////////////////////////////////////////////////////

void RadixSort(int arr[], int size);

void calculateHost(std::vector<int>& h_input, std::vector<int>& h_output) {

	int maxValue = 0;
	for (int i = 0; i < h_input.size(); i++) {

		if (h_input[i] > maxValue)
			maxValue = h_input[i];
	}

	RadixSort(h_input.data(), h_input.size());

	h_output = h_input;

}


	// A utility function to get maximum value in arr[] 
	int getMax(int arr[], int size)
	{
		int max = arr[0];
		for (int i = 1; i < size; i++)
			if (arr[i] > max)
				max = arr[i];
		return max;
	}

	void CountingSort(int arr[], int size, int div)
	{
	//	int output[size];
		int* output = new int[size];


		int count[10] = { 0 };

		for (int i = 0; i < size; i++)
			count[(arr[i] / div) % 10]++;

		for (int i = 1; i < 10; i++)
			count[i] += count[i - 1];

		for (int i = size - 1; i >= 0; i--)
		{
			output[count[(arr[i] / div) % 10] - 1] = arr[i];
			count[(arr[i] / div) % 10]--;
		}

		for (int i = 0; i < size; i++)
			arr[i] = output[i];
	}


	void RadixSort(int arr[], int size)
	{
		int m = getMax(arr, size);
		for (int div = 1; m / div > 0; div *= 10)
			CountingSort(arr, size, div);
	}





//////////////////////////////////////////////////////////////////////////////
// Main function
//////////////////////////////////////////////////////////////////////////////



int main(int argc, char** argv) {
	// Create a context
	//cl::Context context(CL_DEVICE_TYPE_GPU);
	std::vector<cl::Platform> platforms;
	cl::Platform::get(&platforms);
	if (platforms.size() == 0) {
		std::cerr << "No platforms found" << std::endl;
		return 1;
	}
	int platformId = 0;
	for (size_t i = 0; i < platforms.size(); i++) {
		if (platforms[i].getInfo<CL_PLATFORM_NAME>() == "AMD Accelerated Parallel Processing") {
			platformId = i;
			break;
		}
	}
	cl_context_properties prop[4] = { CL_CONTEXT_PLATFORM, (cl_context_properties)platforms[platformId](), 0, 0 };
	std::cout << "Using platform '" << platforms[platformId].getInfo<CL_PLATFORM_NAME>() << "' from '" << platforms[platformId].getInfo<CL_PLATFORM_VENDOR>() << "'" << std::endl;
	cl::Context context(CL_DEVICE_TYPE_GPU, prop);

	// Get the first device of the context
	std::cout << "Context has " << context.getInfo<CL_CONTEXT_DEVICES>().size() << " devices" << std::endl;
	cl::Device device = context.getInfo<CL_CONTEXT_DEVICES>()[0];
	std::vector<cl::Device> devices;
	devices.push_back(device);
	OpenCL::printDeviceInfo(std::cout, device);

	// Create a command queue
	cl::CommandQueue queue(context, device, CL_QUEUE_PROFILING_ENABLE);

	// Load the source code
	cl::Program program = OpenCL::loadProgramSource(context, "K:/Uni-abSommerSemester2020/SS22/Lab-HPC-GPU/Opencl-Basics-ex1_RadixSort/Opencl-ex1/src/OpenCLExercise1_Basics.cl");
	// Compile the source code. This is similar to program.build(devices) but will print more detailed error messages
	OpenCL::buildProgram(program, devices);

	// Create a kernel object
	cl::Kernel kernel1(program, "kernel1");

	// Declare some values
	std::size_t wgSize = 128; // Number of work items per work group
	//std::size_t count = wgSize * 100000; // Overall number of work items = Number of elements

	const std::size_t count = 10000000; // lenght of Array   with data to sort

	std::size_t size = count * sizeof(int); // Size of data in bytes

	// Allocate space for input data and for output data from CPU and GPU on the host
	std::vector<int> h_input(count);
	std::vector<int> h_outputCpu(count);
	std::vector<int> h_outputGpu(count);

	std::vector<int> h_count(count);
	std::vector<int> h_pos(count);


	// Allocate space for input and output data on the device
	cl::Buffer d_input(context, CL_MEM_READ_WRITE, size);
	cl::Buffer d_output(context, CL_MEM_READ_WRITE, size);

	cl::Buffer d_count(context, CL_MEM_READ_WRITE, size);
	cl::Buffer d_pos(context, CL_MEM_READ_WRITE, size);

	// Initialize memory to 0xff (useful for debugging because otherwise GPU memory will contain information from last execution)
	memset(h_input.data(), 255, size);
	memset(h_outputCpu.data(), 255, size);
	memset(h_outputGpu.data(), 255, size);

	memset(h_count.data(), 255, 1);
	memset(h_pos.data(), 255, 1);

	queue.enqueueWriteBuffer(d_input, true, 0, size, h_input.data());
	queue.enqueueWriteBuffer(d_output, true, 0, size, h_outputGpu.data());


	 
	h_count[0] = 10000;//count / (int) log2(count);   // größer von Datenteilen      !

	//h_count[0] = 5;

	int numberOfKernelsToRun = 1000;//(int)log2(count);  //  e.g.  data array has length 15,  kernels to run = 3,  data chunk size (h_count) = 5   (3*5 =15)


	std::cout << "\n count (total array lengh): "<<count << "   size of data chunk: " << h_count[0] <<  "   numberOfKernelsToRun: " << numberOfKernelsToRun << std::endl;

	queue.enqueueWriteBuffer(d_count, true, 0, size, h_count.data());
	


	
	

	// Initialize input data with more or less random values
	for (std::size_t i = 0; i < count; i++)
		h_input[i] = (rand() * 123.456789);

	/*
	
	h_input[0] = 27;
	h_input[1] = 13;
	h_input[2] = 18;
	h_input[3] = 99;
	h_input[4] = 2;

	h_input[5] = 75;
	h_input[6] = 14;
	h_input[7] = 26;
	h_input[8] = 29;
	h_input[9] = 5;

	h_input[10] = 25;
	h_input[11] = 49;
	h_input[12] = 53;
	h_input[13] = 61;
	h_input[14] = 3;


 */




	// Do calculation on the host side
	Core::TimeSpan cpuStart = Core::getCurrentTime();
	calculateHost(h_input, h_outputCpu);
	Core::TimeSpan cpuEnd = Core::getCurrentTime();


	int maxValue = 0;
	int maxStellenInDaten = 0;
	for (int i = 0; i < count; i++) {

		if (h_input[i] > maxValue)
			maxValue = h_input[i];
	}
	maxStellenInDaten = floor(log10(abs(maxValue))) + 1;


	int* reordered;
	cl::Event execution;
	cl::Event copy1;
	cl::Event copy2;



	/////////////////////////////////////////////////////////////////////////////////start of loop /////////////////////////////////////////////////////////
	for (int stelle = 1; stelle <= maxStellenInDaten; stelle++) {

		// Copy input data to device
		queue.enqueueWriteBuffer(d_input, true, 0, size, h_input.data(), NULL, &copy1);

		// Launch kernel on the device
		kernel1.setArg<cl::Buffer>(0, d_input);
		kernel1.setArg<cl::Buffer>(1, d_output);

		kernel1.setArg<cl::Buffer>(2, d_count);

		//h_pos[0] = stelle;
		memset(h_pos.data(), stelle, 1);
		queue.enqueueWriteBuffer(d_pos, true, 0, size, h_pos.data());

		kernel1.setArg<cl::Buffer>(3, d_pos);




		//queue.enqueueNDRangeKernel(kernel1, 0, count, wgSize, NULL, &execution);  //orig line

		queue.enqueueNDRangeKernel(kernel1, 0, /*count*/ numberOfKernelsToRun /*numberOfKernelsToRun */, 1/*numberOfKernelsToRun*/ /* was 1, number of times krnl runs in parallel on the gpu*/, NULL, &execution);

		// Copy output data back to host
		queue.enqueueReadBuffer(d_output, true, 0, size, h_outputGpu.data(), NULL, &copy2);


		/// test start



		// Data redistribution

		//offsets

		// digit = (int)(number / pow(10, (pos - 1))) % 10;

		int pos = stelle;

		
		//int offsets[(int)count];

		int * offsets;
		offsets = (int*) malloc(count * sizeof * offsets);


		int counter[10] = { 0,0,0,0,0,0,0,0,0,0 };

		for (int i = 0; i < count; i++) {

			int digit = (int)(h_outputGpu[i] / pow(10, (pos - 1))) % 10;
			offsets[i] = counter[digit];
			counter[digit] = counter[digit] + 1;

		}


		for (int i = 0; i < count; i++) {

			//std::cout << "Offset[" << i << "]  = " << offsets[i] << "  orig data: " << h_outputGpu[i] << std::endl;

		}

		//hist
		int hist[10] = { 0,0,0,0,0,0,0,0,0,0 };
		int counterForHist[10] = { 0,0,0,0,0,0,0,0,0,0 };
		int digit = 0;

		//hist[0] = 0;

		for (int j = 0; j < count; j++) {
			digit = (int)(h_outputGpu[j] / pow(10, (pos - 1))) % 10;
			counterForHist[digit] = counterForHist[digit] + 1;
		}

		hist[0] = counterForHist[0];

		for (int i = 1; i < 10; i++) {
			hist[i] = hist[i - 1] + counterForHist[i];
		}

		int hist2[10] = { 0,0,0,0,0,0,0,0,0,0 };
		for (int i = 1; i < 10; i++) {
			hist2[i] = hist[i - 1];
		}


		for (int i = 0; i < 10; i++) {

			//std::cout << "hist[" << i << "]  = " << hist[i]  << std::endl;

		}
		for (int i = 0; i < 10; i++) {

			//std::cout << "hist2[" << i << "]  = " << hist2[i] << "   counterForHist[" << i << "] = " << counterForHist[i] << std::endl;

		}


		// new index

		//int newIndex[(int)count];
		int* newIndex;
		newIndex = (int*)malloc(count * sizeof * newIndex);

		for (int i = 0; i < count; i++) {


			int digit = (int)(h_outputGpu[i] / pow(10, (pos - 1))) % 10;
			newIndex[i] = offsets[i] + hist2[digit];
		}

		for (int i = 0; i < count; i++) {

			//std::cout << "newIndex[" << i << "]  = " << newIndex[i] << std::endl;

		}



		// shuffle data

		//int reordered[(int)count];
		
		reordered = (int*)malloc(count * sizeof * reordered);


		for (int i = 0; i < count; i++) {
			reordered[newIndex[i]] = h_outputGpu[i];
		}

		for (int i = 0; i < count; i++) {
			//std::cout << "reordered[" << i << "]  = " << reordered[i] << std::endl;
		}

	//	v.assign(array, array + 5); // 5 is size of array.
//		std::vector<int> v;
	//	v.assign(reordered, reordered + count);

		for (int i = 0; i < count; i++) {
			h_input[i] = reordered[i];
		}


		//h_input = v;

	} // end of big Stelle for loop
	/////////////////////////////////////////////////////////////////////////////////end of loop /////////////////////////////////////////////////////////

	for (int i = 0; i < h_outputGpu.size(); i++) {
		h_outputGpu[i] = reordered[i];
	}

	/**
	function msdRadixSort(data, l, k)
		buckets = list of 10 buckets
		if number of elements in data = 1 then
			return data
		end if
		
		for elem in data do
			d = l’th most significant digit of elem
			Place elem in buckets[d]
		end for
		
		if l ≤ k then
			for bucket in buckets do
				bucket = msdRadixSort(bucket, l + 1, k)
			end for
		end if
		
		//Replace data with the elements from buckets(in the same order)
		return data
	end function
	**/



	/// tests end



	// Print performance data
	Core::TimeSpan cpuTime = cpuEnd - cpuStart;
	Core::TimeSpan gpuTime = OpenCL::getElapsedTime(execution);
	Core::TimeSpan copyTime1 = OpenCL::getElapsedTime(copy1);
	Core::TimeSpan copyTime2 = OpenCL::getElapsedTime(copy2);
	Core::TimeSpan copyTime = copyTime1 + copyTime2;
	Core::TimeSpan overallGpuTime = gpuTime +copyTime;
	std::cout << "CPU Time: " << cpuTime.toString() << std::endl;
	std::cout << "Memory copy Time: " << copyTime.toString() << std::endl;
	std::cout << "GPU Time w/o memory copy: " << gpuTime.toString() << " (speedup = " << (cpuTime.getSeconds() / gpuTime.getSeconds()) << ")" << std::endl;
	std::cout << "GPU Time with memory copy: " << overallGpuTime.toString() << " (speedup = " << (cpuTime.getSeconds() / overallGpuTime.getSeconds()) << ")" << std::endl;

	
	for (int i = 0; i < count; i++) {
		//std::cout << "reordered[" << i << "]  = " << reordered[i] << std::endl;
	}
	

	//for (int i = 0; i < count; i++)
		//std::cout << "Result for " << i << " GPU value is " << h_outputGpu[i] << "\n";


	// Check whether results are correct
	std::size_t errorCount = 0;
	for (std::size_t i = 0; i < count; i++) {
		// Allow small differences between CPU and GPU results (due to different rounding behavior)
		if (!(std::abs (h_outputCpu[i] - h_outputGpu[i]) <= 10e-5)) {
			if (errorCount < 15)
				std::cout << "Result for " << i << " is incorrect: GPU value is " << h_outputGpu[i] << ", CPU value is " << h_outputCpu[i] << std::endl;
			else if (errorCount == 15)
				std::cout << "..." << std::endl;
			errorCount++;
		}
	}
	if (errorCount != 0) {
		std::cout << "Found " << errorCount << " incorrect results" << std::endl;
		return 1;
	}

	std::cout << "Success" << std::endl;

	return 0;
}
