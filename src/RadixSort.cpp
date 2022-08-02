//////////////////////////////////////////////////////////////////////////////
// Radix Sort   
// 
// (based on OpenCL exercise 1)
//////////////////////////////////////////////////////////////////////////////

// includes
#include <Core/Assert.hpp>
#include <Core/Time.hpp>
#include <OpenCL/cl-patched.hpp>
#include <OpenCL/Program.hpp>
#include <OpenCL/Event.hpp>
#include <OpenCL/Device.hpp>

#include <stdio.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <vector>
#include <list>
#include <iomanip>

//#define IN_PLACE_SORT

const unsigned int KEY_SIZE = 32;
const unsigned int DIGIT_SIZE = 8;
const unsigned int DIGIT_PLACE = KEY_SIZE / DIGIT_SIZE;
const unsigned int NUM_DIGIT_BINNINGS = 1 << DIGIT_SIZE;

const unsigned int count = 2048*4086*4; // lenght of Array with data to sort, 1024*1024*4
const unsigned int num_data_per_work_item = 4; // use a unsigned int vector of length 4

/**
Find the path of a file
	@param filename: the name of the file to find
	@param executablePath optional absolute path of the executable
*/
char* findFilePath(const char* filename, const char* executablePath);


//////////////////////////////////////////////////////////////////////////////
// CPU implementation
//////////////////////////////////////////////////////////////////////////////

/**
 * @brief implement in-place radix sort on GPU
 * 
 * @param data: data to be sorted
 * @param sizeOfData: size of data vector 
 */
void radixSort(unsigned int *keys, int sizeOfKeys) {
	/**
	for (int i = 0; i < sizeOfKeys; i++) {
		if (keys[i] == 4293699190)
			std::cout << "[CPU] The last exists" << std::endl;
		if (keys[i] == 1680956)
			std::cout << "[CPU] The first exists" << std::endl;
	}*/

	const unsigned int mask = (1 << DIGIT_SIZE) - 1;
	// Init the histrogram
	int *hist = (int*) malloc(NUM_DIGIT_BINNINGS * sizeof(int));
	unsigned int *copy = (unsigned int*) malloc(sizeOfKeys * sizeof(unsigned int));

	// Iterate through digit places
	for (int i = 0; i < DIGIT_PLACE; i++) {
		// Init the histogram to all 0
		for (int j = 0; j < NUM_DIGIT_BINNINGS; j++) hist[j] = 0;
		// Compute the histrogram
		for (int g = 0; g < sizeOfKeys; g++) {
			int histroIdx = (keys[g] >> (i * DIGIT_SIZE)) & mask;
			copy[g] = keys[g];
			hist[histroIdx]++;
		}
		// Scan through the histrogram
		int sum = 0;
		for (int s = 0; s < NUM_DIGIT_BINNINGS; s++) {
			int tmp = hist[s];
			hist[s] = sum;
			sum += tmp;
		}
		// Scatter the vector
		for (int g = 0; g < sizeOfKeys; g++) {
			unsigned int tmp = (copy[g] >> (i * DIGIT_SIZE)) & mask;
			int newIdx = hist[tmp];
			keys[newIdx] = copy[g];
			hist[tmp]++;
		}
	}
	free(copy);
	free(hist);
}

/**
 * @brief Init keys with random numbers
 * 
 * @param output       pointer to the beginning of key array
 * @param numElements  num of keys
 * @param keyBitSize   size of key (in bits)
 */
void initKeys(unsigned int *keys, unsigned int sizeOfKeys, unsigned int keyBitSize) {
	int keyShiftMask = 0;
	if (keyBitSize > 16) keyShiftMask = (1 << (keyBitSize-16)) - 1;

	int keyMask = 0xffff;
	if (keyBitSize < 16) keyMask = (1 << keyBitSize) - 1;

	srand(5432);
	for (int i = 0; i < sizeOfKeys; ++i) 
		keys[i] = ((rand() & keyShiftMask)<<16 | (rand() & keyMask));
}


//////////////////////////////////////////////////////////////////////////////
// Main function
//////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv) {

	std::size_t size = count * sizeof(unsigned int); // Size of data in bytes
	
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
	// TODO: change the path of kernel file
	char* kernelFilePath = findFilePath("radixSort.cl", argv[0]);
	if (kernelFilePath == nullptr) {
		std::cerr << "Can't find kernel..." << std::endl;
		exit(1);
	}
	//cl::Program program = OpenCL::loadProgramSource(context, "/scratch/radixSort/src/radixSort.cl");
	cl::Program program = OpenCL::loadProgramSource(context, kernelFilePath);
	// Compile the source code. This is similar to program.build(devices) but will print more detailed error messages
	OpenCL::buildProgram(program, devices);

	// Create a kernel object
	cl::Kernel ckUpfrontHistogram(program, "upfrontHistogram");
	cl::Kernel ckExclusiveSum(program, "exclusiveSum");
	cl::Kernel ckChainedScanDigitBinning(program, "chainedScanDigitBinning");

	// Declare some values

	const unsigned int size_work_group = 256; // Number of work items per work group, should be greater than 2^DIGIT_SIZE
	if (size_work_group < NUM_DIGIT_BINNINGS) {
		std::cerr << "The work group size should be greater than 2^DIGIT_SIZE, i.e. " 
				  << NUM_DIGIT_BINNINGS << std::endl;
		exit(1);
	}
	const unsigned int num_work_items = count / num_data_per_work_item;
	const unsigned int num_workgroups = count / size_work_group / num_data_per_work_item;

	// Size of global histogram = 
	// 		#digit binnings: 2^DIGIT_SIZE *
	//		#digit places: KEY_SIZE / DIGIT_SIZE *
	// 		size of single histogram: 2^DIGIT_SIZE
	const unsigned int histogram_size = NUM_DIGIT_BINNINGS * DIGIT_PLACE * sizeof(unsigned int); 

	// ------------------------------------
	// Data and memory preparation
	// ------------------------------------
	std::vector<unsigned int> h_input(count);
#ifndef IN_PLACE_SORT
	std::vector<unsigned int> h_outputCpu(count);
#endif
	std::vector<unsigned int> h_outputGpu(count);

	// Allocate space for input and output data on the device
	cl::Buffer d_input(context, CL_MEM_READ_WRITE, size);
#ifndef IN_PLACE_SORT
	cl::Buffer d_output(context, CL_MEM_READ_WRITE, size);
#endif
	cl::Buffer global_histogram(context, CL_MEM_READ_WRITE, histogram_size);
	cl::Buffer flag_IPS(context, CL_MEM_READ_WRITE, NUM_DIGIT_BINNINGS * num_workgroups * sizeof(unsigned int));
	cl::Buffer partition_counter(context, CL_MEM_READ_WRITE, sizeof(unsigned int));
	cl::Buffer mutex(context, CL_MEM_READ_WRITE, sizeof(unsigned int));

	// Initialize memory to 0xff (useful for debugging because otherwise GPU memory will contain information from last execution)
	memset(h_input.data(), 255, size);
	memset(h_outputGpu.data(), 255, size);
	queue.enqueueWriteBuffer(d_input, true, 0, size, h_input.data());
#ifndef IN_PLACE_SORT
	queue.enqueueWriteBuffer(d_output, true, 0, size, h_outputGpu.data());
#endif

	// Initialize input data with more or less random values
	initKeys(h_input.data(), count, KEY_SIZE);

	// ------------------------------------
	// Host execution
	// ------------------------------------
	Core::TimeSpan cpuStart = Core::getCurrentTime();
	radixSort(h_input.data(), count);
	Core::TimeSpan cpuEnd = Core::getCurrentTime();
	std::cout << "-------------------------" << std::endl;
	std::cout << "------- Host Part -------" << std::endl;
	std::cout << "-------------------------" << std::endl;
	std::cout << "Finish CPU execution" << std::endl;
	std::cout << std::endl;

	// ------------------------------------
	// Device execution
	// ------------------------------------
	// Transfer data from host -> device
	std::cout << "-------------------------" << std::endl;
	std::cout << "----- Device Part -------" << std::endl;
	std::cout << "-------------------------" << std::endl;
	cl::Event writeTo;
	queue.enqueueWriteBuffer(d_input, true, 0, size, h_input.data(), NULL, &writeTo);
	std::cout << "- Data is copied to device" << std::endl;

	cl::Event calGlobalHistogram;	
	// Kernel 1: Get global histogram
	ckUpfrontHistogram.setArg<cl::Buffer>(0, d_input);
	ckUpfrontHistogram.setArg<cl::Buffer>(1, global_histogram);
	ckUpfrontHistogram.setArg(2, cl::Local(histogram_size));
	queue.enqueueNDRangeKernel(ckUpfrontHistogram, cl::NullRange, num_work_items, size_work_group, NULL, &calGlobalHistogram);
	std::cout << "- Finish 1st kernel" << std::endl;
	
	// Kernel 2: Compute exclusive prefix sum of global histogram
	cl::Event calGlobalHistogramPrefix;
	ckExclusiveSum.setArg<cl::Buffer>(0, global_histogram);
	queue.enqueueNDRangeKernel(ckExclusiveSum, cl::NullRange, DIGIT_PLACE, DIGIT_PLACE, NULL, &calGlobalHistogramPrefix);
	std::cout << "- Finish 2nd kernel" << std::endl;

	//const unsigned int zero = 0;
	// Kernel 3: chained scan digit binnings for #digit places times/iterations
	cl::Event chainedEvents[DIGIT_PLACE];
	for (int i = 0; i < DIGIT_PLACE; i++)
	{
		queue.enqueueFillBuffer(flag_IPS, 0, 0, NUM_DIGIT_BINNINGS*num_workgroups*sizeof(unsigned int), NULL, NULL);
		queue.enqueueFillBuffer(partition_counter, 0, 0, sizeof(unsigned int), NULL, NULL);
		queue.enqueueFillBuffer(mutex, 0, 0, sizeof(unsigned int), NULL, NULL);

		ckChainedScanDigitBinning.setArg(0, d_input);
		ckChainedScanDigitBinning.setArg(1, global_histogram);
		ckChainedScanDigitBinning.setArg(2, flag_IPS);
		ckChainedScanDigitBinning.setArg(3, cl::Local(NUM_DIGIT_BINNINGS*sizeof(unsigned int))); // EPS
		ckChainedScanDigitBinning.setArg(4, cl::Local(NUM_DIGIT_BINNINGS*sizeof(unsigned int))); // Aggregate
		ckChainedScanDigitBinning.setArg(5, partition_counter);
		ckChainedScanDigitBinning.setArg(6, i);
		ckChainedScanDigitBinning.setArg(7, d_output);
		queue.enqueueNDRangeKernel(ckChainedScanDigitBinning, cl::NullRange, num_work_items, size_work_group, NULL, &chainedEvents[i]);
	}
	std::cout << "- Finish 3rd kernel" << std::endl;

	// transfer reordered data back to host
	cl::Event writeBack;
#ifndef IN_PLACE_SORT
	queue.enqueueReadBuffer(d_input, CL_TRUE, 0, size, h_outputGpu.data(), NULL, &writeBack);
#else
	queue.enqueueReadBuffer(d_output, CL_TRUE, 0, size, h_outputGpu.data(), NULL, &writeBack);
#endif
	std::cout << "- Copy result back to host" << std::endl;

	// ------------------------------------
	// Validation
	// ------------------------------------
	bool correct = true;	
#ifdef DEBUG_RESULT
	const int newLine = 5;
	std::cout << "GPU reordered:" << std::endl;  
#endif
	for (int i = 1; i < count; i++) {
#ifdef DEBUG_RESULT
		std::cout << std::setw(10) << h_outputGpu[i-1] << " ";
		if ((i % newLine) == 0) std::cout << std::endl;
#endif
		if (h_outputGpu[i-1] > h_outputGpu[i]) {
			correct = false;
			break;
		}
	}
	std::cout << std::endl;
	if (!correct) {
		std::cerr << "[ERROR] GPU sorting wrong..." << std::endl;
		exit(1);
	}

#ifdef DEBUG_RESULT
	std::cout << "CPU reordered:" << std::endl;
#endif
	for (int i = 1; i < count; i++) {
#ifdef DEBUG_RESULT
		std::cout << std::setw(10) << h_input[i-1] << " ";
		if ((i % newLine) == 0) std::cout << std::endl;
#endif
		if (h_input[i-1] > h_input[i]) {
			correct = false;
			break;
		}
	}
	if (!correct) {
		std::cerr << "[ERROR] CPU sorting wrong..." << std::endl;
		exit(1);
	}

	// ------------------------------------
	// Performance evaluation
	// ------------------------------------
	Core::TimeSpan cpuTime = cpuEnd - cpuStart;
	Core::TimeSpan gpuTime = OpenCL::getElapsedTime(calGlobalHistogram) + 
							 OpenCL::getElapsedTime(calGlobalHistogramPrefix);
	for (int i = 0; i < DIGIT_PLACE; i++)
		gpuTime = gpuTime + OpenCL::getElapsedTime(chainedEvents[i]);

	Core::TimeSpan copyTime1 = OpenCL::getElapsedTime(writeTo);
	Core::TimeSpan copyTime2 = OpenCL::getElapsedTime(writeBack);
	Core::TimeSpan copyTime = copyTime1 + copyTime2;
	Core::TimeSpan overallGpuTime = gpuTime +copyTime;
	
	std::cout << "****************************************" << std::endl;
	std::cout << "            Evaluation Part             " << std::endl;
	std::cout << "****************************************" << std::endl;
	std::cout << std::left << std::setw(27) << "Number of key: " << count << std::endl;
	std::cout << std::left << std::setw(27) << "Size of each keys:" << KEY_SIZE << " bits" << std::endl;
	std::cout << std::left << std::setw(27) << "Size of each digit:" << DIGIT_SIZE << " bits" << std::endl;
	std::cout << std::endl;
	std::cout << std::left << std::setw(27) << "CPU Time: " << cpuTime.toString() << std::endl;
	std::cout << std::left << std::setw(27) << "Memory copy Time: " << copyTime.toString() << std::endl;
	std::cout << std::left << std::setw(27) << "GPU Time w/o memory copy: " << gpuTime.toString() << " (speedup = " << (cpuTime.getSeconds() / gpuTime.getSeconds()) << ")" << std::endl;
	std::cout << std::left << std::setw(27) << "GPU Time with memory copy: " << overallGpuTime.toString() << " (speedup = " << (cpuTime.getSeconds() / overallGpuTime.getSeconds()) << ")" << std::endl;

	std::cout << "Success!" << std::endl;

	return 0;
}

char* findFilePath(const char* filename, const char* executable_path) 
{
	// <executable_name> defines a variable that is replaced with the name of the executable

	// Typical relative search paths to locate needed companion files (e.g. sample input data, or JIT source files)
	// The origin for the relative search may be the .exe file, a .bat file launching an .exe, a browser .exe launching the .exe or .bat, etc
	const char* searchPath[] = 
	{
		"./",                                       // same dir 
		"./data/",                                  // "/data/" subdir 
		"./src/",                                   // "/src/" subdir
		"./src/<executable_name>/data/",            // "/src/<executable_name>/data/" subdir 
		"./inc/",                                   // "/inc/" subdir
		"../",                                      // up 1 in tree 
		"../data/",                                 // up 1 in tree, "/data/" subdir 
		"../src/",                                  // up 1 in tree, "/src/" subdir 
		"../inc/",                                  // up 1 in tree, "/inc/" subdir 
		"../OpenCL/src/<executable_name>/",         // up 1 in tree, "/OpenCL/src/<executable_name>/" subdir 
		"../OpenCL/src/<executable_name>/data/",    // up 1 in tree, "/OpenCL/src/<executable_name>/data/" subdir 
		"../OpenCL/src/<executable_name>/src/",     // up 1 in tree, "/OpenCL/src/<executable_name>/src/" subdir 
		"../OpenCL/src/<executable_name>/inc/",     // up 1 in tree, "/OpenCL/src/<executable_name>/inc/" subdir 
		"../C/src/<executable_name>/",              // up 1 in tree, "/C/src/<executable_name>/" subdir 
		"../C/src/<executable_name>/data/",         // up 1 in tree, "/C/src/<executable_name>/data/" subdir 
		"../C/src/<executable_name>/src/",          // up 1 in tree, "/C/src/<executable_name>/src/" subdir 
		"../C/src/<executable_name>/inc/",          // up 1 in tree, "/C/src/<executable_name>/inc/" subdir 
		"../DirectCompute/src/<executable_name>/",      // up 1 in tree, "/DirectCompute/src/<executable_name>/" subdir 
		"../DirectCompute/src/<executable_name>/data/", // up 1 in tree, "/DirectCompute/src/<executable_name>/data/" subdir 
		"../DirectCompute/src/<executable_name>/src/",  // up 1 in tree, "/DirectCompute/src/<executable_name>/src/" subdir 
		"../DirectCompute/src/<executable_name>/inc/",  // up 1 in tree, "/DirectCompute/src/<executable_name>/inc/" subdir 
		"../../",                                   // up 2 in tree 
		"../../data/",                              // up 2 in tree, "/data/" subdir 
		"../../src/",                               // up 2 in tree, "/src/" subdir 
		"../../inc/",                               // up 2 in tree, "/inc/" subdir 
		"../../../",                                // up 3 in tree 
		"../../../src/<executable_name>/",          // up 3 in tree, "/src/<executable_name>/" subdir 
		"../../../src/<executable_name>/data/",     // up 3 in tree, "/src/<executable_name>/data/" subdir 
		"../../../src/<executable_name>/src/",      // up 3 in tree, "/src/<executable_name>/src/" subdir 
		"../../../src/<executable_name>/inc/",      // up 3 in tree, "/src/<executable_name>/inc/" subdir 
		"../../../sandbox/<executable_name>/",      // up 3 in tree, "/sandbox/<executable_name>/" subdir
		"../../../sandbox/<executable_name>/data/", // up 3 in tree, "/sandbox/<executable_name>/data/" subdir
		"../../../sandbox/<executable_name>/src/",  // up 3 in tree, "/sandbox/<executable_name>/src/" subdir
		"../../../sandbox/<executable_name>/inc/"   // up 3 in tree, "/sandbox/<executable_name>/inc/" subdir
	};
    
	// Extract the executable name
	std::string executable_name;
	if (executable_path != 0) 
	{
		executable_name = std::string(executable_path);

	#ifdef _WIN32        
		// Windows path delimiter
		size_t delimiter_pos = executable_name.find_last_of('\\');        
		executable_name.erase(0, delimiter_pos + 1);

		if (executable_name.rfind(".exe") != string::npos)
		{
			// we strip .exe, only if the .exe is found
			executable_name.resize(executable_name.size() - 4);        
		}
	#else
		// Linux & OSX path delimiter
		size_t delimiter_pos = executable_name.find_last_of('/');
		executable_name.erase(0, delimiter_pos+1);
	#endif
	}
    
	// Loop over all search paths and return the first hit
	for(unsigned int i = 0; i < sizeof(searchPath)/sizeof(char*); ++i)
	{
		std::string path(searchPath[i]);        
		size_t executable_name_pos = path.find("<executable_name>");

		// If there is executable_name variable in the searchPath 
		// replace it with the value
		if(executable_name_pos != std::string::npos)
		{
			if(executable_path != 0) 
			{
				path.replace(executable_name_pos, strlen("<executable_name>"), executable_name);
			} 
			else 
			{
				// Skip this path entry if no executable argument is given
				continue;
			}
		}
        
		// Test if the file exists
		path.append(filename);
		std::fstream fh(path.c_str(), std::fstream::in);
		if (fh.good())
		{
			// File found
			// returning an allocated array here for backwards compatibility reasons
			char* file_path = (char*) malloc(path.length() + 1);
		#ifdef _WIN32  
			strcpy_s(file_path, path.length() + 1, path.c_str());
		#else
			strcpy(file_path, path.c_str());
		#endif                
			return file_path;
		}
	}    

	// File not found
	return 0;
}