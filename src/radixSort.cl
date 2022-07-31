#ifndef __OPENCL_VERSION__
#include <OpenCL/OpenCLKernel.hpp> // Hack to make syntax highlighting in Eclipse work
#endif

#define KEY_SIZE   32
#define DIGIT_SIZE 8
#define GLOBAL_WORK_SIZE 2

/**
Compute the upfront global histogram, by first locally aggregating, then to the global
    @param data: global, data to be handled
    @param histogram: global, histogram to for all digits of every digit places
    @param local_hist: local, local histogram used to store intermediate histograms
 */
__kernel void upfrontHistogram(__global const uint4* data, 
                               __global uint* histogram, 
                               __local uint* local_hist) 
{
    uint global_id = get_global_id(0);
    uint local_id = get_local_id(0);

    uint4 local_data;
    local_data = data[global_id];
    uint num_digit_binnings = 1 << DIGIT_SIZE;
    uint num_digit_places = KEY_SIZE / DIGIT_SIZE;
    //printf("block %d/%d thread %d/%d \n", get_group_id(0), get_num_groups(0), get_local_id(0), get_local_size(0));
    
    if (local_id < num_digit_places) {
        for (uint i = 0; i < num_digit_binnings; i++) {
            local_hist[local_id * num_digit_binnings + i] = 0;
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    // --------------------------------------------------------
    // use counting sort to get local histogram for each block
    // --------------------------------------------------------
    for (uint i = 0; i < num_digit_places; i++)
    {
        int idx_0 = (local_data.s0 >> (i*DIGIT_SIZE)) & 0xff;
        int idx_1 = (local_data.s1 >> (i*DIGIT_SIZE)) & 0xff;
        int idx_2 = (local_data.s2 >> (i*DIGIT_SIZE)) & 0xff;
        int idx_3 = (local_data.s3 >> (i*DIGIT_SIZE)) & 0xff;

        atomic_inc(&local_hist[idx_0 + i * num_digit_binnings]);
        atomic_inc(&local_hist[idx_1 + i * num_digit_binnings]);
        atomic_inc(&local_hist[idx_2 + i * num_digit_binnings]);
        atomic_inc(&local_hist[idx_3 + i * num_digit_binnings]);
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    // --------------------------------------------------------
    // Add the local histogram to global histogram atomically
    // --------------------------------------------------------
    if(local_id < num_digit_places) {
        for (int i = 0; i < num_digit_binnings; i++) {
            atomic_add(&histogram[local_id * num_digit_binnings + i], local_hist[local_id * num_digit_binnings + i]);
        }
    }
}

/**
Scan along each digit place.
    Single block with #digitPlaces threads
    @param histogram: global, global histogram
*/
__kernel void exclusiveSum(__global uint* histogram)
{
    //printf("block %d/%d thread %d/%d \n", get_group_id(0), get_num_groups(0), get_local_id(0), get_local_size(0));
    //int group_id = get_group_id(0);
    uint local_id = get_local_id(0);
    uint num_digit_binnings = 1 << DIGIT_SIZE;
    uint sum = 0;

    for (uint i = 0; i < num_digit_binnings; i++) {
        //printf("DB %d at dp %d: %d\n", i, local_id, histogram[local_id*num_digit_binnings + i]);
        uint tmp = histogram[local_id*num_digit_binnings + i]; 
        histogram[local_id*num_digit_binnings + i] = sum;
        sum += tmp;
    }

    //printf("[T%d] sum: %d\n", local_id, sum);
}

/** 
Chianed scan digit binnings of some digit place 
    @param data, pointer to the data in global memory
    @param histogram, histogram with global offsets for each digit at different digit places
    @param partition_counter, used to instruct the workgrounps to fetch data in sequence
    @param EPS: local, exclusive prefix sum
    @param flag_IPS: global, flag and inclusive prefix sum 
                decoupled look-back type:
                    bits 31-30: 3 states, namely 
                        - 00: not ready
                        - 01: local count
                        - 10: global sum
                    bits 29-0: prefix sum
*/
__kernel void chainedScanDigitBinning(__global uint* data,                  // 0 
                                      __global const uint* histogram,       // 1
                                      __global uint* flag_IPS,              // 2 
                                      __local  uint* EPS,                   // 3 
                                      __local  uint* aggregate,             // 4 
                                      __global uint* partition_counter,     // 5 
                                      uint           digit_place            // 6
                                      /**__global uint* num_reads,             // 7
                                      uint           num_tiles*/)             // 8
{
    // Fetch global data to local momory
    // data tiles should be handled in sequence
    uint wg_size = get_local_size(0);
    uint local_id = get_local_id(0);
    uint group_id = get_group_id(0);
    uint group_num = get_num_groups(0); 
    uint num_digit_binnings = 1 << DIGIT_SIZE;

    __local uint pid;
    if (local_id == 0) {
        pid = atomic_inc(partition_counter);
    }

    barrier(CLK_LOCAL_MEM_FENCE); // @@@@@@@

    uint4 local_data = (uint4)(data[pid*wg_size*4 + local_id*4], 
                               data[pid*wg_size*4 + local_id*4 + 1],
                               data[pid*wg_size*4 + local_id*4 + 2],
                               data[pid*wg_size*4 + local_id*4 + 3]);

    if (local_id < num_digit_binnings) {
        EPS[local_id] = 0; // init the exclusive prefix sum to 0s
        aggregate[local_id] = 0;
        //flag_IPS[pid * num_digit_binnings + local_id] = 0;
    }
/**
    if (local_id == 0) {
        uint r = atomic_inc(num_reads); 
        printf("%d read\n", r);
    }
*/
    barrier(CLK_LOCAL_MEM_FENCE); // @@@@@@@

    // -----------------------------------
    // Block-wide reduce
    // -----------------------------------
    // ANDing with mask to get digit
    uint idx_0 = (local_data.s0 >> (digit_place*DIGIT_SIZE)) & 0xff;
    uint idx_1 = (local_data.s1 >> (digit_place*DIGIT_SIZE)) & 0xff;
    uint idx_2 = (local_data.s2 >> (digit_place*DIGIT_SIZE)) & 0xff;
    uint idx_3 = (local_data.s3 >> (digit_place*DIGIT_SIZE)) & 0xff;
    
    atomic_inc(&aggregate[idx_0]);
    atomic_inc(&aggregate[idx_1]);
    atomic_inc(&aggregate[idx_2]);
    atomic_inc(&aggregate[idx_3]); 

    barrier(CLK_LOCAL_MEM_FENCE); // @@@@@@@

    // threads in each block finish reduce 
    // then the first thread in each block updates the status
    if (local_id < num_digit_binnings) {
        // change the state to "local count" (01) except the first partition to "global sum" (10)
        if (pid > 0) {
            //uint tmp = flag_IPS[pid*num_digit_binnings + local_id];
            flag_IPS[pid*num_digit_binnings + local_id] = (aggregate[local_id] | 0x40000000);  // OR with 0100 0000 ... 0000
            //printf("Check the val: %X -> %X\n", tmp, flag_IPS[pid*num_digit_binnings + local_id]);
        }

        else if (pid == 0) { // the first partition copy aggregate to IPS and set the flag
            EPS[local_id] = histogram[digit_place*num_digit_binnings + local_id];
            //printf("[B%d-T%d] fetch global hist. val %d at DP %d\n", group_id, local_id, EPS[local_id], digit_place);
            //uint tmp = flag_IPS[local_id];
            flag_IPS[local_id] = ((EPS[local_id] + aggregate[local_id]) | 0x80000000);
            //printf("Check the val: %X -> %X\n", tmp, flag_IPS[local_id]);
        }
    }
    //barrier(CLK_GLOBAL_MEM_FENCE); // @@@@@@@

    // decoupled look-back to determine the exclusive prefix sum
    if (pid > 0 && local_id < num_digit_binnings) {    // the first thread of each block will look back
        int ppid = pid - 1;
        while (ppid >= 0) {
            uint p_IPS = flag_IPS[ppid*num_digit_binnings + local_id];

            if ((p_IPS>>(KEY_SIZE-2)) == 2) {
                EPS[local_id] += (p_IPS & 0x3fffffff); // AND with 0011 1111 ... 1111
                //printf("[B%d-T%d_p%d] looks back until pid-%d with EPS=%d\n", group_id, local_id, pid, ppid, EPS[local_id]);
                break;
            } else if ((p_IPS>>(KEY_SIZE-2)) == 1) {
                EPS[local_id] += (p_IPS & 0x3fffffff);
                ppid--;
            }
        }
        // Compute inclusive prefix sum and set flag to "global count", i.e., 01 + 01 = 10
        flag_IPS[pid*num_digit_binnings + local_id] += (EPS[local_id] | 0x40000000); // OR 0100 0000 ... 0000
    }

    //barrier(CLK_LOCAL_MEM_FENCE); // @@@@@@@

    /** 
    // Compute inclusive prefix sum and set flag to "global count", i.e., 01 + 01 = 10
    if (pid > 0 && local_id < num_digit_binnings) {
        flag_IPS[pid*num_digit_binnings + local_id] += (EPS[local_id] | 0x40000000); // OR 0100 0000 ... 0000
    }
    */

    barrier(CLK_LOCAL_MEM_FENCE); // @@@@@@@
    //barrier(CLK_GLOBAL_MEM_FENCE); // @@@@@@@
    
    // -------------------------------------
    // Scan -> get global idx by summing up 
    //         prefix sum + local sum
    // (Block-wide operation)
    // -------------------------------------
    //while (*num_reads != num_tiles);
    
    for (uint i = 0; i < wg_size; i++) {
        if (local_id == i) {
            //printf("[DP %d] block %d/%d thread %d/%d - ESP[%d]: %d\n", digit_place, group_id, group_num, local_id, wg_size, idx_0, EPS[idx_0]);
            //printf("%4d, %4d, %4d, %5d, %8X, %5d\n", group_id, local_id, digit_place, pid, local_data.s0, EPS[idx_0]);
            //printf("%4d, %4d, %4d, %5d, %8X, %5d\n", group_id, local_id, digit_place, pid, local_data.s1, EPS[idx_1]);
            //printf("%4d, %4d, %4d, %5d, %8X, %5d\n", group_id, local_id, digit_place, pid, local_data.s2, EPS[idx_2]);
            //printf("%4d, %4d, %4d, %5d, %8X, %5d\n", group_id, local_id, digit_place, pid, local_data.s3, EPS[idx_3]);
            data[EPS[idx_0]] = local_data.s0;
            EPS[idx_0] = EPS[idx_0] + 1;
            data[EPS[idx_1]] = local_data.s1;
            EPS[idx_1] = EPS[idx_1] + 1;
            data[EPS[idx_2]] = local_data.s2;
            EPS[idx_2] = EPS[idx_2] + 1;
            data[EPS[idx_3]] = local_data.s3;
            EPS[idx_3] = EPS[idx_3] + 1;
        }
        barrier(CLK_GLOBAL_MEM_FENCE); // @@@@@@@
    }
}
