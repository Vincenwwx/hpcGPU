#ifndef __OPENCL_VERSION__
#include <OpenCL/OpenCLKernel.hpp> // Hack to make syntax highlighting in Eclipse work
#endif

/*
MSD radix sort algorithm

this kernel only sorts the given input array according to the Most Significant Digit

//this part is like counting Sort

*/



//To get digit at "pos" position(starting at position 1 as Least Significant Digit(LSD)) :
// digit = (int)(number / pow(10, (pos - 1))) % 10;



__kernel void kernel1 (__global const int* d_input, __global int* d_output, __global int* d_count, __global int* d_pos) {
	size_t index = get_global_id(0);

    int startAt =  index * *d_count;
    int endAtExclusive = startAt + *d_count;

     //start at  index *d_count 
     // to  index *d_count +   *d_count - 1


   __local int C[10];

   int pos = 1;  //  digit to sort after  (0 = right most digit)
   pos = *d_pos;

	//d_output[index] = cos (d_input[index]);
 
 // init helper array with zeros
  for (int m=0; m<10; m=m+1)  {
    C[m] = 0;
  }
  

 /* for (int j=startAt; j < endAtExclusive; j = j+1) {   //  10 just temp   should be  sizeof(d_input)   -> *d_count
   
   //C[d_input[j]] = C[d_input[j]] + 1;

    
   C[(int)(d_input[j] / pow((double)10, (double)(pos - 1))) % 10 ] = C[(int)(d_input[j] / pow((double)10, (double)(pos - 1))) % 10] + 1;

  
  
   

  } */

  //try reverse  to be inorder

   for (int j=endAtExclusive -1; j >= startAt; j = j -1 ) {   //  10 just temp   should be  sizeof(d_input)   -> *d_count
   
   //C[d_input[j]] = C[d_input[j]] + 1;

    
   C[(int)(d_input[j] / pow((double)10, (double)(pos - 1))) % 10 ] = C[(int)(d_input[j] / pow((double)10, (double)(pos - 1))) % 10] + 1;

  

  }
  
   

  int i=0;

 
  int k = 10;  //? is that true?  i hope so because 10 digits..

 //präfixsumme in C spechichern
 for(int m=0; m < 10; m = m+1) {
     if(m>0)
         C[m] = C[m-1]+C[m];
     
 }



 // for(int m=startAt; m <  endAtExclusive; m = m+1) {
  for(int m=endAtExclusive -1; m >= startAt ; m = m-1) {

   //d_output[C[d_input[m]]] = d_input[m];

   d_output[  C[(int)(d_input[m] / pow((double)10, (double)(pos - 1))) % 10] -1 +  startAt ] = d_input[m];


   if(C[(int)(d_input[m] / pow((double)10, (double)(pos - 1))) % 10] > 0) {
        C[(int)(d_input[m] / pow((double)10, (double)(pos - 1))) % 10] = C[(int)(d_input[m] / pow((double)10, (double)(pos - 1))) % 10] -1;
     }
     

   }



  


  /*
  for(int m=0; m < k; m = m+1) {

    while (C[m]>0) {
      //d_output[i] = m;
      d_output[i] = d_input[m];
      C[m] = C[m]-1;
      i = i+1;
  }
  }
  */

 
 // d_output[0] = *d_count;





 //d_output[0]=    (int)(15 / pow((double)10, (double)(pos - 1))) % 10;

 // d_output[ index] = *d_count * index*1000;

	// d_output[index] = startAt;
	//d_output[index] = native_cos (d_input[index]);
}
