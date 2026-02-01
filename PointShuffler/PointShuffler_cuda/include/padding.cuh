#include "common.cuh"





template <typename T>
__global__ void parallel_prefix(const T *atten_mask, int *batch_idx, int *word_idx,
                                const int batch_size, const int max_seq_len, const int channel, int* print,int* d_print ) {
  const int tid = threadIdx.x;
  const int warp_count = blockDim.x >> 5;
  int warp_id = tid >> 5;
  int warp_tid = tid & 0x1F;

  extern __shared__ int base[];

  int *seq_len = base;
  int *seq_offset = base + batch_size;

  for (int wid = warp_id; wid < batch_size; wid += warp_count) {
    int count = 0;
    for (int i = warp_tid; i < (max_seq_len + 31) / 32 * 32; i += 32) {

      T mask = i < max_seq_len ? atten_mask[wid * max_seq_len * channel + i*channel] : (T)0.0f;
      count += __popc(__ballot_sync(0xFFFFFFFF, mask >= (T)0.0f));
    }
    if (warp_tid == 0)
      seq_len[wid] = count;
  }

  __syncthreads();




  if (warp_id == 0) {
    int offset = 0, temp = 0;
    for (int i = warp_tid; i < ((batch_size + 31) / 32) * 32; i += 32) {
      offset = warp_tid == 0 ? temp : 0;
      int len = i < batch_size ? seq_len[i] : 0;
      temp = warpPrefixSum(warp_tid, offset + len);
      if (i < batch_size)
        seq_offset[i] = temp - len;

      temp = __shfl_sync(0xffffffff, temp, 31);
    }
    if (warp_tid == 0)
      seq_offset[batch_size] = temp;
  }

  __syncthreads();
  

  for (int i = tid; i <= batch_size; i += blockDim.x)
    batch_idx[i] = seq_offset[i];

  for (int wid = warp_id; wid < batch_size; wid += warp_count) {
    int offset = seq_offset[wid];
    for (int i = warp_tid; i < seq_len[wid]; i += 32)
      word_idx[offset + i] = wid * max_seq_len + i;
  }
  
  __syncthreads();
  for(int wid = warp_id;wid<batch_size;wid += warp_count)
  {
    for (int i = warp_tid; i < (max_seq_len + 31) / 32 * 32; i += 32)
      d_print[wid*max_seq_len+i]=word_idx[wid*max_seq_len+i];
  }
}

template <typename T>
void build_sequence_length_padding_offset_kernelLauncher(const T *atten_mask, int *batch_idx,
                                                         int *word_idx, int *valid_word_num,
                                                         const int batch_size,
                                                         const int max_seq_len, const int channel, int* print, int* d_print,
                                                         cudaStream_t stream) {
  dim3 block(batch_size * 32);  
  if (block.x > 1024)
    block.x = 1024;
  parallel_prefix<<<1, block, (2 * batch_size + 1) * sizeof(int), stream>>>(
      atten_mask, batch_idx, word_idx, batch_size, max_seq_len, channel , print, d_print);
  cudaMemcpyAsync(valid_word_num, batch_idx + batch_size, sizeof(int), cudaMemcpyDeviceToHost,stream);
  cudaMemcpyAsync(print,d_print,batch_size*max_seq_len* sizeof(int), cudaMemcpyDeviceToHost,stream);
}
