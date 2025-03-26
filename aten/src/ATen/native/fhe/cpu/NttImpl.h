#pragma once
#include <thread>
#include <chrono>
#include "ATen/native/fhe/cpu/Utils.h"
#include<iostream>
#include <stdexcept>
#include <tuple>
#include <cstdint>
#include<omp.h>
namespace fhe {

static inline __attribute__((always_inline))
void butt_intt_local(
    uint64_t& x,
    uint64_t& y,
    const uint64_t& w,
    const uint64_t& w_,
    const uint64_t& p) {
  const uint64_t two_p = 2 * p;
  const uint64_t T = two_p - y + x;
  uint64_t new_x = x + y;
  if (new_x >= two_p)
    new_x -= two_p;
  if (T & 1)
    new_x += p;
  x = (new_x >> 1);
  y = mul_and_reduce_shoup(T, w, w_, p);
}

void Intt8PointPerThreadPhase2OoP(
    uint64_t* in,
    const int m,
    const int num_prime,
    const int N,
    const int start_prime_idx,
    const int ceil_curr_limbs,
    const int gap,
    const int radix,
    const uint64_t* base_inv,
    const uint64_t* base_inv_,
    const uint64_t* primes,
    uint64_t* out,
    size_t GRID_SIZE,
    size_t BLOCK_SIZE,
    size_t SHARED_SIZE) {

      //fhe::Intt8PointPerThreadPhase2OoP(
        //             op_ptr,
        //             first_stage_radix_size,
        //             batch,
        //             param_degree,
        //             start_prime_idx,
        //             curr_limbs,
        //             gap,
        //             second_radix_size / per_thread_ntt_size,
        //             inverse_power_of_roots_div_two_ptr,
        //             inverse_scaled_power_of_roots_div_two_ptr,
        //             param_primes_ptr,
        //             op_ptr,
        //             gridDim,
        //             blockDim,
        //             per_thread_storage / sizeof(uint64_t));
    
  for(size_t blockIdx_x = 0; blockIdx_x < GRID_SIZE; blockIdx_x++) {
    std::vector<uint64_t> temp_vec(SHARED_SIZE, 0);
    uint64_t* temp = temp_vec.data();
    std::vector<std::array<uint64_t, 8>> local(BLOCK_SIZE, {0});
    std::vector<uint64_t> tail(BLOCK_SIZE, 0);
    
    for(size_t threadIdx_x = 0; threadIdx_x < BLOCK_SIZE; threadIdx_x++) {
      int set = threadIdx_x / radix;
      for (int i = blockIdx_x * BLOCK_SIZE + threadIdx_x; i < (N / 8 * num_prime); i += BLOCK_SIZE * GRID_SIZE) {
        
        int t = N / 2 / m;
        // prime idx
        int np_idx = i / (N / 8) + start_prime_idx;
        int prime_idx =
            np_idx + ((np_idx >= 0 && np_idx < ceil_curr_limbs) ? 0 : gap);
        // index in N/2 range
        int N_idx = i % (N / 8);
        // i'th block
        int m_idx = N_idx / (t / 4);
        int t_idx = N_idx % (t / 4);
        // base address
        uint64_t* in_addr = in + np_idx * N;
        uint64_t* out_addr = out + np_idx * N;
        const uint64_t* prime_table = primes;
        uint64_t prime = prime_table[prime_idx];
        int N_init = 2 * m_idx * t + t_idx;

        for (int j = 0; j < 8; j++) {
          temp[set * 8 * radix + t_idx + t / 4 * j] =
              *(in_addr + N_init + t / 4 * j);
        }
      }
    }
    __syncthreads();
    for(size_t threadIdx_x = 0; threadIdx_x < BLOCK_SIZE; threadIdx_x++) {
      int set = threadIdx_x / radix;
      for (int i = blockIdx_x * BLOCK_SIZE + threadIdx_x; i < (N / 8 * num_prime); i += BLOCK_SIZE * GRID_SIZE) {   
        
        int t = N / 2 / m;
        // prime idx
        int np_idx = i / (N / 8) + start_prime_idx;
        int prime_idx =
            np_idx + ((np_idx >= 0 && np_idx < ceil_curr_limbs) ? 0 : gap);
        // index in N/2 range
        int N_idx = i % (N / 8);
        // i'th block
        int m_idx = N_idx / (t / 4);
        int t_idx = N_idx % (t / 4);
        // base address
        uint64_t* in_addr = in + np_idx * N;
        uint64_t* out_addr = out + np_idx * N;
        const uint64_t* prime_table = primes;
        uint64_t prime = prime_table[prime_idx];
        int N_init = 2 * m_idx * t + t_idx;
        
        
        for (int l = 0; l < 8; l++) {
          local[threadIdx_x][l] = temp[set * 8 * radix + 8 * t_idx + l];
        }
        int tw_idx = m + m_idx;
        int tw_idx2 = (t / 4) * tw_idx + t_idx;
        const uint64_t* WInv = base_inv + N * prime_idx;
        const uint64_t* WInv_ = base_inv_ + N * prime_idx;
        for (int j = 0; j < 4; j++) {
          butt_intt_local(
              local[threadIdx_x][2 * j],
              local[threadIdx_x][2 * j + 1],
              WInv[4 * tw_idx2 + j],
              WInv_[4 * tw_idx2 + j],
              prime);
        }
        for (int j = 0; j < 2; j++) {
          butt_intt_local(
              local[threadIdx_x][4 * j],
              local[threadIdx_x][4 * j + 2],
              WInv[2 * tw_idx2 + j],
              WInv_[2 * tw_idx2 + j],
              prime);
          butt_intt_local(
              local[threadIdx_x][4 * j + 1],
              local[threadIdx_x][4 * j + 3],
              WInv[2 * tw_idx2 + j],
              WInv_[2 * tw_idx2 + j],
              prime);
        }
        for (int j = 0; j < 4; j++) {
          butt_intt_local(
              local[threadIdx_x][j], local[threadIdx_x][j + 4], WInv[tw_idx2], WInv_[tw_idx2], prime);
        }
      }
    }
    __syncthreads();
    for(size_t threadIdx_x = 0; threadIdx_x < BLOCK_SIZE; threadIdx_x++) {
      int set = threadIdx_x / radix;
      for (int i = blockIdx_x * BLOCK_SIZE + threadIdx_x; i < (N / 8 * num_prime); i += BLOCK_SIZE * GRID_SIZE) { 


        int t = N / 2 / m;
        // prime idx
        int np_idx = i / (N / 8) + start_prime_idx;
        int prime_idx =
            np_idx + ((np_idx >= 0 && np_idx < ceil_curr_limbs) ? 0 : gap);
        // index in N/2 range
        int N_idx = i % (N / 8);
        // i'th block
        int m_idx = N_idx / (t / 4);
        int t_idx = N_idx % (t / 4);
        // base address
        uint64_t* in_addr = in + np_idx * N;
        uint64_t* out_addr = out + np_idx * N;
        const uint64_t* prime_table = primes;
        uint64_t prime = prime_table[prime_idx];
        int N_init = 2 * m_idx * t + t_idx;


        for (int l = 0; l < 8; l++) {
          temp[set * 8 * radix + 8 * t_idx + l] = local[threadIdx_x][l];
        }
      }
    }
    __syncthreads();
    
    int t = N / 2 / m;

    for (int j = t / 32, k = 32; j > 0; j >>= 3, k *= 8) {
      for(size_t threadIdx_x = 0; threadIdx_x < BLOCK_SIZE; threadIdx_x++) {
        int set = threadIdx_x / radix;
        for (int i = blockIdx_x * BLOCK_SIZE + threadIdx_x; i < (N / 8 * num_prime); i += BLOCK_SIZE * GRID_SIZE) {  
          int t = N / 2 / m;
          // prime idx
          int np_idx = i / (N / 8) + start_prime_idx;
          int prime_idx =
              np_idx + ((np_idx >= 0 && np_idx < ceil_curr_limbs) ? 0 : gap);
          // index in N/2 range
          int N_idx = i % (N / 8);
          // i'th block
          int m_idx = N_idx / (t / 4);
          int t_idx = N_idx % (t / 4);
          // base address
          uint64_t* in_addr = in + np_idx * N;
          uint64_t* out_addr = out + np_idx * N;
          const uint64_t* prime_table = primes;
          uint64_t prime = prime_table[prime_idx];
          int N_init = 2 * m_idx * t + t_idx;


          int m_idx2 = t_idx / (k / 4);
          int t_idx2 = t_idx % (k / 4);
          for (int l = 0; l < 8; l++) {
            local[threadIdx_x][l] =
                temp[set * 8 * radix + 2 * m_idx2 * k + t_idx2 + (k / 4) * l];
          }
          
          int tw_idx = m + m_idx;
          int tw_idx2 = j * tw_idx + m_idx2;
          const uint64_t* WInv = base_inv + N * prime_idx;
          const uint64_t* WInv_ = base_inv_ + N * prime_idx;

          for (int l = 0; l < 4; l++) {
            butt_intt_local(
                local[threadIdx_x][2 * l],
                local[threadIdx_x][2 * l + 1],
                WInv[4 * tw_idx2 + l],
                WInv_[4 * tw_idx2 + l],
                prime);
          }
          for (int l = 0; l < 2; l++) {
            butt_intt_local(
                local[threadIdx_x][4 * l],
                local[threadIdx_x][4 * l + 2],
                WInv[2 * tw_idx2 + l],
                WInv_[2 * tw_idx2 + l],
                prime);
            butt_intt_local(
                local[threadIdx_x][4 * l + 1],
                local[threadIdx_x][4 * l + 3],
                WInv[2 * tw_idx2 + l],
                WInv_[2 * tw_idx2 + l],
                prime);
          }
          for (int l = 0; l < 4; l++) {
            butt_intt_local(
                local[threadIdx_x][l], local[threadIdx_x][l + 4], WInv[tw_idx2], WInv_[tw_idx2], prime);
          }
          for (int l = 0; l < 8; l++) {
            temp[set * 8 * radix + 2 * m_idx2 * k + t_idx2 + (k / 4) * l] =
                local[threadIdx_x][l];
          }
          if (j == 2)
            tail[threadIdx_x] = 1;
          if (j == 4)
            tail[threadIdx_x] = 2;
        }
      }
      __syncthreads();
    }

    for(size_t threadIdx_x = 0; threadIdx_x < BLOCK_SIZE; threadIdx_x++) {
      int set = threadIdx_x / radix;
      for (int i = blockIdx_x * BLOCK_SIZE + threadIdx_x; i < (N / 8 * num_prime); i += BLOCK_SIZE * GRID_SIZE) {  
        
        int t = N / 2 / m;
        // prime idx
        int np_idx = i / (N / 8) + start_prime_idx;
        int prime_idx =
            np_idx + ((np_idx >= 0 && np_idx < ceil_curr_limbs) ? 0 : gap);
        // index in N/2 range
        int N_idx = i % (N / 8);
        // i'th block
        int m_idx = N_idx / (t / 4);
        int t_idx = N_idx % (t / 4);
        // base address
        uint64_t* in_addr = in + np_idx * N;
        uint64_t* out_addr = out + np_idx * N;
        const uint64_t* prime_table = primes;
        uint64_t prime = prime_table[prime_idx];
        int N_init = 2 * m_idx * t + t_idx;
        
        int tw_idx = m + m_idx;
        const uint64_t* WInv = base_inv + N * prime_idx;
        const uint64_t* WInv_ = base_inv_ + N * prime_idx;

        if (tail[threadIdx_x] == 1) {
          for (int j = 0; j < 8; j++) {
            local[threadIdx_x][j] = temp[set * 8 * radix + t_idx + t / 4 * j];
          }
          butt_intt_local(local[threadIdx_x][0], local[threadIdx_x][4], WInv[tw_idx], WInv_[tw_idx], prime);
          butt_intt_local(local[threadIdx_x][1], local[threadIdx_x][5], WInv[tw_idx], WInv_[tw_idx], prime);
          butt_intt_local(local[threadIdx_x][2], local[threadIdx_x][6], WInv[tw_idx], WInv_[tw_idx], prime);
          butt_intt_local(local[threadIdx_x][3], local[threadIdx_x][7], WInv[tw_idx], WInv_[tw_idx], prime);
        } else if (tail[threadIdx_x] == 2) {
          for (int j = 0; j < 8; j++) {
            local[threadIdx_x][j] = temp[set * 8 * radix + t_idx + t / 4 * j];
          }
          butt_intt_local(
              local[threadIdx_x][0], local[threadIdx_x][2], WInv[2 * tw_idx], WInv_[2 * tw_idx], prime);
          butt_intt_local(
              local[threadIdx_x][1], local[threadIdx_x][3], WInv[2 * tw_idx], WInv_[2 * tw_idx], prime);
          butt_intt_local(
              local[threadIdx_x][4],
              local[threadIdx_x][6],
              WInv[2 * tw_idx + 1],
              WInv_[2 * tw_idx + 1],
              prime);
          butt_intt_local(
              local[threadIdx_x][5],
              local[threadIdx_x][7],
              WInv[2 * tw_idx + 1],
              WInv_[2 * tw_idx + 1],
              prime);
          butt_intt_local(local[threadIdx_x][0], local[threadIdx_x][4], WInv[tw_idx], WInv_[tw_idx], prime);
          butt_intt_local(local[threadIdx_x][1], local[threadIdx_x][5], WInv[tw_idx], WInv_[tw_idx], prime);
          butt_intt_local(local[threadIdx_x][2], local[threadIdx_x][6], WInv[tw_idx], WInv_[tw_idx], prime);
          butt_intt_local(local[threadIdx_x][3], local[threadIdx_x][7], WInv[tw_idx], WInv_[tw_idx], prime);
        }
        for (int j = 0; j < 8; j++) {
          *(out_addr + N_init + t / 4 * j) = local[threadIdx_x][j];
        }
      }
    }
  }
}

void Intt8PointPerThreadPhase1OoP(
    uint64_t* in,
    const int m,
    const int num_prime,
    const int N,
    const int start_prime_idx,
    const int ceil_curr_limbs,
    const int gap,
    int pad,
    int radix,
    const uint64_t* base_inv,
    const uint64_t* base_inv_,
    const uint64_t* primes,
    uint64_t* out,
    size_t GRID_SIZE,
    size_t BLOCK_SIZE,
    size_t SHARED_SIZE) {
  
  for(size_t blockIdx_x = 0; blockIdx_x < GRID_SIZE; blockIdx_x++) {
    std::vector<uint64_t> temp_vec(SHARED_SIZE, 0);
    uint64_t* temp = temp_vec.data();
    std::vector<std::array<uint64_t, 8>> local(BLOCK_SIZE, {0});
    std::vector<uint64_t> tail(BLOCK_SIZE, 0);
    for(size_t threadIdx_x = 0; threadIdx_x < BLOCK_SIZE; threadIdx_x++) {
      int Warp_t = threadIdx_x % pad;
      int WarpID = threadIdx_x / pad;
      for (int i = blockIdx_x * BLOCK_SIZE + threadIdx_x; i < (N / 8 * num_prime); i += BLOCK_SIZE * GRID_SIZE) {
        int t = N / 2 / m;
        // prime idx
        int np_idx = i / (N / 8) + start_prime_idx;
        int prime_idx =
            np_idx + ((np_idx >= 0 && np_idx < ceil_curr_limbs) ? 0 : gap);
        // index in N/2 range
        int N_idx = i % (N / 8);
        // i'th block
        int m_idx = N_idx / (t / 4);
        int t_idx = N_idx % (t / 4);
        // base address
        const uint64_t* in_addr = in + np_idx * N;
        uint64_t* out_addr = out + np_idx * N;
        const uint64_t* prime_table = primes;
        const uint64_t* WInv = base_inv + N * prime_idx;
        const uint64_t* WInv_ = base_inv_ + N * prime_idx;
        uint64_t prime = prime_table[prime_idx];
        int N_init =
            2 * t / radix * WarpID + Warp_t + pad * (t_idx / (radix * pad));
        for (int j = 0; j < 8; j++) {
          local[threadIdx_x][j] = *(in_addr + N_init + t / 4 / radix * j);
        }
        int eradix = 8 * radix;
        int tw_idx = m + m_idx;
        int tw_idx2 = radix * tw_idx + WarpID;
        for (int j = 0; j < 4; j++) {
          butt_intt_local(
              local[threadIdx_x][2 * j],
              local[threadIdx_x][2 * j + 1],
              WInv[4 * tw_idx2 + j],
              WInv_[4 * tw_idx2 + j],
              prime);
        }
        for (int j = 0; j < 2; j++) {
          butt_intt_local(
              local[threadIdx_x][4 * j],
              local[threadIdx_x][4 * j + 2],
              WInv[2 * tw_idx2 + j],
              WInv_[2 * tw_idx2 + j],
              prime);
          butt_intt_local(
              local[threadIdx_x][4 * j + 1],
              local[threadIdx_x][4 * j + 3],
              WInv[2 * tw_idx2 + j],
              WInv_[2 * tw_idx2 + j],
              prime);
        }
        for (int j = 0; j < 4; j++) {
          butt_intt_local(
              local[threadIdx_x][j], local[threadIdx_x][j + 4], WInv[tw_idx2], WInv_[tw_idx2], prime);
        }
        for (int j = 0; j < 8; j++) {
          temp[Warp_t * (eradix + pad) + 8 * WarpID + j] = local[threadIdx_x][j];
        }
      }
    }
    __syncthreads();
    for (int j = radix / 8, k = 32; j > 0; j >>= 3, k *= 8) {
      for(size_t threadIdx_x = 0; threadIdx_x < BLOCK_SIZE; threadIdx_x++) {
        int Warp_t = threadIdx_x % pad;
        int WarpID = threadIdx_x / pad;
        for (int i = blockIdx_x * BLOCK_SIZE + threadIdx_x; i < (N / 8 * num_prime); i += BLOCK_SIZE * GRID_SIZE) {

          int t = N / 2 / m;
          // prime idx
          int np_idx = i / (N / 8) + start_prime_idx;
          int prime_idx =
              np_idx + ((np_idx >= 0 && np_idx < ceil_curr_limbs) ? 0 : gap);
          // index in N/2 range
          int N_idx = i % (N / 8);
          // i'th block
          int m_idx = N_idx / (t / 4);
          int t_idx = N_idx % (t / 4);
          // base address
          const uint64_t* in_addr = in + np_idx * N;
          uint64_t* out_addr = out + np_idx * N;
          const uint64_t* prime_table = primes;
          const uint64_t* WInv = base_inv + N * prime_idx;
          const uint64_t* WInv_ = base_inv_ + N * prime_idx;
          uint64_t prime = prime_table[prime_idx];
          int N_init =
              2 * t / radix * WarpID + Warp_t + pad * (t_idx / (radix * pad));
          int eradix = 8 * radix;
          int tw_idx = m + m_idx;

          int m_idx2 = WarpID / (k / 4);
          int t_idx2 = WarpID % (k / 4);
          for (int l = 0; l < 8; l++) {
            local[threadIdx_x][l] = temp
                [(eradix + pad) * Warp_t + 2 * m_idx2 * k + t_idx2 + (k / 4) * l];
          }
          int tw_idx2 = j * tw_idx + m_idx2;
          for (int l = 0; l < 4; l++) {
            butt_intt_local(
                local[threadIdx_x][2 * l],
                local[threadIdx_x][2 * l + 1],
                WInv[4 * tw_idx2 + l],
                WInv_[4 * tw_idx2 + l],
                prime);
          }
          for (int l = 0; l < 2; l++) {
            butt_intt_local(
                local[threadIdx_x][4 * l],
                local[threadIdx_x][4 * l + 2],
                WInv[2 * tw_idx2 + l],
                WInv_[2 * tw_idx2 + l],
                prime);
            butt_intt_local(
                local[threadIdx_x][4 * l + 1],
                local[threadIdx_x][4 * l + 3],
                WInv[2 * tw_idx2 + l],
                WInv_[2 * tw_idx2 + l],
                prime);
          }
          for (int l = 0; l < 4; l++) {
            butt_intt_local(
                local[threadIdx_x][l], local[threadIdx_x][l + 4], WInv[tw_idx2], WInv_[tw_idx2], prime);
          }
          for (int l = 0; l < 8; l++) {
            temp[(eradix + pad) * Warp_t + 2 * m_idx2 * k + t_idx2 + (k / 4) * l] =
                local[threadIdx_x][l];
          }
          if (j == 2)
            tail[threadIdx_x] = 1;
          if (j == 4)
            tail[threadIdx_x] = 2;
        }
      }
      __syncthreads();
    }
    for(size_t threadIdx_x = 0; threadIdx_x < BLOCK_SIZE; threadIdx_x++) {
      int Warp_t = threadIdx_x % pad;
      int WarpID = threadIdx_x / pad;
      for (int i = blockIdx_x * BLOCK_SIZE + threadIdx_x; i < (N / 8 * num_prime); i += BLOCK_SIZE * GRID_SIZE) {
        
        int t = N / 2 / m;
        // prime idx
        int np_idx = i / (N / 8) + start_prime_idx;
        int prime_idx =
            np_idx + ((np_idx >= 0 && np_idx < ceil_curr_limbs) ? 0 : gap);
        // index in N/2 range
        int N_idx = i % (N / 8);
        // i'th block
        int m_idx = N_idx / (t / 4);
        int t_idx = N_idx % (t / 4);
        // base address
        const uint64_t* in_addr = in + np_idx * N;
        uint64_t* out_addr = out + np_idx * N;
        const uint64_t* prime_table = primes;
        const uint64_t* WInv = base_inv + N * prime_idx;
        const uint64_t* WInv_ = base_inv_ + N * prime_idx;
        uint64_t prime = prime_table[prime_idx];
        int N_init =
            2 * t / radix * WarpID + Warp_t + pad * (t_idx / (radix * pad));
        int eradix = 8 * radix;
        int tw_idx = m + m_idx;


        if (radix < 8)
          tail[threadIdx_x] = (radix == 4) ? 2 : 1;
        for (int l = 0; l < 8; l++) {
          local[threadIdx_x][l] = temp[Warp_t * (eradix + pad) + WarpID + radix * l];
        }
        if (tail[threadIdx_x] == 1) {
          butt_intt_local(local[threadIdx_x][0], local[threadIdx_x][4], WInv[tw_idx], WInv_[tw_idx], prime);
          butt_intt_local(local[threadIdx_x][1], local[threadIdx_x][5], WInv[tw_idx], WInv_[tw_idx], prime);
          butt_intt_local(local[threadIdx_x][2], local[threadIdx_x][6], WInv[tw_idx], WInv_[tw_idx], prime);
          butt_intt_local(local[threadIdx_x][3], local[threadIdx_x][7], WInv[tw_idx], WInv_[tw_idx], prime);
        } else if (tail[threadIdx_x] == 2) {
          butt_intt_local(
              local[threadIdx_x][0], local[threadIdx_x][2], WInv[2 * tw_idx], WInv_[2 * tw_idx], prime);
          butt_intt_local(
              local[threadIdx_x][1], local[threadIdx_x][3], WInv[2 * tw_idx], WInv_[2 * tw_idx], prime);
          butt_intt_local(
              local[threadIdx_x][4],
              local[threadIdx_x][6],
              WInv[2 * tw_idx + 1],
              WInv_[2 * tw_idx + 1],
              prime);
          butt_intt_local(
              local[threadIdx_x][5],
              local[threadIdx_x][7],
              WInv[2 * tw_idx + 1],
              WInv_[2 * tw_idx + 1],
              prime);
          butt_intt_local(local[threadIdx_x][0], local[threadIdx_x][4], WInv[tw_idx], WInv_[tw_idx], prime);
          butt_intt_local(local[threadIdx_x][1], local[threadIdx_x][5], WInv[tw_idx], WInv_[tw_idx], prime);
          butt_intt_local(local[threadIdx_x][2], local[threadIdx_x][6], WInv[tw_idx], WInv_[tw_idx], prime);
          butt_intt_local(local[threadIdx_x][3], local[threadIdx_x][7], WInv[tw_idx], WInv_[tw_idx], prime);
        }
        for (int j = 0; j < 8; j++) {
          if (local[threadIdx_x][j] >= prime)
            local[threadIdx_x][j] -= prime;
        }
        N_init = t / 4 / radix * WarpID + Warp_t + pad * (t_idx / (radix * pad));
        for (int j = 0; j < 8; j++) {
          *(out_addr + N_init + t / 4 * j) = local[threadIdx_x][j];
        }
      }
    }
  }
}

static inline __attribute__((always_inline))
void butt_ntt_local(
    uint64_t& a,
    uint64_t& b,
    const uint64_t& w,
    const uint64_t& w_,
    const uint64_t p) {
  uint64_t two_p = 2 * p;
  uint64_t U = mul_and_reduce_shoup(b, w, w_, p);
  if (a >= two_p)
    a -= two_p;
  b = a + (two_p - U);
  a += U;
}

void Ntt8PointPerThreadPhase1(
    uint64_t* op,
    const int m,
    const int num_prime,
    const int N,
    const int start_prime_idx,
    const int pad,
    const int radix,
    const uint64_t* base_inv,
    const uint64_t* base_inv_,
    const uint64_t* primes,
    size_t GRID_SIZE,//2048
    size_t BLOCK_SIZE,//128
    size_t SHARED_SIZE) {
//         fhe::Ntt8PointPerThreadPhase1(
//             op_ptr,
//             1,
//             batch,
//             param_degree,
//             start_prime_idx,
//             pad,
//             first_stage_radix_size / per_thread_ntt_size,
//             param_power_of_roots_ptr,
//             param_power_of_roots_shoup_ptr,
//             param_primes_ptr,
//             gridDim,
//             (first_stage_radix_size / 8) * pad,
//             (first_stage_radix_size + pad + 1) * pad);
  for(size_t blockIdx_x = 0; blockIdx_x < GRID_SIZE; blockIdx_x++) {
    std::vector<uint64_t> temp_vec(SHARED_SIZE, 0);
    uint64_t* temp = temp_vec.data();
    std::vector<std::array<uint64_t, 8>> local(BLOCK_SIZE, {0});
    std::vector<uint64_t> tail(BLOCK_SIZE, 0);
    for(size_t threadIdx_x = 0; threadIdx_x < BLOCK_SIZE; threadIdx_x++) {
      int Warp_t = threadIdx_x % pad;
      int WarpID = threadIdx_x / pad;
      for (int i = blockIdx_x * BLOCK_SIZE + threadIdx_x; i < (N / 8 * num_prime); i += BLOCK_SIZE * GRID_SIZE) {

        int t = N / 2 / m;
        // prime idx
        int np_idx = i / (N / 8) + start_prime_idx;
        // index in N/2 range
        int N_idx = i % (N / 8);
        // i'th block
        int m_idx = N_idx / (t / 4);
        int t_idx = N_idx % (t / 4);
        // base address
        uint64_t* a_np = op + np_idx * N;
        const uint64_t* prime_table = primes;
        const uint64_t* W = base_inv + N * np_idx;
        const uint64_t* W_ = base_inv_ + N * np_idx;
        uint64_t prime = prime_table[np_idx];
        int N_init = 2 * m_idx * t + t / 4 / radix * WarpID + Warp_t +
            pad * (t_idx / (radix * pad));
        for (int j = 0; j < 8; j++) {
          local[threadIdx_x][j] = *(a_np + N_init + t / 4 * j);
        }
      }
    }
    __syncthreads();
    for(size_t threadIdx_x = 0; threadIdx_x < BLOCK_SIZE; threadIdx_x++) {
      int Warp_t = threadIdx_x % pad;
      int WarpID = threadIdx_x / pad;
      for (int i = blockIdx_x * BLOCK_SIZE + threadIdx_x; i < (N / 8 * num_prime); i += BLOCK_SIZE * GRID_SIZE) {
        
        int t = N / 2 / m;
        // prime idx
        int np_idx = i / (N / 8) + start_prime_idx;
        // index in N/2 range
        int N_idx = i % (N / 8);
        // i'th block
        int m_idx = N_idx / (t / 4);
        int t_idx = N_idx % (t / 4);
        // base address
        uint64_t* a_np = op + np_idx * N;
        const uint64_t* prime_table = primes;
        const uint64_t* W = base_inv + N * np_idx;
        const uint64_t* W_ = base_inv_ + N * np_idx;
        uint64_t prime = prime_table[np_idx];
        int N_init = 2 * m_idx * t + t / 4 / radix * WarpID + Warp_t +
            pad * (t_idx / (radix * pad));

        int eradix = 8 * radix;
        int tw_idx = m + m_idx;
        for (int j = 0; j < 4; j++) {
          butt_ntt_local(local[threadIdx_x][j], local[threadIdx_x][j + 4], W[tw_idx], W_[tw_idx], prime);
        }
        for (int j = 0; j < 2; j++) {
          butt_ntt_local(
              local[threadIdx_x][4 * j],
              local[threadIdx_x][4 * j + 2],
              W[2 * tw_idx + j],
              W_[2 * tw_idx + j],
              prime);
          butt_ntt_local(
              local[threadIdx_x][4 * j + 1],
              local[threadIdx_x][4 * j + 3],
              W[2 * tw_idx + j],
              W_[2 * tw_idx + j],
              prime);
        }
        for (int j = 0; j < 4; j++) {
          butt_ntt_local(
              local[threadIdx_x][2 * j],
              local[threadIdx_x][2 * j + 1],
              W[4 * tw_idx + j],
              W_[4 * tw_idx + j],
              prime);
        }
        for (int j = 0; j < 8; j++) {
          temp[Warp_t * (eradix + pad) + WarpID + radix * j] = local[threadIdx_x][j];
        }
      }
    }
    __syncthreads();
    for (int j = 8, k = radix / 2; j < radix + 1; j *= 8, k >>= 3) {
      for(size_t threadIdx_x = 0; threadIdx_x < BLOCK_SIZE; threadIdx_x++) {
        int Warp_t = threadIdx_x % pad;
        int WarpID = threadIdx_x / pad;
        for (int i = blockIdx_x * BLOCK_SIZE + threadIdx_x; i < (N / 8 * num_prime); i += BLOCK_SIZE * GRID_SIZE) {

          int t = N / 2 / m;
          // prime idx
          int np_idx = i / (N / 8) + start_prime_idx;
          // index in N/2 range
          int N_idx = i % (N / 8);
          // i'th block
          int m_idx = N_idx / (t / 4);
          int t_idx = N_idx % (t / 4);
          // base address
          uint64_t* a_np = op + np_idx * N;
          const uint64_t* prime_table = primes;
          const uint64_t* W = base_inv + N * np_idx;
          const uint64_t* W_ = base_inv_ + N * np_idx;
          uint64_t prime = prime_table[np_idx];
          int N_init = 2 * m_idx * t + t / 4 / radix * WarpID + Warp_t +
              pad * (t_idx / (radix * pad));

          int eradix = 8 * radix;
          int tw_idx = m + m_idx;

          int m_idx2 = WarpID / (k / 4);
          int t_idx2 = WarpID % (k / 4);
          for (int l = 0; l < 8; l++) {
            local[threadIdx_x][l] = temp
                [(eradix + pad) * Warp_t + 2 * m_idx2 * k + t_idx2 + (k / 4) * l];
          }
          int tw_idx2 = j * tw_idx + m_idx2;
          for (int j2 = 0; j2 < 4; j2++) {
            butt_ntt_local(
                local[threadIdx_x][j2], local[threadIdx_x][j2 + 4], W[tw_idx2], W_[tw_idx2], prime);
          }
          for (int j2 = 0; j2 < 2; j2++) {
            butt_ntt_local(
                local[threadIdx_x][4 * j2],
                local[threadIdx_x][4 * j2 + 2],
                W[2 * tw_idx2 + j2],
                W_[2 * tw_idx2 + j2],
                prime);
            butt_ntt_local(
                local[threadIdx_x][4 * j2 + 1],
                local[threadIdx_x][4 * j2 + 3],
                W[2 * tw_idx2 + j2],
                W_[2 * tw_idx2 + j2],
                prime);
          }
          for (int j2 = 0; j2 < 4; j2++) {
            butt_ntt_local(
                local[threadIdx_x][2 * j2],
                local[threadIdx_x][2 * j2 + 1],
                W[4 * tw_idx2 + j2],
                W_[4 * tw_idx2 + j2],
                prime);
          }

          for (int l = 0; l < 8; l++) {
            temp[(eradix + pad) * Warp_t + 2 * m_idx2 * k + t_idx2 + (k / 4) * l] =
                local[threadIdx_x][l];
          }
          if (j == radix / 2)
            tail[threadIdx_x] = 1;
          if (j == radix / 4)
            tail[threadIdx_x] = 2;
        }
      }
      __syncthreads();
    }
    for(size_t threadIdx_x = 0; threadIdx_x < BLOCK_SIZE; threadIdx_x++) {
      int Warp_t = threadIdx_x % pad;
      int WarpID = threadIdx_x / pad;
      for (int i = blockIdx_x * BLOCK_SIZE + threadIdx_x; i < (N / 8 * num_prime); i += BLOCK_SIZE * GRID_SIZE) {

        int t = N / 2 / m;
        // prime idx
        int np_idx = i / (N / 8) + start_prime_idx;
        // index in N/2 range
        int N_idx = i % (N / 8);
        // i'th block
        int m_idx = N_idx / (t / 4);
        int t_idx = N_idx % (t / 4);
        // base address
        uint64_t* a_np = op + np_idx * N;
        const uint64_t* prime_table = primes;
        const uint64_t* W = base_inv + N * np_idx;
        const uint64_t* W_ = base_inv_ + N * np_idx;
        uint64_t prime = prime_table[np_idx];
        int N_init = 2 * m_idx * t + t / 4 / radix * WarpID + Warp_t +
            pad * (t_idx / (radix * pad));
            
        int eradix = 8 * radix;
        int tw_idx = m + m_idx;

        if (radix < 8)
          tail[threadIdx_x] = (radix == 4) ? 2 : 1;
        if (tail[threadIdx_x] == 1) {
          for (int l = 0; l < 8; l++) {
            local[threadIdx_x][l] = temp[(eradix + pad) * Warp_t + 8 * WarpID + l];
          }
          int tw_idx2 = (4 * radix) * tw_idx + 4 * WarpID;
          butt_ntt_local(local[threadIdx_x][0], local[threadIdx_x][1], W[tw_idx2], W_[tw_idx2], prime);
          butt_ntt_local(
              local[threadIdx_x][2], local[threadIdx_x][3], W[tw_idx2 + 1], W_[tw_idx2 + 1], prime);
          butt_ntt_local(
              local[threadIdx_x][4], local[threadIdx_x][5], W[tw_idx2 + 2], W_[tw_idx2 + 2], prime);
          butt_ntt_local(
              local[threadIdx_x][6], local[threadIdx_x][7], W[tw_idx2 + 3], W_[tw_idx2 + 3], prime);
          for (int l = 0; l < 8; l++) {
            temp[(eradix + pad) * Warp_t + 8 * WarpID + l] = local[threadIdx_x][l];
          }
        } else if (tail[threadIdx_x] == 2) {
          for (int l = 0; l < 8; l++) {
            local[threadIdx_x][l] = temp[(eradix + pad) * Warp_t + 8 * WarpID + l];
          }
          int tw_idx2 = 2 * radix * tw_idx + 2 * WarpID;
          butt_ntt_local(local[threadIdx_x][0], local[threadIdx_x][2], W[tw_idx2], W_[tw_idx2], prime);
          butt_ntt_local(local[threadIdx_x][1], local[threadIdx_x][3], W[tw_idx2], W_[tw_idx2], prime);
          butt_ntt_local(
              local[threadIdx_x][4], local[threadIdx_x][6], W[tw_idx2 + 1], W_[tw_idx2 + 1], prime);
          butt_ntt_local(
              local[threadIdx_x][5], local[threadIdx_x][7], W[tw_idx2 + 1], W_[tw_idx2 + 1], prime);
          butt_ntt_local(
              local[threadIdx_x][0], local[threadIdx_x][1], W[2 * tw_idx2], W_[2 * tw_idx2], prime);
          butt_ntt_local(
              local[threadIdx_x][2], local[threadIdx_x][3], W[2 * tw_idx2 + 1], W_[2 * tw_idx2 + 1], prime);
          butt_ntt_local(
              local[threadIdx_x][4], local[threadIdx_x][5], W[2 * tw_idx2 + 2], W_[2 * tw_idx2 + 2], prime);
          butt_ntt_local(
              local[threadIdx_x][6], local[threadIdx_x][7], W[2 * tw_idx2 + 3], W_[2 * tw_idx2 + 3], prime);
          for (int l = 0; l < 8; l++) {
            temp[(eradix + pad) * Warp_t + 8 * WarpID + l] = local[threadIdx_x][l];
          }
        }
      }
    }
    __syncthreads();
    for(size_t threadIdx_x = 0; threadIdx_x < BLOCK_SIZE; threadIdx_x++) {
      int Warp_t = threadIdx_x % pad;
      int WarpID = threadIdx_x / pad;
      for (int i = blockIdx_x * BLOCK_SIZE + threadIdx_x; i < (N / 8 * num_prime); i += BLOCK_SIZE * GRID_SIZE) {
        
        int t = N / 2 / m;
        // prime idx
        int np_idx = i / (N / 8) + start_prime_idx;
        // index in N/2 range
        int N_idx = i % (N / 8);
        // i'th block
        int m_idx = N_idx / (t / 4);
        int t_idx = N_idx % (t / 4);
        // base address
        uint64_t* a_np = op + np_idx * N;
        const uint64_t* prime_table = primes;
        const uint64_t* W = base_inv + N * np_idx;
        const uint64_t* W_ = base_inv_ + N * np_idx;
        uint64_t prime = prime_table[np_idx];
        int N_init = 2 * m_idx * t + t / 4 / radix * WarpID + Warp_t +
            pad * (t_idx / (radix * pad));

        int eradix = 8 * radix;
        int tw_idx = m + m_idx;

        for (int j = 0; j < 8; j++) {
          local[threadIdx_x][j] = temp[Warp_t * (eradix + pad) + WarpID + radix * j];
        }
        for (int j = 0; j < 8; j++) {
          *(a_np + N_init + t / 4 * j) = local[threadIdx_x][j];
        }
      }
    }
  }
}

void Ntt8PointPerThreadPhase2(
    uint64_t* op,
    const int m,
    const int num_prime,
    const int N,
    const int start_prime_idx,
    const int radix,
    const uint64_t* base_inv,
    const uint64_t* base_inv_,
    const uint64_t* primes,
    size_t GRID_SIZE,
    size_t BLOCK_SIZE,
    size_t SHARED_SIZE) {

  for(size_t blockIdx_x = 0; blockIdx_x < GRID_SIZE; blockIdx_x++) {
    std::vector<uint64_t> temp_vec(SHARED_SIZE, 0);
    uint64_t* temp = temp_vec.data();
    std::vector<std::array<uint64_t, 8>> local(BLOCK_SIZE, {0});
    std::vector<uint64_t> tail(BLOCK_SIZE, 0);
    for(size_t threadIdx_x = 0; threadIdx_x < BLOCK_SIZE; threadIdx_x++) {
      int set = threadIdx_x / radix;
      for (int i = blockIdx_x * BLOCK_SIZE + threadIdx_x; i < (N / 8 * num_prime); i += BLOCK_SIZE * GRID_SIZE) {

        int t = N / 2 / m;
        // prime idx
        int np_idx = num_prime - 1 - (i / (N / 8)) + start_prime_idx;
        // index in N/2 range
        int N_idx = i % (N / 8);
        // i'th block
        int m_idx = N_idx / (t / 4);
        int t_idx = N_idx % (t / 4);
        // base address
        uint64_t* a_np = op + np_idx * N;
        const uint64_t* prime_table = primes;
        uint64_t prime = prime_table[np_idx];
        int N_init = 2 * m_idx * t + t_idx;
        for (int j = 0; j < 8; j++) {
          local[threadIdx_x][j] = *(a_np + N_init + t / 4 * j);
        }
        int tw_idx = m + m_idx;
        const uint64_t* W = base_inv + N * np_idx;
        const uint64_t* W_ = base_inv_ + N * np_idx;
        for (int j = 0; j < 4; j++) {
          butt_ntt_local(local[threadIdx_x][j], local[threadIdx_x][j + 4], W[tw_idx], W_[tw_idx], prime);
        }
        for (int j = 0; j < 2; j++) {
          butt_ntt_local(
              local[threadIdx_x][4 * j],
              local[threadIdx_x][4 * j + 2],
              W[2 * tw_idx + j],
              W_[2 * tw_idx + j],
              prime);
          butt_ntt_local(
              local[threadIdx_x][4 * j + 1],
              local[threadIdx_x][4 * j + 3],
              W[2 * tw_idx + j],
              W_[2 * tw_idx + j],
              prime);
        }
        for (int j = 0; j < 4; j++) {
          butt_ntt_local(
              local[threadIdx_x][2 * j],
              local[threadIdx_x][2 * j + 1],
              W[4 * tw_idx + j],
              W_[4 * tw_idx + j],
              prime);
        }
        for (int j = 0; j < 8; j++) {
          temp[set * 8 * radix + t_idx + t / 4 * j] = local[threadIdx_x][j];
        }
      }
    }
    __syncthreads();
    int t = N / 2 / m;
    for (int j = 8, k = t / 8; j < t / 4 + 1; j *= 8, k >>= 3) {
      for(size_t threadIdx_x = 0; threadIdx_x < BLOCK_SIZE; threadIdx_x++) {
        int set = threadIdx_x / radix;
        for (int i = blockIdx_x * BLOCK_SIZE + threadIdx_x; i < (N / 8 * num_prime); i += BLOCK_SIZE * GRID_SIZE) {
          int t = N / 2 / m;
          // prime idx
          int np_idx = num_prime - 1 - (i / (N / 8)) + start_prime_idx;
          // index in N/2 range
          int N_idx = i % (N / 8);
          // i'th block
          int m_idx = N_idx / (t / 4);
          int t_idx = N_idx % (t / 4);
          // base address
          uint64_t* a_np = op + np_idx * N;
          const uint64_t* prime_table = primes;
          uint64_t prime = prime_table[np_idx];
          int N_init = 2 * m_idx * t + t_idx;
          for (int j = 0; j < 8; j++) {
            local[threadIdx_x][j] = *(a_np + N_init + t / 4 * j);
          }
          int tw_idx = m + m_idx;
          const uint64_t* W = base_inv + N * np_idx;
          const uint64_t* W_ = base_inv_ + N * np_idx;

          
          int m_idx2 = t_idx / (k / 4);
          int t_idx2 = t_idx % (k / 4);
          for (int l = 0; l < 8; l++) {
            local[threadIdx_x][l] =
                temp[set * 8 * radix + 2 * m_idx2 * k + t_idx2 + (k / 4) * l];
          }
          int tw_idx2 = j * tw_idx + m_idx2;
          for (int j2 = 0; j2 < 4; j2++) {
            butt_ntt_local(
                local[threadIdx_x][j2], local[threadIdx_x][j2 + 4], W[tw_idx2], W_[tw_idx2], prime);
          }
          for (int j2 = 0; j2 < 2; j2++) {
            butt_ntt_local(
                local[threadIdx_x][4 * j2],
                local[threadIdx_x][4 * j2 + 2],
                W[2 * tw_idx2 + j2],
                W_[2 * tw_idx2 + j2],
                prime);
            butt_ntt_local(
                local[threadIdx_x][4 * j2 + 1],
                local[threadIdx_x][4 * j2 + 3],
                W[2 * tw_idx2 + j2],
                W_[2 * tw_idx2 + j2],
                prime);
          }
          for (int j2 = 0; j2 < 4; j2++) {
            butt_ntt_local(
                local[threadIdx_x][2 * j2],
                local[threadIdx_x][2 * j2 + 1],
                W[4 * tw_idx2 + j2],
                W_[4 * tw_idx2 + j2],
                prime);
          }

          for (int l = 0; l < 8; l++) {
            temp[set * 8 * radix + 2 * m_idx2 * k + t_idx2 + (k / 4) * l] =
                local[threadIdx_x][l];
          }
          if (j == t / 8)
            tail[threadIdx_x] = 1;
          if (j == t / 16)
            tail[threadIdx_x] = 2;
        }
      }
      __syncthreads();
    }
    for(size_t threadIdx_x = 0; threadIdx_x < BLOCK_SIZE; threadIdx_x++) {
      int set = threadIdx_x / radix;
      for (int i = blockIdx_x * BLOCK_SIZE + threadIdx_x; i < (N / 8 * num_prime); i += BLOCK_SIZE * GRID_SIZE) {

        int t = N / 2 / m;
        // prime idx
        int np_idx = num_prime - 1 - (i / (N / 8)) + start_prime_idx;
        // index in N/2 range
        int N_idx = i % (N / 8);
        // i'th block
        int m_idx = N_idx / (t / 4);
        int t_idx = N_idx % (t / 4);
        // base address
        uint64_t* a_np = op + np_idx * N;
        const uint64_t* prime_table = primes;
        uint64_t prime = prime_table[np_idx];
        int N_init = 2 * m_idx * t + t_idx;
        for (int j = 0; j < 8; j++) {
          local[threadIdx_x][j] = *(a_np + N_init + t / 4 * j);
        }
        int tw_idx = m + m_idx;
        const uint64_t* W = base_inv + N * np_idx;
        const uint64_t* W_ = base_inv_ + N * np_idx;


        if (tail[threadIdx_x] == 1) {
          for (int l = 0; l < 8; l++) {
            local[threadIdx_x][l] = temp[set * 8 * radix + 8 * t_idx + l];
          }
          int tw_idx2 = t * tw_idx + 4 * t_idx;
          butt_ntt_local(local[threadIdx_x][0], local[threadIdx_x][1], W[tw_idx2], W_[tw_idx2], prime);
          butt_ntt_local(
              local[threadIdx_x][2], local[threadIdx_x][3], W[tw_idx2 + 1], W_[tw_idx2 + 1], prime);
          butt_ntt_local(
              local[threadIdx_x][4], local[threadIdx_x][5], W[tw_idx2 + 2], W_[tw_idx2 + 2], prime);
          butt_ntt_local(
              local[threadIdx_x][6], local[threadIdx_x][7], W[tw_idx2 + 3], W_[tw_idx2 + 3], prime);
          for (int l = 0; l < 8; l++) {
            temp[set * 8 * radix + 8 * t_idx + l] = local[threadIdx_x][l];
          }
        } else if (tail[threadIdx_x] == 2) {
          for (int l = 0; l < 8; l++) {
            local[threadIdx_x][l] = temp[set * 8 * radix + 8 * t_idx + l];
          }
          int tw_idx2 = (t / 2) * tw_idx + 2 * t_idx;
          butt_ntt_local(local[threadIdx_x][0], local[threadIdx_x][2], W[tw_idx2], W_[tw_idx2], prime);
          butt_ntt_local(local[threadIdx_x][1], local[threadIdx_x][3], W[tw_idx2], W_[tw_idx2], prime);
          butt_ntt_local(
              local[threadIdx_x][4], local[threadIdx_x][6], W[tw_idx2 + 1], W_[tw_idx2 + 1], prime);
          butt_ntt_local(
              local[threadIdx_x][5], local[threadIdx_x][7], W[tw_idx2 + 1], W_[tw_idx2 + 1], prime);
          butt_ntt_local(
              local[threadIdx_x][0], local[threadIdx_x][1], W[2 * tw_idx2], W_[2 * tw_idx2], prime);
          butt_ntt_local(
              local[threadIdx_x][2], local[threadIdx_x][3], W[2 * tw_idx2 + 1], W_[2 * tw_idx2 + 1], prime);
          butt_ntt_local(
              local[threadIdx_x][4], local[threadIdx_x][5], W[2 * tw_idx2 + 2], W_[2 * tw_idx2 + 2], prime);
          butt_ntt_local(
              local[threadIdx_x][6], local[threadIdx_x][7], W[2 * tw_idx2 + 3], W_[2 * tw_idx2 + 3], prime);
          for (int l = 0; l < 8; l++) {
            temp[set * 8 * radix + 8 * t_idx + l] = local[threadIdx_x][l];
          }
        }
      }
    }
    __syncthreads();
    for(size_t threadIdx_x = 0; threadIdx_x < BLOCK_SIZE; threadIdx_x++) {
      int set = threadIdx_x / radix;
      for (int i = blockIdx_x * BLOCK_SIZE + threadIdx_x; i < (N / 8 * num_prime); i += BLOCK_SIZE * GRID_SIZE) {

        int t = N / 2 / m;
        // prime idx
        int np_idx = num_prime - 1 - (i / (N / 8)) + start_prime_idx;
        // index in N/2 range
        int N_idx = i % (N / 8);
        // i'th block
        int m_idx = N_idx / (t / 4);
        int t_idx = N_idx % (t / 4);
        // base address
        uint64_t* a_np = op + np_idx * N;
        const uint64_t* prime_table = primes;
        uint64_t prime = prime_table[np_idx];
        int N_init = 2 * m_idx * t + t_idx;
        for (int j = 0; j < 8; j++) {
          local[threadIdx_x][j] = *(a_np + N_init + t / 4 * j);
        }
        int tw_idx = m + m_idx;
        const uint64_t* W = base_inv + N * np_idx;
        const uint64_t* W_ = base_inv_ + N * np_idx;


        for (int j = 0; j < 8; j++) {
          local[threadIdx_x][j] = temp[set * 8 * radix + t_idx + t / 4 * j];
          for (int k = 0; k < 3; k++) {
            if (local[threadIdx_x][j] >= prime)
              local[threadIdx_x][j] -= prime;
          }
        }
        for (int j = 0; j < 8; j++) {
          *(a_np + N_init + t / 4 * j) = local[threadIdx_x][j];
        }
      }
    }
  }
}

void Ntt8PointPerThreadPhase1ExcludeSomeRange(
    uint64_t* op,
    const int m,
    const int num_prime,
    const int N,
    const int start_prime_idx,
    const int excluded_range_start,
    const int excluded_range_end,
    const int curr_limbs,
    const int gap,
    const int pad,
    const int radix,
    const uint64_t* base_inv,
    const uint64_t* base_inv_,
    const uint64_t* primes,
    size_t GRID_SIZE,
    size_t BLOCK_SIZE,
    size_t SHARED_SIZE) {

  for(size_t blockIdx_x = 0; blockIdx_x < GRID_SIZE; blockIdx_x++) {
    std::vector<uint64_t> temp_vec(SHARED_SIZE, 0);
    uint64_t* temp = temp_vec.data();
    std::vector<std::array<uint64_t, 8>> local(BLOCK_SIZE, {0});
    std::vector<uint64_t> tail(BLOCK_SIZE, 0);
    for(size_t threadIdx_x = 0; threadIdx_x < BLOCK_SIZE; threadIdx_x++) {
      int Warp_t = threadIdx_x % pad;
      int WarpID = threadIdx_x / pad;
      for (int i = blockIdx_x * BLOCK_SIZE + threadIdx_x; i < (N / 8 * num_prime); i += BLOCK_SIZE * GRID_SIZE) {
      
        int t = N / 2 / m;
        // prime idx
        int np_idx = i / (N / 8) + start_prime_idx;
        if (np_idx >= excluded_range_start && np_idx < excluded_range_end)
          continue;
        int prime_idx = np_idx + ((np_idx >= 0 && np_idx < curr_limbs) ? 0 : gap);
        // index in N/2 range
        int N_idx = i % (N / 8);
        // i'th block
        int m_idx = N_idx / (t / 4);
        int t_idx = N_idx % (t / 4);
        // base address
        uint64_t* a_np = op + np_idx * N;
        const uint64_t* prime_table = primes;
        const uint64_t* W = base_inv + N * prime_idx;
        const uint64_t* W_ = base_inv_ + N * prime_idx;
        uint64_t prime = prime_table[prime_idx];
        int N_init = 2 * m_idx * t + t / 4 / radix * WarpID + Warp_t +
            pad * (t_idx / (radix * pad));
        for (int j = 0; j < 8; j++) {
          local[threadIdx_x][j] = *(a_np + N_init + t / 4 * j);
        }
      }
    }
    __syncthreads();
    for(size_t threadIdx_x = 0; threadIdx_x < BLOCK_SIZE; threadIdx_x++) {
      int Warp_t = threadIdx_x % pad;
      int WarpID = threadIdx_x / pad;
      for (int i = blockIdx_x * BLOCK_SIZE + threadIdx_x; i < (N / 8 * num_prime); i += BLOCK_SIZE * GRID_SIZE) {
      
        int t = N / 2 / m;
        // prime idx
        int np_idx = i / (N / 8) + start_prime_idx;
        if (np_idx >= excluded_range_start && np_idx < excluded_range_end)
          continue;
        int prime_idx = np_idx + ((np_idx >= 0 && np_idx < curr_limbs) ? 0 : gap);
        // index in N/2 range
        int N_idx = i % (N / 8);
        // i'th block
        int m_idx = N_idx / (t / 4);
        int t_idx = N_idx % (t / 4);
        // base address
        uint64_t* a_np = op + np_idx * N;
        const uint64_t* prime_table = primes;
        const uint64_t* W = base_inv + N * prime_idx;
        const uint64_t* W_ = base_inv_ + N * prime_idx;
        uint64_t prime = prime_table[prime_idx];
        int N_init = 2 * m_idx * t + t / 4 / radix * WarpID + Warp_t +
            pad * (t_idx / (radix * pad));

        int eradix = 8 * radix;
        int tw_idx = m + m_idx;
        for (int j = 0; j < 4; j++) {
          butt_ntt_local(local[threadIdx_x][j], local[threadIdx_x][j + 4], W[tw_idx], W_[tw_idx], prime);
        }
        for (int j = 0; j < 2; j++) {
          butt_ntt_local(
              local[threadIdx_x][4 * j],
              local[threadIdx_x][4 * j + 2],
              W[2 * tw_idx + j],
              W_[2 * tw_idx + j],
              prime);
          butt_ntt_local(
              local[threadIdx_x][4 * j + 1],
              local[threadIdx_x][4 * j + 3],
              W[2 * tw_idx + j],
              W_[2 * tw_idx + j],
              prime);
        }
        for (int j = 0; j < 4; j++) {
          butt_ntt_local(
              local[threadIdx_x][2 * j],
              local[threadIdx_x][2 * j + 1],
              W[4 * tw_idx + j],
              W_[4 * tw_idx + j],
              prime);
        }
        for (int j = 0; j < 8; j++) {
          temp[Warp_t * (eradix + pad) + WarpID + radix * j] = local[threadIdx_x][j];
        }
      }
    }
    __syncthreads();

    for (int j = 8, k = radix / 2; j < radix + 1; j *= 8, k >>= 3) {
      for(size_t threadIdx_x = 0; threadIdx_x < BLOCK_SIZE; threadIdx_x++) {
      int Warp_t = threadIdx_x % pad;
      int WarpID = threadIdx_x / pad;
        for (int i = blockIdx_x * BLOCK_SIZE + threadIdx_x; i < (N / 8 * num_prime); i += BLOCK_SIZE * GRID_SIZE) {
        
          int t = N / 2 / m;
          // prime idx
          int np_idx = i / (N / 8) + start_prime_idx;
          if (np_idx >= excluded_range_start && np_idx < excluded_range_end)
            continue;
          int prime_idx = np_idx + ((np_idx >= 0 && np_idx < curr_limbs) ? 0 : gap);
          // index in N/2 range
          int N_idx = i % (N / 8);
          // i'th block
          int m_idx = N_idx / (t / 4);
          int t_idx = N_idx % (t / 4);
          // base address
          uint64_t* a_np = op + np_idx * N;
          const uint64_t* prime_table = primes;
          const uint64_t* W = base_inv + N * prime_idx;
          const uint64_t* W_ = base_inv_ + N * prime_idx;
          uint64_t prime = prime_table[prime_idx];
          int N_init = 2 * m_idx * t + t / 4 / radix * WarpID + Warp_t +
              pad * (t_idx / (radix * pad));

          int eradix = 8 * radix;
          int tw_idx = m + m_idx;

          int m_idx2 = WarpID / (k / 4);
          int t_idx2 = WarpID % (k / 4);
          for (int l = 0; l < 8; l++) {
            local[threadIdx_x][l] = temp
                [(eradix + pad) * Warp_t + 2 * m_idx2 * k + t_idx2 + (k / 4) * l];
          }
          int tw_idx2 = j * tw_idx + m_idx2;
          for (int j2 = 0; j2 < 4; j2++) {
            butt_ntt_local(
                local[threadIdx_x][j2], local[threadIdx_x][j2 + 4], W[tw_idx2], W_[tw_idx2], prime);
          }
          for (int j2 = 0; j2 < 2; j2++) {
            butt_ntt_local(
                local[threadIdx_x][4 * j2],
                local[threadIdx_x][4 * j2 + 2],
                W[2 * tw_idx2 + j2],
                W_[2 * tw_idx2 + j2],
                prime);
            butt_ntt_local(
                local[threadIdx_x][4 * j2 + 1],
                local[threadIdx_x][4 * j2 + 3],
                W[2 * tw_idx2 + j2],
                W_[2 * tw_idx2 + j2],
                prime);
          }
          for (int j2 = 0; j2 < 4; j2++) {
            butt_ntt_local(
                local[threadIdx_x][2 * j2],
                local[threadIdx_x][2 * j2 + 1],
                W[4 * tw_idx2 + j2],
                W_[4 * tw_idx2 + j2],
                prime);
          }

          for (int l = 0; l < 8; l++) {
            temp[(eradix + pad) * Warp_t + 2 * m_idx2 * k + t_idx2 + (k / 4) * l] =
                local[threadIdx_x][l];
          }
          if (j == radix / 2)
            tail[threadIdx_x] = 1;
          if (j == radix / 4)
            tail[threadIdx_x] = 2;
        }
      }
      __syncthreads();
    }
    for(size_t threadIdx_x = 0; threadIdx_x < BLOCK_SIZE; threadIdx_x++) {
      int Warp_t = threadIdx_x % pad;
      int WarpID = threadIdx_x / pad;
      for (int i = blockIdx_x * BLOCK_SIZE + threadIdx_x; i < (N / 8 * num_prime); i += BLOCK_SIZE * GRID_SIZE) {
      
        int t = N / 2 / m;
        // prime idx
        int np_idx = i / (N / 8) + start_prime_idx;
        if (np_idx >= excluded_range_start && np_idx < excluded_range_end)
          continue;
        int prime_idx = np_idx + ((np_idx >= 0 && np_idx < curr_limbs) ? 0 : gap);
        // index in N/2 range
        int N_idx = i % (N / 8);
        // i'th block
        int m_idx = N_idx / (t / 4);
        int t_idx = N_idx % (t / 4);
        // base address
        uint64_t* a_np = op + np_idx * N;
        const uint64_t* prime_table = primes;
        const uint64_t* W = base_inv + N * prime_idx;
        const uint64_t* W_ = base_inv_ + N * prime_idx;
        uint64_t prime = prime_table[prime_idx];
        int N_init = 2 * m_idx * t + t / 4 / radix * WarpID + Warp_t +
            pad * (t_idx / (radix * pad));
    
        int eradix = 8 * radix;
        int tw_idx = m + m_idx;

        if (radix < 8)
          tail[threadIdx_x] = (radix == 4) ? 2 : 1;
        if (tail[threadIdx_x] == 1) {
          for (int l = 0; l < 8; l++) {
            local[threadIdx_x][l] = temp[(eradix + pad) * Warp_t + 8 * WarpID + l];
          }
          int tw_idx2 = (4 * radix) * tw_idx + 4 * WarpID;
          butt_ntt_local(local[threadIdx_x][0], local[threadIdx_x][1], W[tw_idx2], W_[tw_idx2], prime);
          butt_ntt_local(
              local[threadIdx_x][2], local[threadIdx_x][3], W[tw_idx2 + 1], W_[tw_idx2 + 1], prime);
          butt_ntt_local(
              local[threadIdx_x][4], local[threadIdx_x][5], W[tw_idx2 + 2], W_[tw_idx2 + 2], prime);
          butt_ntt_local(
              local[threadIdx_x][6], local[threadIdx_x][7], W[tw_idx2 + 3], W_[tw_idx2 + 3], prime);
          for (int l = 0; l < 8; l++) {
            temp[(eradix + pad) * Warp_t + 8 * WarpID + l] = local[threadIdx_x][l];
          }
        } else if (tail[threadIdx_x] == 2) {
          for (int l = 0; l < 8; l++) {
            local[threadIdx_x][l] = temp[(eradix + pad) * Warp_t + 8 * WarpID + l];
          }
          int tw_idx2 = 2 * radix * tw_idx + 2 * WarpID;
          butt_ntt_local(local[threadIdx_x][0], local[threadIdx_x][2], W[tw_idx2], W_[tw_idx2], prime);
          butt_ntt_local(local[threadIdx_x][1], local[threadIdx_x][3], W[tw_idx2], W_[tw_idx2], prime);
          butt_ntt_local(
              local[threadIdx_x][4], local[threadIdx_x][6], W[tw_idx2 + 1], W_[tw_idx2 + 1], prime);
          butt_ntt_local(
              local[threadIdx_x][5], local[threadIdx_x][7], W[tw_idx2 + 1], W_[tw_idx2 + 1], prime);
          butt_ntt_local(
              local[threadIdx_x][0], local[threadIdx_x][1], W[2 * tw_idx2], W_[2 * tw_idx2], prime);
          butt_ntt_local(
              local[threadIdx_x][2], local[threadIdx_x][3], W[2 * tw_idx2 + 1], W_[2 * tw_idx2 + 1], prime);
          butt_ntt_local(
              local[threadIdx_x][4], local[threadIdx_x][5], W[2 * tw_idx2 + 2], W_[2 * tw_idx2 + 2], prime);
          butt_ntt_local(
              local[threadIdx_x][6], local[threadIdx_x][7], W[2 * tw_idx2 + 3], W_[2 * tw_idx2 + 3], prime);
          for (int l = 0; l < 8; l++) {
            temp[(eradix + pad) * Warp_t + 8 * WarpID + l] = local[threadIdx_x][l];
          }
        }
      }
    }
    __syncthreads();
    for(size_t threadIdx_x = 0; threadIdx_x < BLOCK_SIZE; threadIdx_x++) {
      int Warp_t = threadIdx_x % pad;
      int WarpID = threadIdx_x / pad;
      for (int i = blockIdx_x * BLOCK_SIZE + threadIdx_x; i < (N / 8 * num_prime); i += BLOCK_SIZE * GRID_SIZE) {
      
        int t = N / 2 / m;
        // prime idx
        int np_idx = i / (N / 8) + start_prime_idx;
        if (np_idx >= excluded_range_start && np_idx < excluded_range_end)
          continue;
        int prime_idx = np_idx + ((np_idx >= 0 && np_idx < curr_limbs) ? 0 : gap);
        // index in N/2 range
        int N_idx = i % (N / 8);
        // i'th block
        int m_idx = N_idx / (t / 4);
        int t_idx = N_idx % (t / 4);
        // base address
        uint64_t* a_np = op + np_idx * N;
        const uint64_t* prime_table = primes;
        const uint64_t* W = base_inv + N * prime_idx;
        const uint64_t* W_ = base_inv_ + N * prime_idx;
        uint64_t prime = prime_table[prime_idx];
        int N_init = 2 * m_idx * t + t / 4 / radix * WarpID + Warp_t +
            pad * (t_idx / (radix * pad));

        int eradix = 8 * radix;
        int tw_idx = m + m_idx;

        for (int j = 0; j < 8; j++) {
          local[threadIdx_x][j] = temp[Warp_t * (eradix + pad) + WarpID + radix * j];
        }
        for (int j = 0; j < 8; j++) {
          *(a_np + N_init + t / 4 * j) = local[threadIdx_x][j];
        }
      }
    }
  }
}

void Ntt8PointPerThreadPhase2ExcludeSomeRange(
    uint64_t* op,
    const int m,
    const int num_prime,
    const int N,
    const int start_prime_idx,
    const int excluded_range_start,
    const int excluded_range_end,
    const int curr_limbs,
    const int gap,
    const int radix,
    const uint64_t* base_inv,
    const uint64_t* base_inv_,
    const uint64_t* primes,
    size_t GRID_SIZE,
    size_t BLOCK_SIZE,
    size_t SHARED_SIZE) {

  for(size_t blockIdx_x = 0; blockIdx_x < GRID_SIZE; blockIdx_x++) {
    std::vector<uint64_t> temp_vec(SHARED_SIZE, 0);
    uint64_t* temp = temp_vec.data();
    std::vector<std::array<uint64_t, 8>> local(BLOCK_SIZE, {0});
    std::vector<uint64_t> tail(BLOCK_SIZE, 0);
    for(size_t threadIdx_x = 0; threadIdx_x < BLOCK_SIZE; threadIdx_x++) {
      int set = threadIdx_x / radix;
      for (int i = blockIdx_x * BLOCK_SIZE + threadIdx_x; i < (N / 8 * num_prime); i += BLOCK_SIZE * GRID_SIZE) {

        int t = N / 2 / m;
        // prime idx
        int np_idx = num_prime - 1 - (i / (N / 8)) + start_prime_idx;
        if (np_idx >= excluded_range_start && np_idx < excluded_range_end)
          continue;
        int prime_idx = np_idx + ((np_idx >= 0 && np_idx < curr_limbs) ? 0 : gap);
        // index in N/2 range
        int N_idx = i % (N / 8);
        // i'th block
        int m_idx = N_idx / (t / 4);
        int t_idx = N_idx % (t / 4);
        // base address
        uint64_t* a_np = op + np_idx * N;
        const uint64_t* prime_table = primes;
        uint64_t prime = prime_table[prime_idx];
        int N_init = 2 * m_idx * t + t_idx;

        for (int j = 0; j < 8; j++) {
          local[threadIdx_x][j] = *(a_np + N_init + t / 4 * j);
        }
        int tw_idx = m + m_idx;
        const uint64_t* W = base_inv + N * prime_idx;
        const uint64_t* W_ = base_inv_ + N * prime_idx;
        for (int j = 0; j < 4; j++) {
          butt_ntt_local(local[threadIdx_x][j], local[threadIdx_x][j + 4], W[tw_idx], W_[tw_idx], prime);
        }
        for (int j = 0; j < 2; j++) {
          butt_ntt_local(
              local[threadIdx_x][4 * j],
              local[threadIdx_x][4 * j + 2],
              W[2 * tw_idx + j],
              W_[2 * tw_idx + j],
              prime);
          butt_ntt_local(
              local[threadIdx_x][4 * j + 1],
              local[threadIdx_x][4 * j + 3],
              W[2 * tw_idx + j],
              W_[2 * tw_idx + j],
              prime);
        }
        for (int j = 0; j < 4; j++) {
          butt_ntt_local(
              local[threadIdx_x][2 * j],
              local[threadIdx_x][2 * j + 1],
              W[4 * tw_idx + j],
              W_[4 * tw_idx + j],
              prime);
        }
        for (int j = 0; j < 8; j++) {
          temp[set * 8 * radix + t_idx + t / 4 * j] = local[threadIdx_x][j];
        }
      }
    }
    __syncthreads();
    int t = N / 2 / m;
    for (int j = 8, k = t / 8; j < t / 4 + 1; j *= 8, k >>= 3) {
      for(size_t threadIdx_x = 0; threadIdx_x < BLOCK_SIZE; threadIdx_x++) {
        int set = threadIdx_x / radix;
        for (int i = blockIdx_x * BLOCK_SIZE + threadIdx_x; i < (N / 8 * num_prime); i += BLOCK_SIZE * GRID_SIZE) {

          int t = N / 2 / m;
          // prime idx
          int np_idx = num_prime - 1 - (i / (N / 8)) + start_prime_idx;
          if (np_idx >= excluded_range_start && np_idx < excluded_range_end)
            continue;
          int prime_idx = np_idx + ((np_idx >= 0 && np_idx < curr_limbs) ? 0 : gap);
          // index in N/2 range
          int N_idx = i % (N / 8);
          // i'th block
          int m_idx = N_idx / (t / 4);
          int t_idx = N_idx % (t / 4);
          // base address
          uint64_t* a_np = op + np_idx * N;
          const uint64_t* prime_table = primes;
          uint64_t prime = prime_table[prime_idx];
          int N_init = 2 * m_idx * t + t_idx;

          int tw_idx = m + m_idx;
          const uint64_t* W = base_inv + N * prime_idx;
          const uint64_t* W_ = base_inv_ + N * prime_idx;

          int m_idx2 = t_idx / (k / 4);
          int t_idx2 = t_idx % (k / 4);
          for (int l = 0; l < 8; l++) {
            local[threadIdx_x][l] =
                temp[set * 8 * radix + 2 * m_idx2 * k + t_idx2 + (k / 4) * l];
          }
          int tw_idx2 = j * tw_idx + m_idx2;
          for (int j2 = 0; j2 < 4; j2++) {
            butt_ntt_local(
                local[threadIdx_x][j2], local[threadIdx_x][j2 + 4], W[tw_idx2], W_[tw_idx2], prime);
          }
          for (int j2 = 0; j2 < 2; j2++) {
            butt_ntt_local(
                local[threadIdx_x][4 * j2],
                local[threadIdx_x][4 * j2 + 2],
                W[2 * tw_idx2 + j2],
                W_[2 * tw_idx2 + j2],
                prime);
            butt_ntt_local(
                local[threadIdx_x][4 * j2 + 1],
                local[threadIdx_x][4 * j2 + 3],
                W[2 * tw_idx2 + j2],
                W_[2 * tw_idx2 + j2],
                prime);
          }
          for (int j2 = 0; j2 < 4; j2++) {
            butt_ntt_local(
                local[threadIdx_x][2 * j2],
                local[threadIdx_x][2 * j2 + 1],
                W[4 * tw_idx2 + j2],
                W_[4 * tw_idx2 + j2],
                prime);
          }

          for (int l = 0; l < 8; l++) {
            temp[set * 8 * radix + 2 * m_idx2 * k + t_idx2 + (k / 4) * l] =
                local[threadIdx_x][l];
          }
          if (j == t / 8)
            tail[threadIdx_x] = 1;
          if (j == t / 16)
            tail[threadIdx_x] = 2;
        }
      }
      __syncthreads();
    }
    for(size_t threadIdx_x = 0; threadIdx_x < BLOCK_SIZE; threadIdx_x++) {
      int set = threadIdx_x / radix;
      for (int i = blockIdx_x * BLOCK_SIZE + threadIdx_x; i < (N / 8 * num_prime); i += BLOCK_SIZE * GRID_SIZE) {

        int t = N / 2 / m;
        // prime idx
        int np_idx = num_prime - 1 - (i / (N / 8)) + start_prime_idx;
        if (np_idx >= excluded_range_start && np_idx < excluded_range_end)
          continue;
        int prime_idx = np_idx + ((np_idx >= 0 && np_idx < curr_limbs) ? 0 : gap);
        // index in N/2 range
        int N_idx = i % (N / 8);
        // i'th block
        int m_idx = N_idx / (t / 4);
        int t_idx = N_idx % (t / 4);
        // base address
        uint64_t* a_np = op + np_idx * N;
        const uint64_t* prime_table = primes;
        uint64_t prime = prime_table[prime_idx];
        int N_init = 2 * m_idx * t + t_idx;

        int tw_idx = m + m_idx;
        const uint64_t* W = base_inv + N * prime_idx;
        const uint64_t* W_ = base_inv_ + N * prime_idx;

        if (tail[threadIdx_x] == 1) {
          for (int l = 0; l < 8; l++) {
            local[threadIdx_x][l] = temp[set * 8 * radix + 8 * t_idx + l];
          }
          int tw_idx2 = t * tw_idx + 4 * t_idx;
          butt_ntt_local(local[threadIdx_x][0], local[threadIdx_x][1], W[tw_idx2], W_[tw_idx2], prime);
          butt_ntt_local(
              local[threadIdx_x][2], local[threadIdx_x][3], W[tw_idx2 + 1], W_[tw_idx2 + 1], prime);
          butt_ntt_local(
              local[threadIdx_x][4], local[threadIdx_x][5], W[tw_idx2 + 2], W_[tw_idx2 + 2], prime);
          butt_ntt_local(
              local[threadIdx_x][6], local[threadIdx_x][7], W[tw_idx2 + 3], W_[tw_idx2 + 3], prime);
          for (int l = 0; l < 8; l++) {
            temp[set * 8 * radix + 8 * t_idx + l] = local[threadIdx_x][l];
          }
        } else if (tail[threadIdx_x] == 2) {
          for (int l = 0; l < 8; l++) {
            local[threadIdx_x][l] = temp[set * 8 * radix + 8 * t_idx + l];
          }
          int tw_idx2 = (t / 2) * tw_idx + 2 * t_idx;
          butt_ntt_local(local[threadIdx_x][0], local[threadIdx_x][2], W[tw_idx2], W_[tw_idx2], prime);
          butt_ntt_local(local[threadIdx_x][1], local[threadIdx_x][3], W[tw_idx2], W_[tw_idx2], prime);
          butt_ntt_local(
              local[threadIdx_x][4], local[threadIdx_x][6], W[tw_idx2 + 1], W_[tw_idx2 + 1], prime);
          butt_ntt_local(
              local[threadIdx_x][5], local[threadIdx_x][7], W[tw_idx2 + 1], W_[tw_idx2 + 1], prime);
          butt_ntt_local(
              local[threadIdx_x][0], local[threadIdx_x][1], W[2 * tw_idx2], W_[2 * tw_idx2], prime);
          butt_ntt_local(
              local[threadIdx_x][2], local[threadIdx_x][3], W[2 * tw_idx2 + 1], W_[2 * tw_idx2 + 1], prime);
          butt_ntt_local(
              local[threadIdx_x][4], local[threadIdx_x][5], W[2 * tw_idx2 + 2], W_[2 * tw_idx2 + 2], prime);
          butt_ntt_local(
              local[threadIdx_x][6], local[threadIdx_x][7], W[2 * tw_idx2 + 3], W_[2 * tw_idx2 + 3], prime);
          for (int l = 0; l < 8; l++) {
            temp[set * 8 * radix + 8 * t_idx + l] = local[threadIdx_x][l];
          }
        }
      }
    }
    __syncthreads();
    for(size_t threadIdx_x = 0; threadIdx_x < BLOCK_SIZE; threadIdx_x++) {
      int set = threadIdx_x / radix;
      for (int i = blockIdx_x * BLOCK_SIZE + threadIdx_x; i < (N / 8 * num_prime); i += BLOCK_SIZE * GRID_SIZE) {

        int t = N / 2 / m;
        // prime idx
        int np_idx = num_prime - 1 - (i / (N / 8)) + start_prime_idx;
        if (np_idx >= excluded_range_start && np_idx < excluded_range_end)
          continue;
        int prime_idx = np_idx + ((np_idx >= 0 && np_idx < curr_limbs) ? 0 : gap);
        // index in N/2 range
        int N_idx = i % (N / 8);
        // i'th block
        int m_idx = N_idx / (t / 4);
        int t_idx = N_idx % (t / 4);
        // base address
        uint64_t* a_np = op + np_idx * N;
        const uint64_t* prime_table = primes;
        uint64_t prime = prime_table[prime_idx];
        int N_init = 2 * m_idx * t + t_idx;

        for (int j = 0; j < 8; j++) {
          local[threadIdx_x][j] = temp[set * 8 * radix + t_idx + t / 4 * j];
          for (int k = 0; k < 3; k++) {
            if (local[threadIdx_x][j] >= prime)
              local[threadIdx_x][j] -= prime;
          }
        }
        for (int j = 0; j < 8; j++) {
          *(a_np + N_init + t / 4 * j) = local[threadIdx_x][j];
        }
      }
    }
  }
}

} // namespace fhe

namespace at::native {

// void iNTT_impl(
//     uint64_t* op_ptr,
//     int64_t start_prime_idx,
//     int64_t batch,
//     int64_t curr_limbs,
//     int64_t level,
//     int64_t param_degree,
//     const Tensor& inverse_power_of_roots_div_two,
//     const Tensor& param_primes,
//     const Tensor& inverse_scaled_power_of_roots_div_two) {
//   AT_DISPATCH_V2(
//       kUInt64,
//       "iNTT",
//       AT_WRAP([&]() {

//         //   void butt_intt_local(
//     //     uint64_t& x,
//     //     uint64_t& y,
//     //     const uint64_t& w,
//     //     const uint64_t& w_,
//     //     const uint64_t& p) {
//     //   const uint64_t two_p = 2 * p;
//     //   const uint64_t T = two_p - y + x;
//     //   uint64_t new_x = x + y;
//     //   if (new_x >= two_p)
//     //     new_x -= two_p;
//     //   if (T & 1)
//     //     new_x += p;
//     //   x = (new_x >> 1);
//     //   y = mul_and_reduce_shoup(T, w, w_, p);
//     // }
//         size_t gridDim(2048);
//         size_t blockDim(256);
//         const int per_thread_ntt_size = 8;
//         const int first_stage_radix_size = 256;
//         const int second_radix_size = param_degree / first_stage_radix_size;
//         const int pad = 4;
//         const int per_thread_storage =
//             blockDim * per_thread_ntt_size * sizeof(uint64_t);
//         auto inverse_power_of_roots_div_two_ptr = reinterpret_cast<uint64_t*>(
//             inverse_power_of_roots_div_two.data_ptr<uint64_t>());
//         auto param_primes_ptr =
//             reinterpret_cast<uint64_t*>(param_primes.data_ptr<uint64_t>());
//         auto inverse_scaled_power_of_roots_div_two_ptr =
//             reinterpret_cast<uint64_t*>(
//                 inverse_scaled_power_of_roots_div_two.data_ptr<uint64_t>());
//         int gap = level - curr_limbs;

//         fhe::Intt8PointPerThreadPhase2OoP(
//             op_ptr,
//             first_stage_radix_size,
//             batch,
//             param_degree,
//             start_prime_idx,
//             curr_limbs,
//             gap,
//             second_radix_size / per_thread_ntt_size,
//             inverse_power_of_roots_div_two_ptr,
//             inverse_scaled_power_of_roots_div_two_ptr,
//             param_primes_ptr,
//             op_ptr,
//             gridDim,
//             blockDim,
//             per_thread_storage / sizeof(uint64_t));
//         fhe::Intt8PointPerThreadPhase1OoP(
//             op_ptr,
//             1,
//             batch,
//             param_degree,
//             start_prime_idx,
//             curr_limbs,
//             gap,
//             pad,
//             first_stage_radix_size / 8,
//             inverse_power_of_roots_div_two_ptr,
//             inverse_scaled_power_of_roots_div_two_ptr,
//             param_primes_ptr,
//             op_ptr,
//             gridDim,
//             (first_stage_radix_size / 8) * pad,
//             (first_stage_radix_size + pad + 1) * pad
//         );
//       }),
//       kUInt64);
// }
static std::tuple<__int128_t, __int128_t, __int128_t> extended_gcd(__int128_t a, __int128_t b) {
  if (b == 0) {
      return {a, 1, 0};
  }
  auto [g, x1, y1] = extended_gcd(b, a % b);
  __int128_t x = y1;
  __int128_t y = x1 - (a / b) * y1;
  return {g, x, y};
}

uint64_t modInverse(uint64_t n, uint64_t modulus) {
  auto [g, x, y] = extended_gcd(n, modulus);
  if (g != 1) {
      throw std::invalid_argument("模逆元不存在");
  }
  x %= modulus;
  if (x < 0) {
      x += modulus;
  }
  return static_cast<uint64_t>(x);
}




void iNTT_impl(
  uint64_t* op_ptr,
  int64_t start_prime_idx,
  int64_t batch,
  int64_t curr_limbs,
  int64_t level,
  int64_t param_degree,
  const Tensor& inverse_power_of_roots_div_two,
  const Tensor& param_primes,
  const Tensor& inverse_scaled_power_of_roots_div_two) {
AT_DISPATCH_V2(
    kUInt64,
    "iNTT",
    AT_WRAP([&]() {
      const uint64_t n=param_degree;
      auto inverse_power_of_roots_div_two_ptr = reinterpret_cast<uint64_t*>(inverse_power_of_roots_div_two.data_ptr<uint64_t>());
      auto param_primes_ptr =reinterpret_cast<uint64_t*>(param_primes.data_ptr<uint64_t>());
      auto inverse_scaled_power_of_roots_div_two_ptr =reinterpret_cast<uint64_t*>(inverse_scaled_power_of_roots_div_two.data_ptr<uint64_t>());
      int gap = level - curr_limbs;
      //openfhe version
    //   for(int bach=0;bach<batch;++bach)
    // {
    //   uint64_t primeidx=start_prime_idx+bach;
    //   uint64_t prime_idx =primeidx + ((primeidx >= 0 && primeidx < curr_limbs) ? 0 : gap);
    //   uint64_t modulus=param_primes_ptr[prime_idx];
    //   uint64_t base_prime_idx=prime_idx*param_degree;
    //   uint64_t base=primeidx*param_degree;
    //   uint64_t inv=modInverse(n,modulus);
    // for (uint32_t m=n >> 1, t=1, logt=1; m >=1; m >>= 1, t <<= 1, ++logt) {
    //   for (uint32_t i=0; i < m; ++i) {
    //     auto omega=inverse_power_of_roots_div_two_ptr[i + m+base_prime_idx];
    //       auto preconOmega=inverse_scaled_power_of_roots_div_two_ptr[i + m+base_prime_idx];
    //       omega=static_cast<__uint128_t>(omega)*static_cast<__uint128_t>(2)% static_cast<__uint128_t>(modulus);
    //       for (uint32_t j1=i << logt, j2=j1 + t; j1 < j2; ++j1) {
    //           auto loVal=op_ptr[j1 + 0+base];
    //           auto hiVal=op_ptr[j1 + t+base];   
    //     auto omegaFactor=loVal;
    //           if (omegaFactor < hiVal)
    //               omegaFactor += modulus;
    //           omegaFactor -= hiVal;
    //           loVal += hiVal;
    //           if (loVal >= modulus)
    //               loVal -= modulus;
    //           omegaFactor=(static_cast<__uint128_t>(omegaFactor)  * static_cast<__uint128_t>(omega) )% static_cast<__uint128_t>(modulus);
    //          op_ptr[j1 + 0+base]=loVal;
    //          op_ptr[j1 + t+base]=omegaFactor;
    //         }}}
    //             for (uint32_t i = 0; i < n; ++i) 
    //             {
                  
    //               op_ptr[i+base] =(static_cast<__uint128_t>(op_ptr[i+base] ) * static_cast<__uint128_t>(inv)) % static_cast<__uint128_t>(modulus);
    //             }
    //           }
//gpufhe version
              for(int bach=0;bach<batch;++bach)
              {
                uint64_t primeidx=start_prime_idx+bach;
                uint64_t prime_idx =primeidx + ((primeidx >= 0 && primeidx < curr_limbs) ? 0 : gap);
                uint64_t modulus=param_primes_ptr[prime_idx];
                uint64_t base_prime_idx=prime_idx*param_degree;
                uint64_t base=primeidx*param_degree;
                // uint64_t inv=modInverse(n,modulus);

                for (uint32_t m = n >> 1, t = 1, logt = 1; m > 1;
                     m >>= 1, t <<= 1, ++logt) {
                  for (uint32_t i = 0; i < m; ++i) {
                    auto omega = inverse_power_of_roots_div_two_ptr
                        [i + m + base_prime_idx];
                    auto preconOmega = inverse_scaled_power_of_roots_div_two_ptr
                        [i + m + base_prime_idx];
                    for (uint32_t j1 = i << logt, j2 = j1 + t; j1 < j2; ++j1) {
                      auto loVal = op_ptr[j1 + 0 + base];
                      auto hiVal = op_ptr[j1 + t + base];
                      fhe::butt_intt_local(
                          loVal, hiVal, omega, preconOmega, modulus);
                      op_ptr[j1 + 0 + base] = loVal;
                      op_ptr[j1 + t + base] = hiVal;
                    }
                  }
                }

                auto omega =
                    inverse_power_of_roots_div_two_ptr[1 + base_prime_idx];
                auto preconOmega = inverse_scaled_power_of_roots_div_two_ptr
                    [1 + base_prime_idx];
                uint32_t j2 = n >> 1;
                for (uint32_t j1 = 0; j1 < j2; ++j1) {
                  auto loVal = (op_ptr)[j1 + base];
                  auto hiVal = (op_ptr)[j1 + j2 + base];
                  fhe::butt_intt_local(
                      loVal, hiVal, omega, preconOmega, modulus);
                  for (int i = 0; i < 8; i++) {
                    if (loVal > modulus) {
                      loVal -= modulus;
                    }
                    if (hiVal > modulus) {
                      hiVal -= modulus;
                    }
                  }
                  (op_ptr)[j1 + base] = loVal;
                  (op_ptr)[j1 + j2 + base] = hiVal;
                }
              }
    }),
    kUInt64);
}


// void NTT_impl(
//   uint64_t* op_ptr,
//   int64_t start_prime_idx,
//   int64_t batch,
//   int64_t param_degree,
//   const Tensor& param_power_of_roots_shoup,
//   const Tensor& param_primes,
//   const Tensor& param_power_of_roots) {
// size_t gridDim(2048);
// size_t blockDim(256);
// const int per_thread_ntt_size = 8;
// const int first_stage_radix_size = 256;
// const int second_radix_size = param_degree / first_stage_radix_size;
// const int pad = 4;
// const int per_thread_storage =
//     blockDim * per_thread_ntt_size * sizeof(uint64_t);
// AT_DISPATCH_V2(
//     kUInt64,
//     "NTT",
//     AT_WRAP([&]() {
//       auto param_power_of_roots_shoup_ptr = reinterpret_cast<uint64_t*>(
//           param_power_of_roots_shoup.data_ptr<uint64_t>());
//       auto param_primes_ptr =
//           reinterpret_cast<uint64_t*>(param_primes.data_ptr<uint64_t>());
//       auto param_power_of_roots_ptr = reinterpret_cast<uint64_t*>(
//           param_power_of_roots.data_ptr<uint64_t>());
//       fhe::Ntt8PointPerThreadPhase1(
//           op_ptr,
//           1,
//           batch,
//           param_degree,
//           start_prime_idx,
//           pad,
//           first_stage_radix_size / per_thread_ntt_size,
//           param_power_of_roots_ptr,
//           param_power_of_roots_shoup_ptr,
//           param_primes_ptr,
//           gridDim,
//           (first_stage_radix_size / 8) * pad,
//           (first_stage_radix_size + pad + 1) * pad);
//       fhe::Ntt8PointPerThreadPhase2(
//           op_ptr,
//           first_stage_radix_size,
//           batch,
//           param_degree,
//           start_prime_idx,
//           second_radix_size / per_thread_ntt_size,
//           param_power_of_roots_ptr,
//           param_power_of_roots_shoup_ptr,
//           param_primes_ptr,
//           gridDim,
//           blockDim,
//           per_thread_storage / sizeof(uint64_t));
//     }),
//     kUInt64);
// }
int GetMSB(int64_t x) {
    if (x == 0) return -1; // No set bit, return -1

    int position = 0;
    while (x > 0) {
        x >>= 1;   // Shift right by 1 bit
        position++;  // Increment the position
    }
    return position ; // The MSB is 1 less than the number of shifts
}
void NTT_impl(
    uint64_t* op_ptr,
    int64_t start_prime_idx,
    int64_t batch,
    int64_t param_degree,
    const Tensor& param_power_of_roots_shoup,
    const Tensor& param_primes,
    const Tensor& param_power_of_roots) {
  AT_DISPATCH_V2(
      kUInt64,
      "NTT",
      AT_WRAP([&]() {
        auto param_power_of_roots_shoup_ptr = reinterpret_cast<uint64_t*>(
            param_power_of_roots_shoup
                .data_ptr<uint64_t>()); // preconrootOfUnityTable
        auto param_primes_ptr = reinterpret_cast<uint64_t*>(
            param_primes.data_ptr<uint64_t>()); // modulo
        auto param_power_of_roots_ptr = reinterpret_cast<uint64_t*>(
            param_power_of_roots.data_ptr<uint64_t>()); // rootOfUnityTable
        const int64_t n = param_degree >> 1;
        for (int bach = 0; bach < batch; ++bach) {
          auto modulus = param_primes_ptr[start_prime_idx + bach];
          auto primeidx = (start_prime_idx + bach);
          auto base = primeidx * param_degree;
          for (uint32_t m = 1, t = n, logt = GetMSB(t); m < n;
               m <<= 1, t >>= 1, --logt) {
            for (uint32_t i = 0; i < m; ++i) {
              auto omega = param_power_of_roots_ptr[i + m + base]; // S
              auto preconOmega = param_power_of_roots_shoup_ptr
                  [i + m + base]; // NEEDED IN COMPUTE F[j+t]*S MOD Q
              for (uint32_t j1 = (i << logt), j2 = j1 + t; j1 < j2; ++j1) {
                uint64_t a1 = (op_ptr)[j1 + 0 + base];
                uint64_t b1 = (op_ptr)[j1 + t + base];
                fhe::butt_ntt_local(a1, b1, omega, preconOmega, modulus);
                (op_ptr)[j1 + 0 + base] = a1;
                (op_ptr)[j1 + t + base] = b1;
              }
            }
          }
          for (uint32_t i = 0; i < (n << 1); i += 2) {
            auto omega = param_power_of_roots_ptr[(i >> 1) + n + base];
            auto preconOmega =
                param_power_of_roots_shoup_ptr[(i >> 1) + n + base];
            uint64_t a1 = (op_ptr)[i + 0 + base];
            uint64_t b1 = (op_ptr)[i + 1 + base];
            fhe::butt_ntt_local(a1, b1, omega, preconOmega, modulus);
            for (int a = 0; a < 3; a++) {
              if (b1 > modulus) {
                b1 -= modulus;
              }
              if (a1 > modulus) {
                a1 -= modulus;
              }
            }
            (op_ptr)[i + 0 + base] = a1;
            (op_ptr)[i + 1 + base] = b1;
          }
        }
      }),
      kUInt64);
}

} // end native namespace