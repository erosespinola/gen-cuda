#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <cuda.h>

#include <iostream>
using namespace std;

#include <thrust/sequence.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/constant_iterator.h>

#include "cuPrintf.cuh"
#include "cuPrintf.cu"

#define NUM_THREADS 256

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


typedef pair<float, float> city_t;
const float ratio = 10.0;

// Each gene has m chromosomes
void initialize_genes(thrust::host_vector<int> &genes, int m, int n)
{
    for (int i = 0; i < n; i++) 
    {
        // traversal sequence
        thrust::sequence(genes.begin() + (i * m), genes.begin() + (i * m) + m, 1, 1);

        // random numbers for shuffling
        // thrust::generate(back_genes.begin() + (i * m), back_genes.begin() + (i * m) + m, rand);

        for (int j = 0; j < m; j++)
        {
            int r = rand() % m, temp;

            temp = genes[(i * m) + j];
            genes[(i * m) + j] = genes[(i * m) + r];
            genes[(i * m) + r] = temp;
        }
    }
}

void initialize_cities(thrust::host_vector<city_t> &cities, int m, int n) 
{
    for (int i = 0; i < m; i++)
    {
        cities[i] = make_pair(ratio * cos(i / float(m) * 6.28), ratio * sin(i / float(m) * 6.28));
        // cout << cities[i].first << ", " << cities[i].second << endl;
    }
}

void print_genes(thrust::host_vector<int> &genes, int m, int n)
{
    for (int i = 0; i < n; i++) 
    {
        for (int j = 0; j < m; j++) 
        {
            printf("%d ", genes[i * m + j]);
        }

        printf("\n");
    }
}

__device__ float individual_fitness(city_t *cities, int *current_gene, int m)
{
    float p_x = cities[0].first,
          p_y = cities[0].second,
          current_fitness = 0.0,
          x, y;

    for (int i = 0; i < m; i++)
    {
        int current_city = current_gene[i];

        x = cities[current_city].first - p_x,
        y = cities[current_city].second - p_y;

        current_fitness += sqrt(x * x + y * y);

        p_x = cities[current_city].first;
        p_y = cities[current_city].second;
    }

    x = cities[0].first - p_x;
    y = cities[0].second - p_y;

    current_fitness += sqrt(x * x + y * y);

    return 1.0 / current_fitness;
}

__global__ void compute_fitness(float *fitness, city_t *cities, int *genes, int m, int n)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid >= n) 
    {
        return;
    }

    fitness[tid] = individual_fitness(cities, &genes[tid * m], m);
}

// Stochastic universal sampling selection operator
__global__ void selection(float *fitness, city_t *cities, int *genes, int m, int n)
{
    
}

__global__ void crossover(float *fitness, city_t *cities, int *genes, int m, int n)
{
    
}

// 
__global__ void mutation(float *fitness, city_t *cities, int *genes, int m, int n)
{

}

int main(int argc, char **argv)
{    
    // This takes a few seconds to initialize the runtime
    cudaDeviceSynchronize(); 

    srand(time(NULL));

    int m = 9,
        n = 2 << 4;

    // GPU genes data structure
    int *d_genes_raw, *d_back_genes_raw;
    float *d_fitness_raw;
    city_t *d_cities_raw;

    thrust::host_vector<int> genes(m * n);
    thrust::host_vector<int> back_genes(m * n);
    thrust::host_vector<city_t> cities(n);
    thrust::host_vector<float> fitness(n);

    thrust::device_vector<int> d_genes;
    thrust::device_vector<int> d_back_genes;
    thrust::device_vector<city_t> d_cities;
    thrust::device_vector<float> d_fitness;

    int blks = (n + NUM_THREADS - 1) / NUM_THREADS;

    initialize_genes(genes, m, n);
    initialize_cities(cities, m, n);

    // print_genes(genes, m, n);

    d_genes = genes;
    d_back_genes = back_genes;
    d_cities = cities;
    d_fitness = fitness;

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    for (int step = 0; step < 1; step++)
    {
        d_fitness_raw = thrust::raw_pointer_cast(&d_fitness[0]);
        d_genes_raw = thrust::raw_pointer_cast(&d_genes[0]);
        d_cities_raw = thrust::raw_pointer_cast(&d_cities[0]);
        
        compute_fitness <<< blks, NUM_THREADS >>> (d_fitness_raw, d_cities_raw, d_genes_raw, m, n);

        // Normalize fitness (sum of fitness = 1)
        float total_fitness = thrust::reduce(d_fitness.begin(), d_fitness.end());
        thrust::constant_iterator<float> normalization(total_fitness);
        thrust::transform(d_fitness.begin(), d_fitness.end(), normalization, d_fitness.begin(), thrust::divides<float>());

        // d_back_genes_raw = thrust::raw_pointer_cast(&d_back_genes[0]);

        thrust::copy(d_fitness.begin(), d_fitness.end(), fitness.begin());
        for (int i = 0; i < n; i++)
        {
            cout << fitness[i] << endl;
        }

        // bin_particles <<< blks, NUM_THREADS >>> (d_particles, d_bins, d_bin_sizes, n, m);
        cudaDeviceSynchronize();
    }

    return 0;
}
