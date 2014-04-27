#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <cuda.h>

#include <iostream>
#include <algorithm>
using namespace std;

#include <thrust/sequence.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/random.h>
#include <thrust/random/uniform_real_distribution.h>
#include <thrust/extrema.h>

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
const float CROSSOVER_RATE = 0.7;
const float MUTATION_RATE = 0.01;

// Each gene has m chromosomes
void initialize_genes(thrust::host_vector<int> &genes, int m, int n)
{
    for (int i = 0; i < n; i++) 
    {
        // traversal sequence
        thrust::sequence(genes.begin() + (i * m), genes.begin() + (i * m) + m, 1, 1);

        // random numbers for shuffling
        // thrust::generate(offspring.begin() + (i * m), offspring.begin() + (i * m) + m, rand);

        for (int j = 0; j < m; j++)
        {
            int r = rand() % m;

            swap(genes[(i * m) + j], genes[(i * m) + r]);

            // temp = genes[(i * m) + j];
            // genes[(i * m) + j] = genes[(i * m) + r];
            // genes[(i * m) + r] = temp;
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

void print_genes(thrust::host_vector<int> &genes, thrust::device_vector<int> &d_genes, int m, int n)
{
    thrust::copy(d_genes.begin(), d_genes.end(), genes.begin());

    cout << endl;

    for (int i = 0; i < n; i++) 
    {
        for (int j = 0; j < m; j++) 
        {
            printf("%d ", genes[i * m + j]);
        }

        printf("\n");
    }

    printf("\n");
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

// Tournament selection operator
__global__ void selection(float *fitness, city_t *cities, int *genes, int *offspring, float *random, int *selections, int m, int n)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid >= n) 
    {
        return;
    }

    int vs = random[tid] * n;

    selections[tid] = fitness[tid] > fitness[vs] ? tid : vs;
}

// PMX crossover operator
__global__ void crossover(int *genes, int *offspring, float *random, int *selections, int m, int n)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    tid *= 2;

    if (tid >= n) 
    {
        return;
    }

    // This kernel needs two random numbers, one for the random position of the crossover
    // and another for the crossover rate comparison

    int *parent1        = &genes[selections[tid] * m],
        *parent2        = &genes[selections[tid + 1] * m],
        *offspring1     = &offspring[tid * m],
        *offspring2     = &offspring[(tid + 1) * m];

    memcpy(offspring1, parent1, sizeof(int) * m);
    memcpy(offspring2, parent2, sizeof(int) * m);

    if (random[tid] > CROSSOVER_RATE)
    {
        return;
    }

    int crossover_point = random[tid + 1] * m;

    // memcpy(parent2, parent2 + m, offspring2);

    for (int i = 0; i <= crossover_point; i++)
    {
        // TODO: Optimize this N^2 search
        for (int j = 0; j < m; j++)
        {
            if (offspring1[j] == parent2[i])
            {
                int temp = offspring1[i];
                offspring1[i] = offspring1[j];
                offspring1[j] = temp;

                // swap(offspring1[i], offspring1[j]);
            }
        }
    }

    for (int i = 0; i <= crossover_point; i++)
    {
        // TODO: Optimize this N^2 search
        for (int j = 0; j < m; j++)
        {
            if (offspring2[j] == parent1[i])
            {
                int temp = offspring2[i];
                offspring2[i] = offspring2[j];
                offspring2[j] = temp;

                // swap(offspring2[i], offspring2[j]);
            }
        }
    }
}

__global__ void mutation(float *fitness, city_t *cities, int *offspring, float *random, int m, int n)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid >= n) 
    {
        return;
    }

    if (random[tid] > MUTATION_RATE)
    {
        return;
    }

    int *target = &offspring[tid * m],
        mutation_point1 = random[(tid + 1) % n],
        mutation_point2 = random[(tid + 2) % n],
        temp = target[mutation_point1];

    target[mutation_point1] = target[mutation_point2];
    target[mutation_point2] = temp;
}

__host__ __device__
unsigned int hash(unsigned int a)
{
    a = (a+0x7ed55d16) + (a<<12);
    a = (a^0xc761c23c) ^ (a>>19);
    a = (a+0x165667b1) + (a<<5);
    a = (a+0xd3a2646c) ^ (a<<9);
    a = (a+0xfd7046c5) + (a<<3);
    a = (a^0xb55a4f09) ^ (a>>16);
    return a;
}

struct RandomNumberFunctor : 
    public thrust::unary_function<unsigned int, float>
{
    unsigned int mainSeed;
    RandomNumberFunctor(unsigned int _mainSeed) : 
        mainSeed(_mainSeed) {}

    __host__ __device__
        float operator()(unsigned int threadIdx) 
    {
        unsigned int seed = hash(threadIdx) * mainSeed;

        // seed a random number generator
        thrust::default_random_engine rng(seed);

        // create a mapping from random numbers to [0,1)
        thrust::uniform_real_distribution<float> u01(0,1);

        return u01(rng);
    }
};

int main(int argc, char **argv)
{    
    cudaDeviceSynchronize(); 

    unsigned long seed = time(NULL);

    srand(seed);

    int m = 20,
        n = 1 << 12;

    // GPU genes data structure
    int *d_genes_raw, *d_offspring_raw, *d_selections_raw;
    float *d_fitness_raw, *d_random_raw;
    city_t *d_cities_raw;

    thrust::host_vector<int> genes(m * n);
    thrust::host_vector<int> offspring(m * n);
    thrust::host_vector<city_t> cities(n);
    thrust::host_vector<float> fitness(n);
    thrust::host_vector<float> random(n);
    thrust::host_vector<int> selections(n);

    thrust::device_vector<int> d_genes;
    thrust::device_vector<int> d_offspring;
    thrust::device_vector<city_t> d_cities;
    thrust::device_vector<float> d_fitness;
    thrust::device_vector<float> d_random(n);
    thrust::device_vector<int> d_selections(n);

    int blks = (n + NUM_THREADS - 1) / NUM_THREADS;

    initialize_genes(genes, m, n);
    initialize_cities(cities, m, n);

    d_genes = genes;
    d_offspring = offspring;
    d_cities = cities;
    d_fitness = fitness;

    // print_genes(genes, d_genes, m, n);

    for (int step = 0; step < 1000; step++)
    {
        d_fitness_raw = thrust::raw_pointer_cast(&d_fitness[0]);
        d_genes_raw = thrust::raw_pointer_cast(&d_genes[0]);
        d_cities_raw = thrust::raw_pointer_cast(&d_cities[0]);
        
        // Compute fitness for all genes
        compute_fitness <<< blks, NUM_THREADS >>> (d_fitness_raw, d_cities_raw, d_genes_raw, m, n);

        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );

        // Normalize fitness (sum of fitness = 1)
        float total_fitness = thrust::reduce(d_fitness.begin(), d_fitness.end());
        thrust::constant_iterator<float> normalization(total_fitness);
        thrust::transform(d_fitness.begin(), d_fitness.end(), normalization, d_fitness.begin(), thrust::divides<float>());

        // Selection
        d_offspring_raw = thrust::raw_pointer_cast(&d_offspring[0]);
        d_random_raw = thrust::raw_pointer_cast(&d_random[0]);
        d_selections_raw = thrust::raw_pointer_cast(&d_selections[0]);

        // Generate random numbers
        thrust::transform(thrust::counting_iterator<int>(0),
                          thrust::counting_iterator<int>(n),
                          d_random.begin(), RandomNumberFunctor(seed));

        selection <<< blks, NUM_THREADS >>> (d_fitness_raw, d_cities_raw, d_genes_raw, d_offspring_raw, d_random_raw, d_selections_raw, m, n);

        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );

        // Generate random numbers
        thrust::transform(thrust::counting_iterator<int>(0),
                          thrust::counting_iterator<int>(n),
                          d_random.begin(), RandomNumberFunctor(seed));
        
        crossover <<< blks, NUM_THREADS >>> (d_genes_raw, d_offspring_raw, d_random_raw, d_selections_raw, m, n);

        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );

        // Generate random numbers
        thrust::transform(thrust::counting_iterator<int>(0),
                          thrust::counting_iterator<int>(n),
                          d_random.begin(), RandomNumberFunctor(seed));
        
        // Mutation
        mutation <<< blks, NUM_THREADS >>> (d_fitness_raw, d_cities_raw, d_offspring_raw, d_random_raw, m, n);

        // TODO should swap instead
        // swap(d_genes, d_offspring);
        thrust::copy(d_offspring.begin(), d_offspring.end(), d_genes.begin());

        // print_genes(offspring, d_offspring, m, n);
        
        // thrust::device_vector<int> temp = d_genes;
        // d_genes = d_offspring;
        // d_offspring = d_genes;

        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );
    }

    compute_fitness <<< blks, NUM_THREADS >>> (d_fitness_raw, d_cities_raw, d_genes_raw, m, n);
    thrust::copy(d_fitness.begin(), d_fitness.end(), fitness.begin());
    // cout << endl;
    // for (int i = 0; i < n; i++)
    // {
    //     cout << fitness[i] << endl;
    // }

    int the_best = thrust::max_element(fitness.begin(), fitness.begin() + n) - fitness.begin();
    cout << the_best << " : " << fitness[the_best] << endl;

    thrust::copy(d_offspring.begin(), d_offspring.end(), offspring.begin());
    for (int i = 0; i < m; i++)
    {
        cout << offspring[the_best * m + i] << " ";
    }
    cout << endl;

    // print_genes(genes, d_genes, m, n);

    return 0;
}
