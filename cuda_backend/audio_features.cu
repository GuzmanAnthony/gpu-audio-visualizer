#include "audio_features.h"

#include <cuda_runtime.h>
#include <cufft.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <cmath>
#include <cstring>
#include <sstream>
#include <string>

namespace {

constexpr int kBlockSize = 256;
constexpr float kBassMaxHz = 250.0f;
constexpr float kMidMaxHz = 4000.0f;

inline void write_error(AudioFeatureResult* out, const std::string& message) {
    if (!out) {
        return;
    }
    std::snprintf(out->error, sizeof(out->error), "%s", message.c_str());
}

inline bool check_cuda(cudaError_t status, AudioFeatureResult* out, const char* operation) {
    if (status == cudaSuccess) {
        return true;
    }
    std::ostringstream oss;
    oss << operation << " failed: " << cudaGetErrorString(status);
    write_error(out, oss.str());
    return false;
}

inline bool check_cufft(cufftResult status, AudioFeatureResult* out, const char* operation) {
    if (status == CUFFT_SUCCESS) {
        return true;
    }
    std::ostringstream oss;
    oss << operation << " failed with cuFFT error code " << static_cast<int>(status);
    write_error(out, oss.str());
    return false;
}

void free_if_not_null(void* ptr) {
    if (ptr) {
        std::free(ptr);
    }
}

bool copy_from_pinned(float** dst, const float* src, int count) {
    if (!dst) {
        return false;
    }
    *dst = nullptr;
    if (count <= 0) {
        return true;
    }
    float* host = static_cast<float*>(std::malloc(static_cast<size_t>(count) * sizeof(float)));
    if (!host) {
        return false;
    }
    std::memcpy(host, src, static_cast<size_t>(count) * sizeof(float));
    *dst = host;
    return true;
}

__global__ void rms_peak_kernel(
    const float* audio,
    int num_samples,
    int frame_size,
    int hop_size,
    float* rms_out,
    float* peak_out
) {
    extern __shared__ float shared[];
    float* s_sum = shared;
    float* s_max = shared + blockDim.x;

    const int frame = blockIdx.x;
    const int tid = threadIdx.x;
    const int start = frame * hop_size;

    float local_sum = 0.0f;
    float local_max = 0.0f;

    for (int i = tid; i < frame_size; i += blockDim.x) {
        const int idx = start + i;
        const float value = (idx < num_samples) ? audio[idx] : 0.0f;
        local_sum += value * value;
        local_max = fmaxf(local_max, fabsf(value));
    }

    s_sum[tid] = local_sum;
    s_max[tid] = local_max;
    __syncthreads();

    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (tid < offset) {
            s_sum[tid] += s_sum[tid + offset];
            s_max[tid] = fmaxf(s_max[tid], s_max[tid + offset]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        rms_out[frame] = sqrtf(s_sum[0] / static_cast<float>(frame_size));
        peak_out[frame] = s_max[0];
    }
}

__global__ void hann_pack_kernel(
    const float* audio,
    int num_samples,
    int frame_size,
    int hop_size,
    cufftComplex* fft_in
) {
    const int frame = blockIdx.x;
    const int tid = threadIdx.x;
    const int start = frame * hop_size;
    cufftComplex* frame_out = fft_in + static_cast<size_t>(frame) * frame_size;

    for (int i = tid; i < frame_size; i += blockDim.x) {
        const int idx = start + i;
        const float sample = (idx < num_samples) ? audio[idx] : 0.0f;
        const float phase = (frame_size > 1)
            ? (2.0f * 3.14159265358979323846f * static_cast<float>(i) / static_cast<float>(frame_size - 1))
            : 0.0f;
        const float window = 0.5f - 0.5f * cosf(phase);
        frame_out[i].x = sample * window;
        frame_out[i].y = 0.0f;
    }
}

__global__ void band_energy_kernel(
    const cufftComplex* spectrum,
    int num_frames,
    int frame_size,
    int sample_rate,
    float* bass_out,
    float* mid_out,
    float* treble_out,
    float* centroid_out
) {
    __shared__ float s_bass[kBlockSize];
    __shared__ float s_mid[kBlockSize];
    __shared__ float s_treble[kBlockSize];
    __shared__ float s_weighted[kBlockSize];
    __shared__ float s_total[kBlockSize];

    const int frame = blockIdx.x;
    const int tid = threadIdx.x;
    if (frame >= num_frames) {
        return;
    }

    const cufftComplex* frame_spec = spectrum + static_cast<size_t>(frame) * frame_size;
    const int bins = frame_size / 2 + 1;

    float local_bass = 0.0f;
    float local_mid = 0.0f;
    float local_treble = 0.0f;
    float local_weighted = 0.0f;
    float local_total = 0.0f;

    for (int k = tid; k < bins; k += blockDim.x) {
        const float re = frame_spec[k].x;
        const float im = frame_spec[k].y;
        const float power = re * re + im * im;
        const float freq = static_cast<float>(k) * static_cast<float>(sample_rate) / static_cast<float>(frame_size);

        if (freq <= kBassMaxHz) {
            local_bass += power;
        } else if (freq <= kMidMaxHz) {
            local_mid += power;
        } else {
            local_treble += power;
        }

        local_weighted += freq * power;
        local_total += power;
    }

    s_bass[tid] = local_bass;
    s_mid[tid] = local_mid;
    s_treble[tid] = local_treble;
    s_weighted[tid] = local_weighted;
    s_total[tid] = local_total;
    __syncthreads();

    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (tid < offset) {
            s_bass[tid] += s_bass[tid + offset];
            s_mid[tid] += s_mid[tid + offset];
            s_treble[tid] += s_treble[tid + offset];
            s_weighted[tid] += s_weighted[tid + offset];
            s_total[tid] += s_total[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0) {
        bass_out[frame] = s_bass[0];
        mid_out[frame] = s_mid[0];
        treble_out[frame] = s_treble[0];
        centroid_out[frame] = (s_total[0] > 1e-12f) ? (s_weighted[0] / s_total[0]) : 0.0f;
    }
}

__global__ void waveform_minmax_kernel(
    const float* audio,
    int num_samples,
    int waveform_buckets,
    float* min_out,
    float* max_out
) {
    extern __shared__ float shared[];
    float* s_min = shared;
    float* s_max = shared + blockDim.x;

    const int bucket = blockIdx.x;
    const int tid = threadIdx.x;
    if (bucket >= waveform_buckets) {
        return;
    }

    int start = static_cast<int>((static_cast<long long>(bucket) * num_samples) / waveform_buckets);
    int end = static_cast<int>((static_cast<long long>(bucket + 1) * num_samples) / waveform_buckets);
    if (end <= start && start < num_samples) {
        end = start + 1;
    }

    float local_min = 1.0e20f;
    float local_max = -1.0e20f;

    if (start >= num_samples) {
        local_min = 0.0f;
        local_max = 0.0f;
    } else {
        for (int idx = start + tid; idx < end; idx += blockDim.x) {
            const float value = audio[idx];
            local_min = fminf(local_min, value);
            local_max = fmaxf(local_max, value);
        }
        if (local_min > 1.0e10f) {
            local_min = 0.0f;
        }
        if (local_max < -1.0e10f) {
            local_max = 0.0f;
        }
    }

    s_min[tid] = local_min;
    s_max[tid] = local_max;
    __syncthreads();

    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (tid < offset) {
            s_min[tid] = fminf(s_min[tid], s_min[tid + offset]);
            s_max[tid] = fmaxf(s_max[tid], s_max[tid + offset]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        min_out[bucket] = s_min[0];
        max_out[bucket] = s_max[0];
    }
}

}  // namespace

extern "C" int compute_audio_features_cuda(
    const float* audio,
    int num_samples,
    int sample_rate,
    int frame_size,
    int hop_size,
    int waveform_buckets,
    AudioFeatureResult* out_result
) {
    using clock = std::chrono::high_resolution_clock;
    const auto wall_begin = clock::now();
    auto gpu_begin = wall_begin;
    auto d2h_begin = wall_begin;

    if (!out_result) {
        return 1;
    }
    std::memset(out_result, 0, sizeof(AudioFeatureResult));

    if (!audio) {
        write_error(out_result, "Input audio pointer is null.");
        return 1;
    }
    if (num_samples <= 0 || sample_rate <= 0 || frame_size <= 0 || hop_size <= 0 || waveform_buckets <= 0) {
        write_error(out_result, "Arguments must all be positive.");
        return 1;
    }

    out_result->num_samples = num_samples;
    out_result->sample_rate = sample_rate;
    out_result->frame_size = frame_size;
    out_result->hop_size = hop_size;
    out_result->waveform_buckets = waveform_buckets;
    out_result->duration_seconds = static_cast<float>(num_samples) / static_cast<float>(sample_rate);
    out_result->num_frames = (num_samples >= frame_size) ? (1 + (num_samples - frame_size) / hop_size) : 0;

    float* h_audio_pinned = nullptr;
    float* h_rms_pinned = nullptr;
    float* h_peak_pinned = nullptr;
    float* h_bass_pinned = nullptr;
    float* h_mid_pinned = nullptr;
    float* h_treble_pinned = nullptr;
    float* h_centroid_pinned = nullptr;
    float* h_wave_min_pinned = nullptr;
    float* h_wave_max_pinned = nullptr;

    float* d_audio = nullptr;
    float* d_rms = nullptr;
    float* d_peak = nullptr;
    float* d_bass = nullptr;
    float* d_mid = nullptr;
    float* d_treble = nullptr;
    float* d_centroid = nullptr;
    float* d_wave_min = nullptr;
    float* d_wave_max = nullptr;
    cufftComplex* d_fft_in = nullptr;

    cufftHandle fft_plan = 0;
    cudaStream_t copy_stream = nullptr;
    cudaStream_t feature_stream = nullptr;
    cudaStream_t wave_stream = nullptr;

    cudaEvent_t h2d_start = nullptr;
    cudaEvent_t h2d_stop = nullptr;
    cudaEvent_t rms_start = nullptr;
    cudaEvent_t rms_stop = nullptr;
    cudaEvent_t pack_start = nullptr;
    cudaEvent_t pack_stop = nullptr;
    cudaEvent_t fft_start = nullptr;
    cudaEvent_t fft_stop = nullptr;
    cudaEvent_t band_start = nullptr;
    cudaEvent_t band_stop = nullptr;
    cudaEvent_t wave_start = nullptr;
    cudaEvent_t wave_stop = nullptr;
    cudaEvent_t copy_ready = nullptr;

    int status = 1;
    const int num_frames = out_result->num_frames;

    if (!check_cuda(cudaMallocHost(reinterpret_cast<void**>(&h_audio_pinned), static_cast<size_t>(num_samples) * sizeof(float)), out_result, "cudaMallocHost(input)")) {
        goto cleanup;
    }
    std::memcpy(h_audio_pinned, audio, static_cast<size_t>(num_samples) * sizeof(float));

    if (num_frames > 0) {
        if (!check_cuda(cudaMallocHost(reinterpret_cast<void**>(&h_rms_pinned), static_cast<size_t>(num_frames) * sizeof(float)), out_result, "cudaMallocHost(rms)")) {
            goto cleanup;
        }
        if (!check_cuda(cudaMallocHost(reinterpret_cast<void**>(&h_peak_pinned), static_cast<size_t>(num_frames) * sizeof(float)), out_result, "cudaMallocHost(peak)")) {
            goto cleanup;
        }
        if (!check_cuda(cudaMallocHost(reinterpret_cast<void**>(&h_bass_pinned), static_cast<size_t>(num_frames) * sizeof(float)), out_result, "cudaMallocHost(bass)")) {
            goto cleanup;
        }
        if (!check_cuda(cudaMallocHost(reinterpret_cast<void**>(&h_mid_pinned), static_cast<size_t>(num_frames) * sizeof(float)), out_result, "cudaMallocHost(mid)")) {
            goto cleanup;
        }
        if (!check_cuda(cudaMallocHost(reinterpret_cast<void**>(&h_treble_pinned), static_cast<size_t>(num_frames) * sizeof(float)), out_result, "cudaMallocHost(treble)")) {
            goto cleanup;
        }
        if (!check_cuda(cudaMallocHost(reinterpret_cast<void**>(&h_centroid_pinned), static_cast<size_t>(num_frames) * sizeof(float)), out_result, "cudaMallocHost(centroid)")) {
            goto cleanup;
        }
    }

    if (!check_cuda(cudaMallocHost(reinterpret_cast<void**>(&h_wave_min_pinned), static_cast<size_t>(waveform_buckets) * sizeof(float)), out_result, "cudaMallocHost(waveform_min)")) {
        goto cleanup;
    }
    if (!check_cuda(cudaMallocHost(reinterpret_cast<void**>(&h_wave_max_pinned), static_cast<size_t>(waveform_buckets) * sizeof(float)), out_result, "cudaMallocHost(waveform_max)")) {
        goto cleanup;
    }

    if (!check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_audio), static_cast<size_t>(num_samples) * sizeof(float)), out_result, "cudaMalloc(d_audio)")) {
        goto cleanup;
    }
    if (!check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_wave_min), static_cast<size_t>(waveform_buckets) * sizeof(float)), out_result, "cudaMalloc(d_wave_min)")) {
        goto cleanup;
    }
    if (!check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_wave_max), static_cast<size_t>(waveform_buckets) * sizeof(float)), out_result, "cudaMalloc(d_wave_max)")) {
        goto cleanup;
    }

    if (num_frames > 0) {
        if (!check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_rms), static_cast<size_t>(num_frames) * sizeof(float)), out_result, "cudaMalloc(d_rms)")) {
            goto cleanup;
        }
        if (!check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_peak), static_cast<size_t>(num_frames) * sizeof(float)), out_result, "cudaMalloc(d_peak)")) {
            goto cleanup;
        }
        if (!check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_bass), static_cast<size_t>(num_frames) * sizeof(float)), out_result, "cudaMalloc(d_bass)")) {
            goto cleanup;
        }
        if (!check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_mid), static_cast<size_t>(num_frames) * sizeof(float)), out_result, "cudaMalloc(d_mid)")) {
            goto cleanup;
        }
        if (!check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_treble), static_cast<size_t>(num_frames) * sizeof(float)), out_result, "cudaMalloc(d_treble)")) {
            goto cleanup;
        }
        if (!check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_centroid), static_cast<size_t>(num_frames) * sizeof(float)), out_result, "cudaMalloc(d_centroid)")) {
            goto cleanup;
        }
        if (!check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_fft_in), static_cast<size_t>(num_frames) * frame_size * sizeof(cufftComplex)), out_result, "cudaMalloc(d_fft_in)")) {
            goto cleanup;
        }
    }

    if (!check_cuda(cudaStreamCreate(&copy_stream), out_result, "cudaStreamCreate(copy_stream)")) {
        goto cleanup;
    }
    if (!check_cuda(cudaStreamCreate(&feature_stream), out_result, "cudaStreamCreate(feature_stream)")) {
        goto cleanup;
    }
    if (!check_cuda(cudaStreamCreate(&wave_stream), out_result, "cudaStreamCreate(wave_stream)")) {
        goto cleanup;
    }

    if (!check_cuda(cudaEventCreate(&h2d_start), out_result, "cudaEventCreate(h2d_start)")) goto cleanup;
    if (!check_cuda(cudaEventCreate(&h2d_stop), out_result, "cudaEventCreate(h2d_stop)")) goto cleanup;
    if (!check_cuda(cudaEventCreate(&rms_start), out_result, "cudaEventCreate(rms_start)")) goto cleanup;
    if (!check_cuda(cudaEventCreate(&rms_stop), out_result, "cudaEventCreate(rms_stop)")) goto cleanup;
    if (!check_cuda(cudaEventCreate(&pack_start), out_result, "cudaEventCreate(pack_start)")) goto cleanup;
    if (!check_cuda(cudaEventCreate(&pack_stop), out_result, "cudaEventCreate(pack_stop)")) goto cleanup;
    if (!check_cuda(cudaEventCreate(&fft_start), out_result, "cudaEventCreate(fft_start)")) goto cleanup;
    if (!check_cuda(cudaEventCreate(&fft_stop), out_result, "cudaEventCreate(fft_stop)")) goto cleanup;
    if (!check_cuda(cudaEventCreate(&band_start), out_result, "cudaEventCreate(band_start)")) goto cleanup;
    if (!check_cuda(cudaEventCreate(&band_stop), out_result, "cudaEventCreate(band_stop)")) goto cleanup;
    if (!check_cuda(cudaEventCreate(&wave_start), out_result, "cudaEventCreate(wave_start)")) goto cleanup;
    if (!check_cuda(cudaEventCreate(&wave_stop), out_result, "cudaEventCreate(wave_stop)")) goto cleanup;
    if (!check_cuda(cudaEventCreate(&copy_ready), out_result, "cudaEventCreate(copy_ready)")) goto cleanup;

    gpu_begin = clock::now();

    if (!check_cuda(cudaEventRecord(h2d_start, copy_stream), out_result, "cudaEventRecord(h2d_start)")) goto cleanup;
    if (!check_cuda(cudaMemcpyAsync(d_audio, h_audio_pinned, static_cast<size_t>(num_samples) * sizeof(float), cudaMemcpyHostToDevice, copy_stream), out_result, "cudaMemcpyAsync H2D")) {
        goto cleanup;
    }
    if (!check_cuda(cudaEventRecord(h2d_stop, copy_stream), out_result, "cudaEventRecord(h2d_stop)")) goto cleanup;
    if (!check_cuda(cudaEventRecord(copy_ready, copy_stream), out_result, "cudaEventRecord(copy_ready)")) goto cleanup;
    if (!check_cuda(cudaEventSynchronize(h2d_stop), out_result, "cudaEventSynchronize(h2d_stop)")) goto cleanup;
    if (!check_cuda(cudaEventElapsedTime(&out_result->h2d_ms, h2d_start, h2d_stop), out_result, "cudaEventElapsedTime(h2d)")) goto cleanup;

    if (!check_cuda(cudaStreamWaitEvent(feature_stream, copy_ready, 0), out_result, "cudaStreamWaitEvent(feature_stream)")) goto cleanup;
    if (!check_cuda(cudaStreamWaitEvent(wave_stream, copy_ready, 0), out_result, "cudaStreamWaitEvent(wave_stream)")) goto cleanup;

    if (!check_cuda(cudaEventRecord(wave_start, wave_stream), out_result, "cudaEventRecord(wave_start)")) goto cleanup;
    waveform_minmax_kernel<<<waveform_buckets, kBlockSize, static_cast<size_t>(2 * kBlockSize) * sizeof(float), wave_stream>>>(
        d_audio,
        num_samples,
        waveform_buckets,
        d_wave_min,
        d_wave_max
    );
    if (!check_cuda(cudaGetLastError(), out_result, "waveform_minmax_kernel launch")) goto cleanup;
    if (!check_cuda(cudaEventRecord(wave_stop, wave_stream), out_result, "cudaEventRecord(wave_stop)")) goto cleanup;

    if (num_frames > 0) {
        if (!check_cuda(cudaEventRecord(rms_start, feature_stream), out_result, "cudaEventRecord(rms_start)")) goto cleanup;
        rms_peak_kernel<<<num_frames, kBlockSize, static_cast<size_t>(2 * kBlockSize) * sizeof(float), feature_stream>>>(
            d_audio,
            num_samples,
            frame_size,
            hop_size,
            d_rms,
            d_peak
        );
        if (!check_cuda(cudaGetLastError(), out_result, "rms_peak_kernel launch")) goto cleanup;
        if (!check_cuda(cudaEventRecord(rms_stop, feature_stream), out_result, "cudaEventRecord(rms_stop)")) goto cleanup;

        if (!check_cuda(cudaEventRecord(pack_start, feature_stream), out_result, "cudaEventRecord(pack_start)")) goto cleanup;
        hann_pack_kernel<<<num_frames, kBlockSize, 0, feature_stream>>>(
            d_audio,
            num_samples,
            frame_size,
            hop_size,
            d_fft_in
        );
        if (!check_cuda(cudaGetLastError(), out_result, "hann_pack_kernel launch")) goto cleanup;
        if (!check_cuda(cudaEventRecord(pack_stop, feature_stream), out_result, "cudaEventRecord(pack_stop)")) goto cleanup;

        int rank = 1;
        int dims[1] = {frame_size};
        if (!check_cufft(cufftPlanMany(&fft_plan, rank, dims, dims, 1, frame_size, dims, 1, frame_size, CUFFT_C2C, num_frames), out_result, "cufftPlanMany")) {
            goto cleanup;
        }
        if (!check_cufft(cufftSetStream(fft_plan, feature_stream), out_result, "cufftSetStream")) {
            goto cleanup;
        }

        if (!check_cuda(cudaEventRecord(fft_start, feature_stream), out_result, "cudaEventRecord(fft_start)")) goto cleanup;
        if (!check_cufft(cufftExecC2C(fft_plan, d_fft_in, d_fft_in, CUFFT_FORWARD), out_result, "cufftExecC2C")) {
            goto cleanup;
        }
        if (!check_cuda(cudaEventRecord(fft_stop, feature_stream), out_result, "cudaEventRecord(fft_stop)")) goto cleanup;

        if (!check_cuda(cudaEventRecord(band_start, feature_stream), out_result, "cudaEventRecord(band_start)")) goto cleanup;
        band_energy_kernel<<<num_frames, kBlockSize, 0, feature_stream>>>(
            d_fft_in,
            num_frames,
            frame_size,
            sample_rate,
            d_bass,
            d_mid,
            d_treble,
            d_centroid
        );
        if (!check_cuda(cudaGetLastError(), out_result, "band_energy_kernel launch")) goto cleanup;
        if (!check_cuda(cudaEventRecord(band_stop, feature_stream), out_result, "cudaEventRecord(band_stop)")) goto cleanup;
    }

    d2h_begin = clock::now();

    if (num_frames > 0) {
        if (!check_cuda(cudaMemcpyAsync(h_rms_pinned, d_rms, static_cast<size_t>(num_frames) * sizeof(float), cudaMemcpyDeviceToHost, feature_stream), out_result, "cudaMemcpyAsync D2H rms")) goto cleanup;
        if (!check_cuda(cudaMemcpyAsync(h_peak_pinned, d_peak, static_cast<size_t>(num_frames) * sizeof(float), cudaMemcpyDeviceToHost, feature_stream), out_result, "cudaMemcpyAsync D2H peak")) goto cleanup;
        if (!check_cuda(cudaMemcpyAsync(h_bass_pinned, d_bass, static_cast<size_t>(num_frames) * sizeof(float), cudaMemcpyDeviceToHost, feature_stream), out_result, "cudaMemcpyAsync D2H bass")) goto cleanup;
        if (!check_cuda(cudaMemcpyAsync(h_mid_pinned, d_mid, static_cast<size_t>(num_frames) * sizeof(float), cudaMemcpyDeviceToHost, feature_stream), out_result, "cudaMemcpyAsync D2H mid")) goto cleanup;
        if (!check_cuda(cudaMemcpyAsync(h_treble_pinned, d_treble, static_cast<size_t>(num_frames) * sizeof(float), cudaMemcpyDeviceToHost, feature_stream), out_result, "cudaMemcpyAsync D2H treble")) goto cleanup;
        if (!check_cuda(cudaMemcpyAsync(h_centroid_pinned, d_centroid, static_cast<size_t>(num_frames) * sizeof(float), cudaMemcpyDeviceToHost, feature_stream), out_result, "cudaMemcpyAsync D2H centroid")) goto cleanup;
    }
    if (!check_cuda(cudaMemcpyAsync(h_wave_min_pinned, d_wave_min, static_cast<size_t>(waveform_buckets) * sizeof(float), cudaMemcpyDeviceToHost, wave_stream), out_result, "cudaMemcpyAsync D2H waveform_min")) goto cleanup;
    if (!check_cuda(cudaMemcpyAsync(h_wave_max_pinned, d_wave_max, static_cast<size_t>(waveform_buckets) * sizeof(float), cudaMemcpyDeviceToHost, wave_stream), out_result, "cudaMemcpyAsync D2H waveform_max")) goto cleanup;

    if (!check_cuda(cudaStreamSynchronize(feature_stream), out_result, "cudaStreamSynchronize(feature_stream)")) goto cleanup;
    if (!check_cuda(cudaStreamSynchronize(wave_stream), out_result, "cudaStreamSynchronize(wave_stream)")) goto cleanup;
    out_result->d2h_ms = static_cast<float>(std::chrono::duration<double, std::milli>(clock::now() - d2h_begin).count());

    if (num_frames > 0) {
        if (!check_cuda(cudaEventElapsedTime(&out_result->rms_peak_ms, rms_start, rms_stop), out_result, "cudaEventElapsedTime(rms_peak)")) goto cleanup;
        if (!check_cuda(cudaEventElapsedTime(&out_result->window_pack_ms, pack_start, pack_stop), out_result, "cudaEventElapsedTime(window_pack)")) goto cleanup;
        if (!check_cuda(cudaEventElapsedTime(&out_result->fft_ms, fft_start, fft_stop), out_result, "cudaEventElapsedTime(fft)")) goto cleanup;
        if (!check_cuda(cudaEventElapsedTime(&out_result->band_energy_ms, band_start, band_stop), out_result, "cudaEventElapsedTime(band_energy)")) goto cleanup;
    }
    if (!check_cuda(cudaEventElapsedTime(&out_result->waveform_minmax_ms, wave_start, wave_stop), out_result, "cudaEventElapsedTime(waveform_minmax)")) goto cleanup;

    out_result->total_gpu_ms = static_cast<float>(std::chrono::duration<double, std::milli>(clock::now() - gpu_begin).count());

    if (!copy_from_pinned(&out_result->waveform_min, h_wave_min_pinned, waveform_buckets)) {
        write_error(out_result, "Failed to allocate waveform_min host output.");
        goto cleanup;
    }
    if (!copy_from_pinned(&out_result->waveform_max, h_wave_max_pinned, waveform_buckets)) {
        write_error(out_result, "Failed to allocate waveform_max host output.");
        goto cleanup;
    }

    if (num_frames > 0) {
        if (!copy_from_pinned(&out_result->rms, h_rms_pinned, num_frames)) {
            write_error(out_result, "Failed to allocate rms host output.");
            goto cleanup;
        }
        if (!copy_from_pinned(&out_result->peak, h_peak_pinned, num_frames)) {
            write_error(out_result, "Failed to allocate peak host output.");
            goto cleanup;
        }
        if (!copy_from_pinned(&out_result->bass, h_bass_pinned, num_frames)) {
            write_error(out_result, "Failed to allocate bass host output.");
            goto cleanup;
        }
        if (!copy_from_pinned(&out_result->mid, h_mid_pinned, num_frames)) {
            write_error(out_result, "Failed to allocate mid host output.");
            goto cleanup;
        }
        if (!copy_from_pinned(&out_result->treble, h_treble_pinned, num_frames)) {
            write_error(out_result, "Failed to allocate treble host output.");
            goto cleanup;
        }
        if (!copy_from_pinned(&out_result->centroid, h_centroid_pinned, num_frames)) {
            write_error(out_result, "Failed to allocate centroid host output.");
            goto cleanup;
        }
    }

    status = 0;

cleanup:
    out_result->total_wall_ms = static_cast<float>(std::chrono::duration<double, std::milli>(clock::now() - wall_begin).count());

    if (fft_plan) {
        cufftDestroy(fft_plan);
    }

    if (h2d_start) cudaEventDestroy(h2d_start);
    if (h2d_stop) cudaEventDestroy(h2d_stop);
    if (rms_start) cudaEventDestroy(rms_start);
    if (rms_stop) cudaEventDestroy(rms_stop);
    if (pack_start) cudaEventDestroy(pack_start);
    if (pack_stop) cudaEventDestroy(pack_stop);
    if (fft_start) cudaEventDestroy(fft_start);
    if (fft_stop) cudaEventDestroy(fft_stop);
    if (band_start) cudaEventDestroy(band_start);
    if (band_stop) cudaEventDestroy(band_stop);
    if (wave_start) cudaEventDestroy(wave_start);
    if (wave_stop) cudaEventDestroy(wave_stop);
    if (copy_ready) cudaEventDestroy(copy_ready);

    if (copy_stream) cudaStreamDestroy(copy_stream);
    if (feature_stream) cudaStreamDestroy(feature_stream);
    if (wave_stream) cudaStreamDestroy(wave_stream);

    if (d_audio) cudaFree(d_audio);
    if (d_rms) cudaFree(d_rms);
    if (d_peak) cudaFree(d_peak);
    if (d_bass) cudaFree(d_bass);
    if (d_mid) cudaFree(d_mid);
    if (d_treble) cudaFree(d_treble);
    if (d_centroid) cudaFree(d_centroid);
    if (d_wave_min) cudaFree(d_wave_min);
    if (d_wave_max) cudaFree(d_wave_max);
    if (d_fft_in) cudaFree(d_fft_in);

    if (h_audio_pinned) cudaFreeHost(h_audio_pinned);
    if (h_rms_pinned) cudaFreeHost(h_rms_pinned);
    if (h_peak_pinned) cudaFreeHost(h_peak_pinned);
    if (h_bass_pinned) cudaFreeHost(h_bass_pinned);
    if (h_mid_pinned) cudaFreeHost(h_mid_pinned);
    if (h_treble_pinned) cudaFreeHost(h_treble_pinned);
    if (h_centroid_pinned) cudaFreeHost(h_centroid_pinned);
    if (h_wave_min_pinned) cudaFreeHost(h_wave_min_pinned);
    if (h_wave_max_pinned) cudaFreeHost(h_wave_max_pinned);

    return status;
}

extern "C" void free_audio_feature_result(AudioFeatureResult* result) {
    if (!result) {
        return;
    }

    free_if_not_null(result->rms);
    free_if_not_null(result->peak);
    free_if_not_null(result->bass);
    free_if_not_null(result->mid);
    free_if_not_null(result->treble);
    free_if_not_null(result->centroid);
    free_if_not_null(result->waveform_min);
    free_if_not_null(result->waveform_max);

    result->rms = nullptr;
    result->peak = nullptr;
    result->bass = nullptr;
    result->mid = nullptr;
    result->treble = nullptr;
    result->centroid = nullptr;
    result->waveform_min = nullptr;
    result->waveform_max = nullptr;
}
